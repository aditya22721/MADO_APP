"""
Unit Tests for Voice_Finetuning DataManager.
"""

import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch

from Voice_Finetuning.config import PHASE_1_RECORDING_THRESHOLD
from Voice_Finetuning.data_manager import (
    DataManager,
    ReadinessStatus,
    RecordingValidationResult,
)
from Voice_Recognition.speaker_preprocessing import SpeakerAudioPreprocessor
from Voice_Recognition.voice_profile_manager import VoiceProfileManager

TEST_DATA_DIR = Path("data/test_finetuning_data")


def create_test_wav(
    path: Path,
    duration_sec: float = 3.5,
    sample_rate: int = 16000,
    frequency: float = 220.0,
    amplitude: float = 0.5,
    clipping: bool = False,
    silent: bool = False,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)

    if silent:
        data = np.zeros(num_samples, dtype=np.float32)
    elif clipping:
        # Create heavily clipped square wave
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        data = np.sign(np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    else:
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        data = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    sf.write(str(path), data, sample_rate)


class TestDataManager(unittest.TestCase):

    def setUp(self):
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.profiles_dir = TEST_DATA_DIR / "voice_profiles"
        self.transcripts_dir = TEST_DATA_DIR / "text"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = SpeakerAudioPreprocessor()
        self.profile_manager = VoiceProfileManager(profiles_directory=str(self.profiles_dir))
        self.data_manager = DataManager(
            profiles_directory=self.profiles_dir,
            transcripts_directory=self.transcripts_dir,
            preprocessor=self.preprocessor,
            profile_manager=self.profile_manager,
        )

    def tearDown(self):
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)

    def test_valid_recording_validation(self):
        wav_path = TEST_DATA_DIR / "good.wav"
        create_test_wav(wav_path, duration_sec=4.0, amplitude=0.6)

        result = self.data_manager.validate_recording(wav_path)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.rejection_reasons), 0)
        self.assertGreater(result.duration_seconds, 3.0)
        self.assertGreater(result.usable_speech_seconds, 2.0)

    def test_short_recording_rejection(self):
        wav_path = TEST_DATA_DIR / "too_short.wav"
        create_test_wav(wav_path, duration_sec=1.0)

        result = self.data_manager.validate_recording(wav_path)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("below minimum threshold" in r for r in result.rejection_reasons))

    def test_silent_recording_rejection(self):
        wav_path = TEST_DATA_DIR / "silent.wav"
        create_test_wav(wav_path, duration_sec=4.0, silent=True)

        result = self.data_manager.validate_recording(wav_path)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Excessive silence" in r or "Usable speech duration" in r for r in result.rejection_reasons))

    def test_clipped_recording_rejection(self):
        wav_path = TEST_DATA_DIR / "clipped.wav"
        create_test_wav(wav_path, duration_sec=4.0, clipping=True)

        result = self.data_manager.validate_recording(wav_path)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("clipping" in r.lower() for r in result.rejection_reasons))

    def test_duplicate_detection(self):
        wav1 = TEST_DATA_DIR / "sample1.wav"
        wav2 = TEST_DATA_DIR / "sample2.wav"
        create_test_wav(wav1, duration_sec=4.0)
        shutil.copy(wav1, wav2)

        known = set()
        res1 = self.data_manager.validate_recording(wav1, known_hashes=known)
        self.assertTrue(res1.is_valid)
        known.add(res1.file_hash)

        res2 = self.data_manager.validate_recording(wav2, known_hashes=known)
        self.assertFalse(res2.is_valid)
        self.assertTrue(any("Duplicate" in r for r in res2.rejection_reasons))

    def test_threshold_readiness_status(self):
        user_id = "test_user_001"
        user_audio_dir = self.profiles_dir / user_id / "audio"
        user_audio_dir.mkdir(parents=True, exist_ok=True)

        # Create first 10 dummy files
        for i in range(10):
            p = user_audio_dir / f"rec_{i:03d}.wav"
            p.write_bytes(b"RIFF" + bytes([i]) * 100)

        # Mock validate_recording for threshold counting
        from Voice_Finetuning.utils import compute_file_hash
        self.data_manager.validate_recording = lambda path, known_hashes=None: RecordingValidationResult(
            audio_path=Path(path),
            is_valid=True,
            duration_seconds=3.5,
            usable_speech_seconds=3.5,
            file_hash=compute_file_hash(path)
        )

        # 10 recordings (< 50)
        status_10 = self.data_manager.get_user_data_status(user_id)
        self.assertEqual(status_10.valid_recordings_count, 10)
        self.assertEqual(status_10.readiness_status, ReadinessStatus.NOT_READY)
        self.assertEqual(status_10.required_threshold, PHASE_1_RECORDING_THRESHOLD)

        # Add 40 more dummy files (total 50)
        for i in range(10, 50):
            p = user_audio_dir / f"rec_{i:03d}.wav"
            p.write_bytes(b"RIFF" + bytes([i]) * 100)

        status_50 = self.data_manager.get_user_data_status(user_id)
        self.assertEqual(status_50.valid_recordings_count, 50)
        self.assertEqual(status_50.readiness_status, ReadinessStatus.READY_PHASE_1)


if __name__ == "__main__":
    unittest.main()
