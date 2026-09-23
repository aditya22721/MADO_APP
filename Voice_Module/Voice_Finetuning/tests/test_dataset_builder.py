"""
Unit Tests for Voice_Finetuning DatasetBuilder.
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

from Voice_Finetuning.data_manager import RecordingValidationResult
from Voice_Finetuning.dataset_builder import DatasetBuilder

TEST_DS_DIR = Path("data/test_finetuning_builder_data")


def create_sample_wav(path: Path, duration_sec: float = 3.0, sample_rate: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    data = 0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, duration_sec, num_samples, endpoint=False)).astype(np.float32)
    sf.write(str(path), data, sample_rate)


class TestDatasetBuilder(unittest.TestCase):

    def setUp(self):
        if TEST_DS_DIR.exists():
            shutil.rmtree(TEST_DS_DIR, ignore_errors=True)
        TEST_DS_DIR.mkdir(parents=True, exist_ok=True)

        def mock_stt(audio_path):
            txt_file = TEST_DS_DIR / f"{Path(audio_path).stem}.txt"
            txt_file.write_text(f"Transcribed content for {Path(audio_path).name}", encoding="utf-8")
            return txt_file

        self.builder = DatasetBuilder(
            datasets_base_dir=TEST_DS_DIR,
            target_sample_rate=22050,
            stt_func=mock_stt
        )

    def tearDown(self):
        if TEST_DS_DIR.exists():
            shutil.rmtree(TEST_DS_DIR, ignore_errors=True)

    def test_standardize_audio(self):
        src = TEST_DS_DIR / "orig_16k.wav"
        dst = TEST_DS_DIR / "std_22k.wav"
        create_sample_wav(src, duration_sec=3.0, sample_rate=16000)

        dur = self.builder.standardize_audio(src, dst)
        self.assertTrue(dst.exists())

        data, sr = sf.read(str(dst))
        self.assertEqual(sr, 22050)
        self.assertAlmostEqual(dur, 3.0, delta=0.1)

    def test_build_dataset_with_transcripts(self):
        user_id = "user_build_test"
        recs = []

        for i in range(3):
            p = TEST_DS_DIR / f"raw_audio_{i}.wav"
            create_sample_wav(p, duration_sec=3.0)
            rec = RecordingValidationResult(
                audio_path=p,
                is_valid=True,
                duration_seconds=3.0,
                usable_speech_seconds=3.0,
                transcript_text=f"Preset transcript {i}" if i == 0 else None
            )
            recs.append(rec)

        samples = self.builder.build_dataset(user_id=user_id, valid_recordings=recs)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0].text, "Preset transcript 0")
        self.assertIn("Transcribed content", samples[1].text)
        self.assertTrue(samples[0].audio_abs_path.exists())


if __name__ == "__main__":
    unittest.main()
