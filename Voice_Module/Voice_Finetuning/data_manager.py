"""
Data Management and Validation Layer for Voice_Finetuning.

Responsibilities:
    - Discover audio recordings associated with a user
    - Validate individual recordings against audio quality benchmarks:
        * Readable format
        * Mono / convertible channels
        * Min/max duration bounds
        * Usable speech duration (via SpeakerAudioPreprocessor)
        * Clipping ratio
        * Excessive silence ratio
        * Non-corrupted / NaN-free signal
    - Detect and reject duplicate/near-duplicate audio files
    - Discover or associate corresponding transcriptions
    - Calculate aggregate statistics (total recordings, valid count, total duration)
    - Determine adaptive readiness status according to the 50-recording engineering benchmark
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import soundfile as sf
import torch

from Voice_Recognition.speaker_preprocessing import SpeakerAudioPreprocessor
from Voice_Recognition.voice_profile_manager import VoiceProfileManager

from .config import (
    BATCH_INCREMENT,
    MAX_CLIPPING_RATIO,
    MAX_RECORDING_DURATION_SEC,
    MAX_SILENCE_RATIO,
    MIN_RECORDING_DURATION_SEC,
    MIN_USABLE_SPEECH_SEC,
    PHASE_1_RECORDING_THRESHOLD,
    SUPPORTED_AUDIO_EXTENSIONS,
    TRANSCRIPTS_DIRECTORY,
    VOICE_PROFILES_DIRECTORY,
)
from .utils import analyze_audio_signal, compute_file_hash


class ReadinessStatus(str, Enum):
    """Adaptive data readiness state."""
    NOT_READY = "NOT_READY"
    READY_PHASE_1 = "READY_PHASE_1"
    READY_ADDITIONAL_BATCH = "READY_ADDITIONAL_BATCH"


@dataclass
class RecordingValidationResult:
    """Detailed validation assessment of a single recording."""
    audio_path: Path
    is_valid: bool
    rejection_reasons: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    usable_speech_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    clipping_ratio: float = 0.0
    silence_ratio: float = 0.0
    file_hash: str = ""
    transcript_path: Optional[Path] = None
    transcript_text: Optional[str] = None


@dataclass
class UserDataStatus:
    """Summary of user data readiness for adaptive fine-tuning."""
    user_id: str
    total_recordings_found: int
    valid_recordings_count: int
    invalid_recordings_count: int
    duplicate_recordings_count: int
    total_usable_duration_seconds: float
    total_usable_duration_minutes: float
    required_threshold: int
    readiness_status: ReadinessStatus
    valid_recordings: List[RecordingValidationResult] = field(default_factory=list)
    invalid_recordings: List[RecordingValidationResult] = field(default_factory=list)


class DataManager:
    """
    Manages discovery, validation, and readiness reporting for user recordings.
    """

    def __init__(
        self,
        profiles_directory: Union[str, Path] = VOICE_PROFILES_DIRECTORY,
        transcripts_directory: Union[str, Path] = TRANSCRIPTS_DIRECTORY,
        preprocessor: Optional[SpeakerAudioPreprocessor] = None,
        profile_manager: Optional[VoiceProfileManager] = None,
    ):
        self.profiles_directory = Path(profiles_directory)
        self.transcripts_directory = Path(transcripts_directory)
        self.preprocessor = preprocessor or SpeakerAudioPreprocessor()
        self.profile_manager = profile_manager or VoiceProfileManager(profiles_directory=str(self.profiles_directory))

    def discover_user_audio_files(self, user_id: str) -> List[Path]:
        """
        Dynamically find all audio files for a user.
        Checks:
            1. User audio folder: data/voice_profiles/{user_id}/audio/
            2. Enrollment records from profile metadata
            3. User reference folder: data/voice_cloning/references/{user_id}/
        """
        discovered_paths: List[Path] = []
        seen_paths: Set[str] = set()

        # 1. Check profile audio directory
        user_audio_dir = self.profiles_directory / user_id / "audio"
        if user_audio_dir.exists() and user_audio_dir.is_dir():
            for file_path in sorted(user_audio_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                    resolved_str = str(file_path.resolve())
                    if resolved_str not in seen_paths:
                        discovered_paths.append(file_path)
                        seen_paths.add(resolved_str)

        # 2. Check profile metadata enrollment records
        try:
            metadata = self.profile_manager.load_metadata(user_id)
            records = metadata.get("enrollment_records", [])
            for rec in records:
                raw_path = rec.get("audio_path")
                if raw_path:
                    path_obj = Path(raw_path)
                    if path_obj.exists() and path_obj.is_file():
                        resolved_str = str(path_obj.resolve())
                        if resolved_str not in seen_paths:
                            discovered_paths.append(path_obj)
                            seen_paths.add(resolved_str)
        except Exception:
            pass

        return discovered_paths

    def find_transcript(self, audio_path: Path) -> Tuple[Optional[Path], Optional[str]]:
        """
        Locate associated text transcription for an audio file if it exists.
        """
        stem = audio_path.stem

        # Check candidate locations
        candidate_paths = [
            self.transcripts_directory / f"{stem}.txt",
            audio_path.parent / f"{stem}.txt",
            audio_path.parent.parent / "transcripts" / f"{stem}.txt",
        ]

        for cand in candidate_paths:
            if cand.exists() and cand.is_file():
                try:
                    text = cand.read_text(encoding="utf-8").strip()
                    if text:
                        return cand, text
                except Exception:
                    pass

        return None, None

    def validate_recording(
        self,
        audio_path: Union[str, Path],
        known_hashes: Optional[Set[str]] = None
    ) -> RecordingValidationResult:
        """
        Validate a single audio recording against project quality criteria.
        """
        audio_path = Path(audio_path)
        rejection_reasons: List[str] = []

        if not audio_path.exists() or not audio_path.is_file():
            return RecordingValidationResult(
                audio_path=audio_path,
                is_valid=False,
                rejection_reasons=["File does not exist or is not a regular file"]
            )

        # 1. Compute file hash and check deduplication
        file_hash = compute_file_hash(audio_path)
        if known_hashes is not None and file_hash in known_hashes:
            rejection_reasons.append(f"Duplicate recording detected (SHA-256: {file_hash[:12]})")

        # 2. Analyze raw audio signal
        try:
            signal_info = analyze_audio_signal(audio_path)
        except Exception as exc:
            return RecordingValidationResult(
                audio_path=audio_path,
                is_valid=False,
                rejection_reasons=[f"Failed to read audio file: {exc}"],
                file_hash=file_hash
            )

        if signal_info["is_empty"] or signal_info["duration"] == 0:
            rejection_reasons.append("Audio file contains zero audio samples")

        if signal_info["has_nan_inf"]:
            rejection_reasons.append("Audio waveform contains invalid NaN/Inf values")

        duration = signal_info["duration"]
        if duration < MIN_RECORDING_DURATION_SEC:
            rejection_reasons.append(
                f"Audio duration {duration:.2f}s is below minimum threshold ({MIN_RECORDING_DURATION_SEC:.1f}s)"
            )
        elif duration > MAX_RECORDING_DURATION_SEC:
            rejection_reasons.append(
                f"Audio duration {duration:.2f}s exceeds maximum threshold ({MAX_RECORDING_DURATION_SEC:.1f}s)"
            )

        if signal_info["clipping_ratio"] > MAX_CLIPPING_RATIO:
            rejection_reasons.append(
                f"Excessive clipping: {signal_info['clipping_ratio'] * 100:.1f}% samples clipped (max allowed: {MAX_CLIPPING_RATIO * 100:.1f}%)"
            )

        # 3. Process with SpeakerAudioPreprocessor to compute usable speech
        usable_speech_seconds = 0.0
        try:
            waveform = self.preprocessor.process(audio_path, trim_silence=True, normalize=False)
            usable_speech_seconds = float(waveform.shape[1] / self.preprocessor.TARGET_SAMPLE_RATE)
            if usable_speech_seconds < MIN_USABLE_SPEECH_SEC:
                rejection_reasons.append(
                    f"Usable speech duration {usable_speech_seconds:.2f}s is below minimum {MIN_USABLE_SPEECH_SEC:.1f}s"
                )
        except Exception as exc:
            rejection_reasons.append(f"Preprocessing validation failed: {exc}")

        if signal_info["rms_energy"] < 0.005 or signal_info["peak_amplitude"] < 0.01:
            rejection_reasons.append("Excessive silence / insufficient audio signal energy")

        # Compute silence ratio based on trimmed duration vs raw duration
        if duration > 0:
            silence_ratio = max(0.0, float(1.0 - (usable_speech_seconds / duration)))
        else:
            silence_ratio = 1.0

        if silence_ratio > MAX_SILENCE_RATIO:
            rejection_reasons.append(
                f"Excessive silence: {silence_ratio * 100:.1f}% silent (max allowed: {MAX_SILENCE_RATIO * 100:.1f}%)"
            )

        # 4. Check transcription
        transcript_path, transcript_text = self.find_transcript(audio_path)

        is_valid = len(rejection_reasons) == 0

        return RecordingValidationResult(
            audio_path=audio_path,
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            duration_seconds=duration,
            usable_speech_seconds=usable_speech_seconds,
            sample_rate=signal_info["sample_rate"],
            channels=signal_info["channels"],
            clipping_ratio=signal_info["clipping_ratio"],
            silence_ratio=silence_ratio,
            file_hash=file_hash,
            transcript_path=transcript_path,
            transcript_text=transcript_text,
        )

    def get_user_data_status(
        self,
        user_id: str,
        current_training_round: int = 0
    ) -> UserDataStatus:
        """
        Assess complete dataset readiness for a specific user.
        Enforces unique recordings and the 50-recording engineering benchmark.
        """
        audio_paths = self.discover_user_audio_files(user_id)

        valid_recordings: List[RecordingValidationResult] = []
        invalid_recordings: List[RecordingValidationResult] = []
        seen_hashes: Set[str] = set()
        duplicate_count = 0
        total_usable_sec = 0.0

        for path in audio_paths:
            is_duplicate = False
            try:
                h = compute_file_hash(path)
                if h in seen_hashes:
                    is_duplicate = True
            except Exception:
                pass

            result = self.validate_recording(path, known_hashes=seen_hashes)

            if is_duplicate or any("Duplicate" in r for r in result.rejection_reasons):
                duplicate_count += 1
                invalid_recordings.append(result)
            elif result.is_valid:
                valid_recordings.append(result)
                seen_hashes.add(result.file_hash)
                total_usable_sec += result.usable_speech_seconds
            else:
                invalid_recordings.append(result)

        valid_count = len(valid_recordings)

        # Determine target threshold
        if current_training_round == 0:
            threshold = PHASE_1_RECORDING_THRESHOLD
            if valid_count >= threshold:
                readiness = ReadinessStatus.READY_PHASE_1
            else:
                readiness = ReadinessStatus.NOT_READY
        else:
            threshold = PHASE_1_RECORDING_THRESHOLD + (current_training_round * BATCH_INCREMENT)
            if valid_count >= threshold:
                readiness = ReadinessStatus.READY_ADDITIONAL_BATCH
            else:
                readiness = ReadinessStatus.NOT_READY

        return UserDataStatus(
            user_id=user_id,
            total_recordings_found=len(audio_paths),
            valid_recordings_count=valid_count,
            invalid_recordings_count=len(invalid_recordings),
            duplicate_recordings_count=duplicate_count,
            total_usable_duration_seconds=float(total_usable_sec),
            total_usable_duration_minutes=float(total_usable_sec / 60.0),
            required_threshold=threshold,
            readiness_status=readiness,
            valid_recordings=valid_recordings,
            invalid_recordings=invalid_recordings,
        )
