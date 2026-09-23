"""
Dataset Builder for XTTS-v2 Personalization.

Responsibilities:
    - Takes validated user recordings from DataManager
    - Ensures each audio sample has an accurate transcription
      (reusing Faster-Whisper via speech_to_text if missing)
    - Resamples and standardizes audio to 22,050 Hz mono (XTTS target format)
    - Organizes files into standard dataset structure:
        datasets/{user_id}/
            wavs/
            metadata.csv
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import soundfile as sf
import torch
import torchaudio

from Voice_Input.speech_to_text import speech_to_text

from .config import DATASETS_DIRECTORY, XTTS_SAMPLE_RATE
from .data_manager import RecordingValidationResult
from .utils import normalize_text


@dataclass
class DatasetSample:
    """Single sample representation within an XTTS fine-tuning dataset."""
    audio_rel_path: str      # e.g., "wavs/recording_001.wav"
    audio_abs_path: Path     # Absolute path to the standardized WAV file
    text: str                # Transcribed text
    speaker_name: str        # User ID / speaker identifier
    duration_seconds: float  # Audio duration


class DatasetBuilder:
    """
    Builds and packages an XTTS-v2 fine-tuning dataset for a specific user.
    """

    def __init__(
        self,
        datasets_base_dir: Union[str, Path] = DATASETS_DIRECTORY,
        target_sample_rate: int = XTTS_SAMPLE_RATE,
        stt_func: Optional[Callable[[Union[str, Path]], Path]] = None,
    ):
        self.datasets_base_dir = Path(datasets_base_dir)
        self.target_sample_rate = target_sample_rate
        self.stt_func = stt_func or speech_to_text

    def get_user_dataset_dir(self, user_id: str) -> Path:
        """Return root dataset directory for a specific user."""
        return self.datasets_base_dir / user_id

    def standardize_audio(
        self,
        src_path: Path,
        dst_path: Path
    ) -> float:
        """
        Standardize audio to mono float32 at target XTTS sample rate (22,050 Hz).
        Returns final audio duration in seconds.
        """
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        audio_data, sr = sf.read(str(src_path), dtype="float32")

        waveform = torch.from_numpy(audio_data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to XTTS sample rate (22050 Hz)
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)

        waveform_np = waveform.squeeze(0).numpy()
        sf.write(str(dst_path), waveform_np, self.target_sample_rate)

        duration = len(waveform_np) / self.target_sample_rate
        return float(duration)

    def get_or_create_transcript(
        self,
        audio_path: Path,
        cached_text: Optional[str] = None
    ) -> str:
        """
        Get transcription from cache or transcribe using Faster-Whisper.
        """
        if cached_text and cached_text.strip():
            return cached_text.strip()

        # Run Faster-Whisper via speech_to_text
        transcript_path = self.stt_func(audio_path)
        text = Path(transcript_path).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Transcription for {audio_path} produced empty text.")
        return text

    def build_dataset(
        self,
        user_id: str,
        valid_recordings: List[RecordingValidationResult]
    ) -> List[DatasetSample]:
        """
        Build the standardized dataset for the given user and recordings.
        """
        dataset_dir = self.get_user_dataset_dir(user_id)
        wavs_dir = dataset_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        dataset_samples: List[DatasetSample] = []

        for idx, rec in enumerate(valid_recordings, start=1):
            src_audio = rec.audio_path
            dst_name = f"{src_audio.stem}_norm_{idx:04d}.wav"
            dst_audio = wavs_dir / dst_name

            duration = self.standardize_audio(src_audio, dst_audio)
            text = self.get_or_create_transcript(src_audio, rec.transcript_text)

            sample = DatasetSample(
                audio_rel_path=f"wavs/{dst_name}",
                audio_abs_path=dst_audio,
                text=text,
                speaker_name=user_id,
                duration_seconds=duration,
            )
            dataset_samples.append(sample)

        return dataset_samples
