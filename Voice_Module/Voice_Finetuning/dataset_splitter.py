"""
Deterministic Train / Validation Dataset Splitter for Voice_Finetuning.

Guarantees:
    1. Deterministic partitioning based on configured seed
    2. Zero overlap between training and validation recordings
    3. Strict preservation of validation recordings across retraining rounds
    4. Outputs pipe-delimited XTTS metadata CSV files:
       `audio_file|text|speaker_name`
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from .config import (
    DEFAULT_VAL_SPLIT_RATIO,
    PHASE_1_TRAIN_COUNT,
    PHASE_1_VAL_COUNT,
    SPLIT_RANDOM_SEED,
)
from .dataset_builder import DatasetSample


@dataclass
class DatasetSplit:
    """Container for train/validation partitions and metadata paths."""
    train_samples: List[DatasetSample]
    val_samples: List[DatasetSample]
    train_csv_path: Path
    val_csv_path: Path
    total_train_duration_sec: float
    total_val_duration_sec: float


class DatasetSplitter:
    """
    Performs deterministic train/validation split with validation preservation.
    """

    def __init__(
        self,
        seed: int = SPLIT_RANDOM_SEED,
        val_ratio: float = DEFAULT_VAL_SPLIT_RATIO,
    ):
        self.seed = seed
        self.val_ratio = val_ratio

    def split_samples(
        self,
        samples: List[DatasetSample],
        target_val_count: Optional[int] = None,
        fixed_val_audio_names: Optional[Set[str]] = None
    ) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """
        Partition dataset samples into train and validation sets.

        If `fixed_val_audio_names` is provided (e.g. from previous training rounds),
        those samples are guaranteed to remain in the validation set.
        """
        if not samples:
            return [], []

        # Sort deterministically by relative audio path first
        sorted_samples = sorted(samples, key=lambda s: s.audio_rel_path)

        # If previous validation set exists, preserve those exact samples
        if fixed_val_audio_names is not None and len(fixed_val_audio_names) > 0:
            val_samples = [s for s in sorted_samples if s.audio_rel_path in fixed_val_audio_names]
            train_samples = [s for s in sorted_samples if s.audio_rel_path not in fixed_val_audio_names]
            return train_samples, val_samples

        total_count = len(sorted_samples)

        # Determine number of validation samples
        if target_val_count is not None:
            num_val = min(target_val_count, total_count // 2)
        elif total_count == 50:
            num_val = PHASE_1_VAL_COUNT
        else:
            num_val = max(1, int(round(total_count * self.val_ratio)))

        # Deterministic shuffle
        rng = random.Random(self.seed)
        shuffled = list(sorted_samples)
        rng.shuffle(shuffled)

        val_samples = sorted(shuffled[:num_val], key=lambda s: s.audio_rel_path)
        train_samples = sorted(shuffled[num_val:], key=lambda s: s.audio_rel_path)

        # Verify zero overlap
        train_rel = {s.audio_rel_path for s in train_samples}
        val_rel = {s.audio_rel_path for s in val_samples}
        overlap = train_rel.intersection(val_rel)
        if overlap:
            raise ValueError(f"Fatal error: train/validation split has overlapping samples: {overlap}")

        return train_samples, val_samples

    def export_metadata_csv(
        self,
        samples: List[DatasetSample],
        output_csv_path: Path
    ) -> Path:
        """
        Write samples to XTTS pipe-delimited CSV format:
        audio_file|text|speaker_name
        """
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["audio_file", "text", "speaker_name"])
            for sample in samples:
                # XTTS expects relative path like wavs/audio.wav or absolute path
                writer.writerow([sample.audio_rel_path, sample.text, sample.speaker_name])

        return output_csv_path

    def prepare_dataset_split(
        self,
        user_id: str,
        samples: List[DatasetSample],
        dataset_dir: Path,
        fixed_val_audio_names: Optional[Set[str]] = None
    ) -> DatasetSplit:
        """
        Split samples and generate metadata_train.csv and metadata_eval.csv.
        """
        train_samples, val_samples = self.split_samples(
            samples,
            fixed_val_audio_names=fixed_val_audio_names
        )

        train_csv = dataset_dir / "metadata_train.csv"
        val_csv = dataset_dir / "metadata_eval.csv"

        self.export_metadata_csv(train_samples, train_csv)
        self.export_metadata_csv(val_samples, val_csv)

        train_duration = sum(s.duration_seconds for s in train_samples)
        val_duration = sum(s.duration_seconds for s in val_samples)

        return DatasetSplit(
            train_samples=train_samples,
            val_samples=val_samples,
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            total_train_duration_sec=float(train_duration),
            total_val_duration_sec=float(val_duration),
        )
