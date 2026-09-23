"""
Unit Tests for Voice_Finetuning DatasetSplitter.
"""

import csv
import random
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Voice_Finetuning.dataset_builder import DatasetSample
from Voice_Finetuning.dataset_splitter import DatasetSplitter

TEST_SPLIT_DIR = Path("data/test_finetuning_split_data")


class TestDatasetSplitter(unittest.TestCase):

    def setUp(self):
        if TEST_SPLIT_DIR.exists():
            shutil.rmtree(TEST_SPLIT_DIR, ignore_errors=True)
        TEST_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        self.splitter = DatasetSplitter(seed=42)

    def tearDown(self):
        if TEST_SPLIT_DIR.exists():
            shutil.rmtree(TEST_SPLIT_DIR, ignore_errors=True)

    def test_fifty_recordings_split(self):
        # Create 50 dummy samples
        samples = [
            DatasetSample(
                audio_rel_path=f"wavs/sample_{i:03d}.wav",
                audio_abs_path=TEST_SPLIT_DIR / f"sample_{i:03d}.wav",
                text=f"Text for sentence {i}",
                speaker_name="test_user",
                duration_seconds=3.0,
            )
            for i in range(50)
        ]

        train_s, val_s = self.splitter.split_samples(samples)

        self.assertEqual(len(train_s), 40)
        self.assertEqual(len(val_s), 10)

        # Zero overlap check
        train_rel = {s.audio_rel_path for s in train_s}
        val_rel = {s.audio_rel_path for s in val_s}
        self.assertEqual(len(train_rel.intersection(val_rel)), 0)

    def test_deterministic_split(self):
        samples = [
            DatasetSample(
                audio_rel_path=f"wavs/sample_{i:03d}.wav",
                audio_abs_path=TEST_SPLIT_DIR / f"sample_{i:03d}.wav",
                text=f"Text {i}",
                speaker_name="test_user",
                duration_seconds=3.0,
            )
            for i in range(50)
        ]

        splitter_a = DatasetSplitter(seed=42)
        splitter_b = DatasetSplitter(seed=42)

        train_a, val_a = splitter_a.split_samples(samples)
        train_b, val_b = splitter_b.split_samples(samples)

        self.assertEqual([s.audio_rel_path for s in train_a], [s.audio_rel_path for s in train_b])
        self.assertEqual([s.audio_rel_path for s in val_a], [s.audio_rel_path for s in val_b])

    def test_preservation_of_validation_set_across_rounds(self):
        # Round 1: 50 recordings
        round_1_samples = [
            DatasetSample(
                audio_rel_path=f"wavs/sample_{i:03d}.wav",
                audio_abs_path=TEST_SPLIT_DIR / f"sample_{i:03d}.wav",
                text=f"Text {i}",
                speaker_name="test_user",
                duration_seconds=3.0,
            )
            for i in range(50)
        ]
        train_1, val_1 = self.splitter.split_samples(round_1_samples)
        val_1_keys = {s.audio_rel_path for s in val_1}

        # Round 2: Add 25 new recordings (total 75)
        round_2_samples = list(round_1_samples) + [
            DatasetSample(
                audio_rel_path=f"wavs/sample_{i:03d}.wav",
                audio_abs_path=TEST_SPLIT_DIR / f"sample_{i:03d}.wav",
                text=f"Text {i}",
                speaker_name="test_user",
                duration_seconds=3.0,
            )
            for i in range(50, 75)
        ]

        # Split preserving Round 1 validation set
        train_2, val_2 = self.splitter.split_samples(
            round_2_samples,
            fixed_val_audio_names=val_1_keys
        )

        self.assertEqual(len(val_2), 10)
        self.assertEqual(len(train_2), 65)
        self.assertEqual({s.audio_rel_path for s in val_2}, val_1_keys)


if __name__ == "__main__":
    unittest.main()
