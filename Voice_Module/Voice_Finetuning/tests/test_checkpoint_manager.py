"""
Unit Tests for Voice_Finetuning CheckpointManager.
"""

import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Voice_Finetuning.checkpoint_manager import CheckpointManager
from Voice_Finetuning.model_selector import ModelDecision, SelectionResult

TEST_CHK_DIR = Path("data/test_finetuning_checkpoints")


class TestCheckpointManager(unittest.TestCase):

    def setUp(self):
        if TEST_CHK_DIR.exists():
            shutil.rmtree(TEST_CHK_DIR, ignore_errors=True)
        TEST_CHK_DIR.mkdir(parents=True, exist_ok=True)
        self.mgr = CheckpointManager(checkpoints_base_dir=TEST_CHK_DIR)

    def tearDown(self):
        if TEST_CHK_DIR.exists():
            shutil.rmtree(TEST_CHK_DIR, ignore_errors=True)

    def test_default_unpromoted_state(self):
        user_id = "user_chk_001"
        state = self.mgr.load_active_model_state(user_id)
        self.assertEqual(state.primary_model, "zero_shot")
        self.assertEqual(state.fallback_model, "zero_shot")
        self.assertIsNone(state.active_round)

    def test_record_rejected_round(self):
        user_id = "user_chk_001"
        round_dir = self.mgr.get_round_dir(user_id, 1)

        sel = SelectionResult(
            user_id=user_id,
            decision=ModelDecision.REJECTED,
            passed_minimum_criteria=False,
            meaningfully_better_than_zero_shot=False,
            rejection_reasons=["Failed quality"],
            candidate_speaker_similarity=0.68,
            zero_shot_speaker_similarity=0.74,
            speaker_similarity_delta=-0.06,
            candidate_wer=0.15,
            zero_shot_wer=0.12,
            wer_delta=0.03,
            candidate_quality_score=0.65,
            zero_shot_quality_score=0.85,
        )

        self.mgr.record_round(
            user_id=user_id,
            round_number=1,
            selection_result=sel,
            train_count=40,
            val_count=10,
            train_duration=120.0,
            val_duration=30.0,
            round_dir=round_dir,
        )

        state = self.mgr.load_active_model_state(user_id)
        self.assertEqual(state.primary_model, "zero_shot")  # Remains zero_shot
        self.assertEqual(self.mgr.get_latest_round_number(user_id), 1)

    def test_record_promoted_round_and_rollback(self):
        user_id = "user_chk_002"
        round_dir = self.mgr.get_round_dir(user_id, 1)

        best_model = round_dir / "best_model.pth"
        config_f = round_dir / "config.json"
        vocab_f = round_dir / "vocab.json"
        ref_f = round_dir / "speaker_reference.wav"

        best_model.write_text("dummy", encoding="utf-8")
        config_f.write_text("{}", encoding="utf-8")
        vocab_f.write_text("{}", encoding="utf-8")
        ref_f.write_text("dummy", encoding="utf-8")

        sel = SelectionResult(
            user_id=user_id,
            decision=ModelDecision.PROMOTED,
            passed_minimum_criteria=True,
            meaningfully_better_than_zero_shot=True,
            candidate_speaker_similarity=0.82,
            zero_shot_speaker_similarity=0.75,
            speaker_similarity_delta=+0.07,
            candidate_wer=0.08,
            zero_shot_wer=0.11,
            wer_delta=-0.03,
            candidate_quality_score=0.90,
            zero_shot_quality_score=0.85,
        )

        self.mgr.record_round(
            user_id=user_id,
            round_number=1,
            selection_result=sel,
            train_count=40,
            val_count=10,
            train_duration=120.0,
            val_duration=30.0,
            round_dir=round_dir,
            model_checkpoint_path=best_model,
            config_path=config_f,
            vocab_path=vocab_f,
            speaker_reference_path=ref_f,
        )

        state = self.mgr.load_active_model_state(user_id)
        self.assertEqual(state.primary_model, "fine_tuned")
        self.assertEqual(state.active_round, 1)
        self.assertEqual(state.checkpoint_path, str(best_model.resolve()))

        # Rollback
        reverted = self.mgr.rollback_to_zero_shot(user_id)
        self.assertEqual(reverted.primary_model, "zero_shot")


if __name__ == "__main__":
    unittest.main()
