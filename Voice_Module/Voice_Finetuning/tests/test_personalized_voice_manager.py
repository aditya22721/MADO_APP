"""
Unit Tests for PersonalizedVoiceManager and Zero-Shot Fallback.
"""

import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Voice_Finetuning.checkpoint_manager import ActiveModelState, CheckpointManager
from Voice_Finetuning.personalized_voice_manager import PersonalizedVoiceManager

TEST_PVM_DIR = Path("data/test_pvm_data")


class TestPersonalizedVoiceManager(unittest.TestCase):

    def setUp(self):
        if TEST_PVM_DIR.exists():
            shutil.rmtree(TEST_PVM_DIR, ignore_errors=True)
        TEST_PVM_DIR.mkdir(parents=True, exist_ok=True)

        self.mock_chk_mgr = MagicMock(spec=CheckpointManager)
        self.mock_zs_mgr = MagicMock()
        self.mock_trainer = MagicMock()

        self.pvm = PersonalizedVoiceManager(
            checkpoint_manager=self.mock_chk_mgr,
            zero_shot_manager=self.mock_zs_mgr,
            trainer=self.mock_trainer,
        )

    def tearDown(self):
        if TEST_PVM_DIR.exists():
            shutil.rmtree(TEST_PVM_DIR, ignore_errors=True)

    def test_default_routes_to_zero_shot(self):
        user_id = "user_001"
        self.mock_chk_mgr.load_active_model_state.return_value = ActiveModelState(
            user_id=user_id,
            primary_model="zero_shot",
            fallback_model="zero_shot",
        )

        dummy_out = TEST_PVM_DIR / "out.wav"
        self.mock_zs_mgr.generate_for_user.return_value = dummy_out
        self.mock_zs_mgr.get_default_output_path.return_value = dummy_out

        result = self.pvm.generate_for_user(user_id, "Test text")

        self.mock_zs_mgr.generate_for_user.assert_called_once()
        self.mock_trainer.generate_fine_tuned_speech.assert_not_called()
        self.assertEqual(result, dummy_out)

    def test_promoted_routes_to_fine_tuned(self):
        user_id = "user_002"
        chk_path = TEST_PVM_DIR / "best_model.pth"
        cfg_path = TEST_PVM_DIR / "config.json"
        voc_path = TEST_PVM_DIR / "vocab.json"
        ref_path = TEST_PVM_DIR / "ref.wav"

        chk_path.write_text("x", encoding="utf-8")
        cfg_path.write_text("{}", encoding="utf-8")
        voc_path.write_text("{}", encoding="utf-8")
        ref_path.write_text("x", encoding="utf-8")

        self.mock_chk_mgr.load_active_model_state.return_value = ActiveModelState(
            user_id=user_id,
            primary_model="fine_tuned",
            fallback_model="zero_shot",
            active_round=1,
            checkpoint_path=str(chk_path),
            config_path=str(cfg_path),
            vocab_path=str(voc_path),
            speaker_reference_path=str(ref_path),
        )

        dummy_out = TEST_PVM_DIR / "ft_out.wav"
        self.mock_trainer.generate_fine_tuned_speech.return_value = dummy_out
        self.mock_zs_mgr.get_default_output_path.return_value = dummy_out

        result = self.pvm.generate_for_user(user_id, "Test speech")

        self.mock_trainer.generate_fine_tuned_speech.assert_called_once()
        self.mock_zs_mgr.generate_for_user.assert_not_called()
        self.assertEqual(result, dummy_out)

    def test_fine_tuned_failure_triggers_automatic_fallback(self):
        user_id = "user_003"
        chk_path = TEST_PVM_DIR / "best_model.pth"
        cfg_path = TEST_PVM_DIR / "config.json"
        voc_path = TEST_PVM_DIR / "vocab.json"
        ref_path = TEST_PVM_DIR / "ref.wav"

        chk_path.write_text("x", encoding="utf-8")
        cfg_path.write_text("{}", encoding="utf-8")
        voc_path.write_text("{}", encoding="utf-8")
        ref_path.write_text("x", encoding="utf-8")

        self.mock_chk_mgr.load_active_model_state.return_value = ActiveModelState(
            user_id=user_id,
            primary_model="fine_tuned",
            fallback_model="zero_shot",
            active_round=1,
            checkpoint_path=str(chk_path),
            config_path=str(cfg_path),
            vocab_path=str(voc_path),
            speaker_reference_path=str(ref_path),
        )

        dummy_out = TEST_PVM_DIR / "fallback_out.wav"
        self.mock_trainer.generate_fine_tuned_speech.side_effect = RuntimeError("GPU CUDA OOM")
        self.mock_zs_mgr.generate_for_user.return_value = dummy_out
        self.mock_zs_mgr.get_default_output_path.return_value = dummy_out

        # Must NOT raise exception; should automatically fallback to zero-shot
        result = self.pvm.generate_for_user(user_id, "Test text")

        self.mock_trainer.generate_fine_tuned_speech.assert_called_once()
        self.mock_zs_mgr.generate_for_user.assert_called_once()
        self.assertEqual(result, dummy_out)


if __name__ == "__main__":
    unittest.main()
