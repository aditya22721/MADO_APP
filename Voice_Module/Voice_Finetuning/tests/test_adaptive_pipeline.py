"""
Unit & Integration Tests for AdaptiveFineTuningPipeline.
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
from Voice_Finetuning.data_manager import (
    DataManager,
    ReadinessStatus,
    RecordingValidationResult,
    UserDataStatus,
)
from Voice_Finetuning.dataset_builder import DatasetBuilder, DatasetSample
from Voice_Finetuning.dataset_splitter import DatasetSplit, DatasetSplitter
from Voice_Finetuning.evaluator import (
    ComparativeEvaluationReport,
    Evaluator,
    ModelEvaluationReport,
)
from Voice_Finetuning.model_selector import ModelDecision, ModelSelector, SelectionResult
from Voice_Finetuning.pipeline import AdaptiveFineTuningPipeline
from Voice_Finetuning.trainer import TrainingResult, XTTSTrainer

TEST_PIPE_DIR = Path("data/test_pipeline_data")


class TestAdaptivePipeline(unittest.TestCase):

    def setUp(self):
        if TEST_PIPE_DIR.exists():
            shutil.rmtree(TEST_PIPE_DIR, ignore_errors=True)
        TEST_PIPE_DIR.mkdir(parents=True, exist_ok=True)

        self.mock_data_mgr = MagicMock(spec=DataManager)
        self.mock_ds_builder = MagicMock(spec=DatasetBuilder)
        self.mock_ds_splitter = MagicMock(spec=DatasetSplitter)
        self.mock_evaluator = MagicMock(spec=Evaluator)
        self.mock_model_selector = MagicMock(spec=ModelSelector)
        self.mock_chk_mgr = MagicMock(spec=CheckpointManager)
        self.mock_trainer = MagicMock(spec=XTTSTrainer)
        self.mock_zs_mgr = MagicMock()

        self.pipeline = AdaptiveFineTuningPipeline(
            data_manager=self.mock_data_mgr,
            dataset_builder=self.mock_ds_builder,
            dataset_splitter=self.mock_ds_splitter,
            evaluator=self.mock_evaluator,
            model_selector=self.mock_model_selector,
            checkpoint_manager=self.mock_chk_mgr,
            trainer=self.mock_trainer,
            zero_shot_manager=self.mock_zs_mgr,
        )

    def tearDown(self):
        if TEST_PIPE_DIR.exists():
            shutil.rmtree(TEST_PIPE_DIR, ignore_errors=True)

    def test_pipeline_stops_when_not_ready(self):
        user_id = "user_pipeline_001"
        self.mock_chk_mgr.get_latest_round_number.return_value = 0
        self.mock_chk_mgr.load_active_model_state.return_value = ActiveModelState(
            user_id=user_id,
            primary_model="zero_shot",
            fallback_model="zero_shot",
        )

        self.mock_data_mgr.get_user_data_status.return_value = UserDataStatus(
            user_id=user_id,
            total_recordings_found=15,
            valid_recordings_count=15,
            invalid_recordings_count=0,
            duplicate_recordings_count=0,
            total_usable_duration_seconds=60.0,
            total_usable_duration_minutes=1.0,
            required_threshold=50,
            readiness_status=ReadinessStatus.NOT_READY,
        )

        res = self.pipeline.run_for_user(user_id)

        self.assertFalse(res.executed_training)
        self.assertEqual(res.active_primary_engine, "zero_shot")
        self.assertIn("Fine-tuning not triggered", res.message)
        self.mock_trainer.train_user_model.assert_not_called()

    def test_pipeline_executes_when_ready(self):
        user_id = "user_pipeline_002"
        self.mock_chk_mgr.get_latest_round_number.return_value = 0
        self.mock_chk_mgr.get_round_dir.return_value = TEST_PIPE_DIR / user_id / "round_01"
        self.mock_chk_mgr.load_active_model_state.return_value = ActiveModelState(
            user_id=user_id,
            primary_model="fine_tuned",
            fallback_model="zero_shot",
        )

        # 50 recordings ready
        valid_recs = [
            RecordingValidationResult(audio_path=Path(f"a_{i}.wav"), is_valid=True)
            for i in range(50)
        ]
        self.mock_data_mgr.get_user_data_status.return_value = UserDataStatus(
            user_id=user_id,
            total_recordings_found=50,
            valid_recordings_count=50,
            invalid_recordings_count=0,
            duplicate_recordings_count=0,
            total_usable_duration_seconds=300.0,
            total_usable_duration_minutes=5.0,
            required_threshold=50,
            readiness_status=ReadinessStatus.READY_PHASE_1,
            valid_recordings=valid_recs,
        )

        # Mock dataset building and splitting
        samples = [DatasetSample(f"wavs/{i}.wav", Path(f"wavs/{i}.wav"), f"t{i}", user_id, 3.0) for i in range(50)]
        self.mock_ds_builder.build_dataset.return_value = samples
        self.mock_ds_builder.get_user_dataset_dir.return_value = TEST_PIPE_DIR / user_id / "dataset"
        self.mock_ds_splitter.prepare_dataset_split.return_value = DatasetSplit(
            train_samples=samples[:40],
            val_samples=samples[40:],
            train_csv_path=TEST_PIPE_DIR / "train.csv",
            val_csv_path=TEST_PIPE_DIR / "eval.csv",
            total_train_duration_sec=120.0,
            total_val_duration_sec=30.0,
        )

        # Mock Zero-Shot and Candidate reports
        zs_report = ModelEvaluationReport("zero_shot", user_id, 8, 0.74, 0.12, 0.04, 0.88, 0.85, 0.0, 0.1)
        cand_report = ModelEvaluationReport("fine_tuned", user_id, 8, 0.80, 0.09, 0.03, 0.91, 0.88, 0.0, 0.1)
        comp_report = ComparativeEvaluationReport(user_id, zs_report, cand_report, 0.06, -0.03, 0.03, True)

        self.mock_evaluator.evaluate_model_generations.side_effect = [zs_report, cand_report]
        self.mock_evaluator.compare_reports.return_value = comp_report

        # Mock training result
        self.mock_trainer.train_user_model.return_value = TrainingResult(
            user_id=user_id,
            round_number=1,
            output_dir=TEST_PIPE_DIR / "round_01",
            best_model_path=TEST_PIPE_DIR / "round_01" / "best_model.pth",
            config_path=TEST_PIPE_DIR / "round_01" / "config.json",
            vocab_path=TEST_PIPE_DIR / "round_01" / "vocab.json",
            speaker_reference_path=TEST_PIPE_DIR / "round_01" / "ref.wav",
            training_success=True,
        )

        # Mock selector decision
        sel_res = SelectionResult(
            user_id=user_id,
            decision=ModelDecision.PROMOTED,
            passed_minimum_criteria=True,
            meaningfully_better_than_zero_shot=True,
            candidate_speaker_similarity=0.80,
            zero_shot_speaker_similarity=0.74,
            speaker_similarity_delta=0.06,
            candidate_wer=0.09,
            zero_shot_wer=0.12,
            wer_delta=-0.03,
            candidate_quality_score=0.88,
            zero_shot_quality_score=0.85,
            next_action="Promote fine-tuned checkpoint",
        )
        self.mock_model_selector.evaluate_decision.return_value = sel_res

        custom_val_texts = [{"id": f"val_{i}", "category": "test", "text": f"Sentence {i}"} for i in range(2)]

        res = self.pipeline.run_for_user(user_id, custom_val_texts=custom_val_texts)

        self.assertTrue(res.executed_training)
        self.assertEqual(res.decision, "PROMOTED")
        self.assertEqual(res.active_primary_engine, "fine_tuned")
        self.mock_trainer.train_user_model.assert_called_once()
        self.mock_chk_mgr.record_round.assert_called_once()


if __name__ == "__main__":
    unittest.main()
