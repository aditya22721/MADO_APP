"""
Adaptive Fine-Tuning Pipeline for CARE DOLL Voice System.

Orchestrates the complete adaptive fine-tuning lifecycle:
    1. Data Quality & Threshold Readiness Check (50-recording engineering benchmark)
    2. Dataset Standardization & Deterministic Train/Eval Partitioning
    3. Zero-Shot Baseline Generation & Multi-Signal Evaluation
    4. Safe Candidate XTTS Fine-Tuning Run
    5. Candidate Model Generation & Multi-Signal Evaluation
    6. Comparative Model Promotion / Rejection Decision
    7. Per-User Checkpoint Versioning & Metadata Logging
"""

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from Voice_Cloning.voice_cloning_manager import VoiceCloningManager

from .checkpoint_manager import CheckpointManager
from .config import (
    LOGS_DIRECTORY,
    RESULTS_DIRECTORY,
    create_required_directories,
)
from .data_manager import DataManager, ReadinessStatus, UserDataStatus
from .dataset_builder import DatasetBuilder
from .dataset_splitter import DatasetSplitter
from .evaluator import Evaluator
from .model_selector import ModelDecision, ModelSelector, SelectionResult
from .trainer import XTTSTrainer
from .validation_texts import get_validation_texts


@dataclass
class PipelineRunResult:
    """Summary of an adaptive fine-tuning pipeline execution."""
    user_id: str
    round_number: int
    data_status: UserDataStatus
    executed_training: bool
    selection_result: Optional[SelectionResult] = None
    zero_shot_speaker_similarity: Optional[float] = None
    candidate_speaker_similarity: Optional[float] = None
    zero_shot_wer: Optional[float] = None
    candidate_wer: Optional[float] = None
    decision: Optional[str] = None
    active_primary_engine: str = "zero_shot"
    log_file_path: Optional[str] = None
    message: str = ""


class AdaptiveFineTuningPipeline:
    """
    End-to-end adaptive personalization coordinator.
    """

    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        dataset_builder: Optional[DatasetBuilder] = None,
        dataset_splitter: Optional[DatasetSplitter] = None,
        evaluator: Optional[Evaluator] = None,
        model_selector: Optional[ModelSelector] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        trainer: Optional[XTTSTrainer] = None,
        zero_shot_manager: Optional[VoiceCloningManager] = None,
    ):
        create_required_directories()

        self.data_manager = data_manager or DataManager()
        self.dataset_builder = dataset_builder or DatasetBuilder()
        self.dataset_splitter = dataset_splitter or DatasetSplitter()
        self.evaluator = evaluator or Evaluator()
        self.model_selector = model_selector or ModelSelector()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.trainer = trainer or XTTSTrainer()
        self.zero_shot_manager = zero_shot_manager or VoiceCloningManager()

    def run_for_user(
        self,
        user_id: str,
        custom_val_texts: Optional[List[Dict[str, str]]] = None,
        dry_run: bool = False,
    ) -> PipelineRunResult:
        """
        Execute adaptive personalization cycle for a specific user.
        """
        print("=" * 70)
        print(f"ADAPTIVE XTTS-v2 FINE-TUNING PIPELINE: USER '{user_id}'")
        print("=" * 70)

        # 1. Determine current round & check data readiness
        current_round = self.checkpoint_manager.get_latest_round_number(user_id)
        next_round = current_round + 1

        data_status = self.data_manager.get_user_data_status(
            user_id=user_id,
            current_training_round=current_round
        )

        print(f"Total recordings found:     {data_status.total_recordings_found}")
        print(f"Valid recordings count:     {data_status.valid_recordings_count}")
        print(f"Required threshold:         {data_status.required_threshold}")
        print(f"Total usable speech:        {data_status.total_usable_duration_minutes:.2f} minutes")
        print(f"Readiness status:           {data_status.readiness_status.value}")

        active_state = self.checkpoint_manager.load_active_model_state(user_id)

        # -------------------------------------------------------------
        # 2. If NOT READY: Return immediately, preserve zero-shot
        # -------------------------------------------------------------
        if data_status.readiness_status == ReadinessStatus.NOT_READY:
            msg = (
                f"User '{user_id}' has {data_status.valid_recordings_count} valid recordings "
                f"(requires {data_status.required_threshold} for round {next_round}). "
                f"Fine-tuning not triggered. Zero-Shot XTTS remains active."
            )
            print(f"\n[PIPELINE STATUS] {msg}\n")
            return PipelineRunResult(
                user_id=user_id,
                round_number=current_round,
                data_status=data_status,
                executed_training=False,
                active_primary_engine=active_state.primary_model,
                message=msg,
            )

        if dry_run:
            msg = f"User '{user_id}' is READY for round {next_round}. Dry run specified, skipping training."
            print(f"\n[PIPELINE STATUS] {msg}\n")
            return PipelineRunResult(
                user_id=user_id,
                round_number=next_round,
                data_status=data_status,
                executed_training=False,
                active_primary_engine=active_state.primary_model,
                message=msg,
            )

        # -------------------------------------------------------------
        # 3. Build & Partition Dataset
        # -------------------------------------------------------------
        print(f"\n[Step 1/5] Building standardized dataset for {user_id}...")
        dataset_samples = self.dataset_builder.build_dataset(
            user_id=user_id,
            valid_recordings=data_status.valid_recordings
        )

        user_ds_dir = self.dataset_builder.get_user_dataset_dir(user_id)
        split = self.dataset_splitter.prepare_dataset_split(
            user_id=user_id,
            samples=dataset_samples,
            dataset_dir=user_ds_dir
        )
        print(f"Train samples: {len(split.train_samples)} ({split.total_train_duration_sec:.1f}s)")
        print(f"Val samples:   {len(split.val_samples)} ({split.total_val_duration_sec:.1f}s)")

        # -------------------------------------------------------------
        # 4. Zero-Shot Baseline Evaluation
        # -------------------------------------------------------------
        print(f"\n[Step 2/5] Generating Zero-Shot baseline validation audio...")
        val_texts = get_validation_texts(custom_val_texts)
        zs_eval_dir = RESULTS_DIRECTORY / user_id / f"round_{next_round:02d}" / "zero_shot_val_audio"
        zs_eval_dir.mkdir(parents=True, exist_ok=True)

        zs_audio_paths: Dict[str, Path] = {}
        for item in val_texts:
            s_id = item["id"]
            out_file = zs_eval_dir / f"{s_id}.wav"
            self.zero_shot_manager.generate_for_user(
                user_id=user_id,
                text=item["text"],
                output_path=out_file
            )
            zs_audio_paths[s_id] = out_file

        print(f"Evaluating Zero-Shot baseline on {len(val_texts)} sentences...")
        zs_report = self.evaluator.evaluate_model_generations(
            model_type="zero_shot",
            user_id=user_id,
            validation_items=val_texts,
            generated_audio_paths=zs_audio_paths,
        )
        print(f"Zero-Shot Speaker Similarity: {zs_report.mean_speaker_similarity:.3f}")
        print(f"Zero-Shot WER:                {zs_report.mean_wer:.3f}")
        print(f"Zero-Shot Quality Score:      {zs_report.mean_quality_score:.3f}")

        # -------------------------------------------------------------
        # 5. Train Candidate Fine-Tuned Model
        # -------------------------------------------------------------
        print(f"\n[Step 3/5] Starting XTTS GPT fine-tuning (Round {next_round})...")
        round_dir = self.checkpoint_manager.get_round_dir(user_id, next_round)

        training_result = self.trainer.train_user_model(
            user_id=user_id,
            round_number=next_round,
            train_csv_path=split.train_csv_path,
            eval_csv_path=split.val_csv_path,
            output_round_dir=round_dir
        )

        if not training_result.training_success:
            err_msg = f"Training failed during round {next_round}: {training_result.error_message}"
            print(f"[ERROR] {err_msg}")
            return PipelineRunResult(
                user_id=user_id,
                round_number=next_round,
                data_status=data_status,
                executed_training=True,
                active_primary_engine=active_state.primary_model,
                message=err_msg,
            )

        # -------------------------------------------------------------
        # 6. Candidate Model Evaluation
        # -------------------------------------------------------------
        print(f"\n[Step 4/5] Generating Candidate fine-tuned validation audio...")
        cand_eval_dir = RESULTS_DIRECTORY / user_id / f"round_{next_round:02d}" / "candidate_val_audio"
        cand_eval_dir.mkdir(parents=True, exist_ok=True)

        cand_audio_paths: Dict[str, Path] = {}
        for item in val_texts:
            s_id = item["id"]
            out_file = cand_eval_dir / f"{s_id}.wav"
            self.trainer.generate_fine_tuned_speech(
                text=item["text"],
                reference_audio_path=training_result.speaker_reference_path,
                output_path=out_file,
                checkpoint_path=training_result.best_model_path,
                config_path=training_result.config_path,
                vocab_path=training_result.vocab_path,
            )
            cand_audio_paths[s_id] = out_file

        print(f"Evaluating Candidate fine-tuned model on {len(val_texts)} sentences...")
        cand_report = self.evaluator.evaluate_model_generations(
            model_type="fine_tuned",
            user_id=user_id,
            validation_items=val_texts,
            generated_audio_paths=cand_audio_paths,
        )
        print(f"Candidate Speaker Similarity: {cand_report.mean_speaker_similarity:.3f}")
        print(f"Candidate WER:                {cand_report.mean_wer:.3f}")
        print(f"Candidate Quality Score:      {cand_report.mean_quality_score:.3f}")

        # -------------------------------------------------------------
        # 7. Model Selection & Checkpoint Recording
        # -------------------------------------------------------------
        print(f"\n[Step 5/5] Performing comparative model selection...")
        comp_report = self.evaluator.compare_reports(zs_report, cand_report)
        selection_result = self.model_selector.evaluate_decision(comp_report)

        print(f"DECISION: {selection_result.decision.value}")
        print(f"Reason:   {selection_result.next_action}")
        if selection_result.rejection_reasons:
            for r in selection_result.rejection_reasons:
                print(f"  - {r}")

        # Save checkpoint metadata & update active state
        self.checkpoint_manager.record_round(
            user_id=user_id,
            round_number=next_round,
            selection_result=selection_result,
            train_count=len(split.train_samples),
            val_count=len(split.val_samples),
            train_duration=split.total_train_duration_sec,
            val_duration=split.total_val_duration_sec,
            round_dir=round_dir,
            model_checkpoint_path=training_result.best_model_path,
            config_path=training_result.config_path,
            vocab_path=training_result.vocab_path,
            speaker_reference_path=training_result.speaker_reference_path,
        )

        # Save structured results to results/
        self._save_results_summary(user_id, next_round, comp_report, selection_result)

        updated_state = self.checkpoint_manager.load_active_model_state(user_id)

        return PipelineRunResult(
            user_id=user_id,
            round_number=next_round,
            data_status=data_status,
            executed_training=True,
            selection_result=selection_result,
            zero_shot_speaker_similarity=zs_report.mean_speaker_similarity,
            candidate_speaker_similarity=cand_report.mean_speaker_similarity,
            zero_shot_wer=zs_report.mean_wer,
            candidate_wer=cand_report.mean_wer,
            decision=selection_result.decision.value,
            active_primary_engine=updated_state.primary_model,
            message=selection_result.next_action,
        )

    def _save_results_summary(
        self,
        user_id: str,
        round_number: int,
        comp_report: Any,
        selection_result: SelectionResult
    ) -> None:
        """Save results to CSV and JSON in Voice_Finetuning/results/."""
        user_res_dir = RESULTS_DIRECTORY / user_id
        user_res_dir.mkdir(parents=True, exist_ok=True)

        # Append to evaluation_results.csv
        csv_path = user_res_dir / "evaluation_results.csv"
        file_exists = csv_path.exists()

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "user_id",
                    "round_number",
                    "decision",
                    "zs_speaker_sim",
                    "cand_speaker_sim",
                    "sim_delta",
                    "zs_wer",
                    "cand_wer",
                    "wer_delta",
                    "zs_quality",
                    "cand_quality",
                ])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                user_id,
                round_number,
                selection_result.decision.value,
                f"{selection_result.zero_shot_speaker_similarity:.4f}",
                f"{selection_result.candidate_speaker_similarity:.4f}",
                f"{selection_result.speaker_similarity_delta:+.4f}",
                f"{selection_result.zero_shot_wer:.4f}",
                f"{selection_result.candidate_wer:.4f}",
                f"{selection_result.wer_delta:+.4f}",
                f"{selection_result.zero_shot_quality_score:.4f}",
                f"{selection_result.candidate_quality_score:.4f}",
            ])
