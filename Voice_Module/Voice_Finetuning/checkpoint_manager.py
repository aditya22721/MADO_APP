"""
Per-User Checkpoint and Training Round Manager for Voice_Finetuning.

Responsibilities:
    - Maintains isolated per-user checkpoint hierarchy:
        checkpoints/{user_id}/
            active_model.json
            history.json
            round_01/
                best_model.pth
                config.json
                vocab.json
                speaker_reference.wav
                metadata.json
            round_02/
                ...
    - Tracks complete historical metadata per round
    - Sets active primary and fallback model pointers
    - Enables safe rollback to previous known-good checkpoints or zero-shot
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import CHECKPOINTS_DIRECTORY
from .model_selector import ModelDecision, SelectionResult


@dataclass
class CheckpointMetadata:
    """Metadata recorded for each fine-tuning round."""
    user_id: str
    training_round: int
    created_at: str
    promotion_status: str  # "PROMOTED" or "REJECTED"
    training_record_count: int
    validation_record_count: int
    training_duration_seconds: float
    validation_duration_seconds: float
    checkpoint_dir: str
    model_checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    vocab_path: Optional[str] = None
    speaker_reference_path: Optional[str] = None
    speaker_similarity: float = 0.0
    zero_shot_speaker_similarity: float = 0.0
    speaker_similarity_delta: float = 0.0
    wer: float = 0.0
    zero_shot_wer: float = 0.0
    quality_score: float = 0.0
    zero_shot_quality_score: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class ActiveModelState:
    """Active model state definition for a user."""
    user_id: str
    primary_model: str  # "fine_tuned" or "zero_shot"
    fallback_model: str  # "zero_shot"
    active_round: Optional[int] = None
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    vocab_path: Optional[str] = None
    speaker_reference_path: Optional[str] = None
    last_updated: str = ""


class CheckpointManager:
    """
    Manages per-user fine-tuning checkpoints, metadata, and active model state.
    """

    def __init__(
        self,
        checkpoints_base_dir: Union[str, Path] = CHECKPOINTS_DIRECTORY
    ):
        self.checkpoints_base_dir = Path(checkpoints_base_dir)

    def get_user_checkpoints_dir(self, user_id: str) -> Path:
        """Return the checkpoints directory for a user."""
        p = self.checkpoints_base_dir / user_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_active_model_file(self, user_id: str) -> Path:
        """Return path to active_model.json for a user."""
        return self.get_user_checkpoints_dir(user_id) / "active_model.json"

    def get_history_file(self, user_id: str) -> Path:
        """Return path to history.json for a user."""
        return self.get_user_checkpoints_dir(user_id) / "history.json"

    def get_round_dir(self, user_id: str, round_number: int) -> Path:
        """Return path to round directory: checkpoints/{user_id}/round_{n:02d}/."""
        p = self.get_user_checkpoints_dir(user_id) / f"round_{round_number:02d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_latest_round_number(self, user_id: str) -> int:
        """Return highest training round number recorded for user (0 if none)."""
        history = self.load_training_history(user_id)
        if not history:
            return 0
        return max(item.get("training_round", 0) for item in history)

    def load_active_model_state(self, user_id: str) -> ActiveModelState:
        """Load user's current active model configuration."""
        active_file = self.get_active_model_file(user_id)
        if not active_file.exists():
            return ActiveModelState(
                user_id=user_id,
                primary_model="zero_shot",
                fallback_model="zero_shot",
                active_round=None,
                checkpoint_path=None,
                config_path=None,
                vocab_path=None,
                speaker_reference_path=None,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

        with open(active_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return ActiveModelState(**data)

    def save_active_model_state(self, state: ActiveModelState) -> None:
        """Save user's active model state."""
        active_file = self.get_active_model_file(state.user_id)
        state.last_updated = datetime.now(timezone.utc).isoformat()
        with open(active_file, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=4)

    def load_training_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Load list of historical training round metadata for user."""
        hist_file = self.get_history_file(user_id)
        if not hist_file.exists():
            return []
        with open(hist_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def append_to_history(self, metadata: CheckpointMetadata) -> None:
        """Append metadata of a completed round to user's history."""
        history = self.load_training_history(metadata.user_id)
        history.append(asdict(metadata))
        hist_file = self.get_history_file(metadata.user_id)
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

    def record_round(
        self,
        user_id: str,
        round_number: int,
        selection_result: SelectionResult,
        train_count: int,
        val_count: int,
        train_duration: float,
        val_duration: float,
        round_dir: Path,
        model_checkpoint_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        vocab_path: Optional[Path] = None,
        speaker_reference_path: Optional[Path] = None,
    ) -> CheckpointMetadata:
        """
        Record a completed round's artifacts, metadata, and update active model state.
        """
        metadata = CheckpointMetadata(
            user_id=user_id,
            training_round=round_number,
            created_at=datetime.now(timezone.utc).isoformat(),
            promotion_status=selection_result.decision.value,
            training_record_count=train_count,
            validation_record_count=val_count,
            training_duration_seconds=train_duration,
            validation_duration_seconds=val_duration,
            checkpoint_dir=str(round_dir.resolve()),
            model_checkpoint_path=str(model_checkpoint_path.resolve()) if model_checkpoint_path else None,
            config_path=str(config_path.resolve()) if config_path else None,
            vocab_path=str(vocab_path.resolve()) if vocab_path else None,
            speaker_reference_path=str(speaker_reference_path.resolve()) if speaker_reference_path else None,
            speaker_similarity=selection_result.candidate_speaker_similarity,
            zero_shot_speaker_similarity=selection_result.zero_shot_speaker_similarity,
            speaker_similarity_delta=selection_result.speaker_similarity_delta,
            wer=selection_result.candidate_wer,
            zero_shot_wer=selection_result.zero_shot_wer,
            quality_score=selection_result.candidate_quality_score,
            zero_shot_quality_score=selection_result.zero_shot_quality_score,
            rejection_reasons=selection_result.rejection_reasons,
        )

        # Save round-specific metadata.json
        meta_path = round_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=4)

        # Update history
        self.append_to_history(metadata)

        # If PROMOTED, update active model state to point to this fine-tuned model
        if selection_result.decision == ModelDecision.PROMOTED:
            new_state = ActiveModelState(
                user_id=user_id,
                primary_model="fine_tuned",
                fallback_model="zero_shot",
                active_round=round_number,
                checkpoint_path=metadata.model_checkpoint_path,
                config_path=metadata.config_path,
                vocab_path=metadata.vocab_path,
                speaker_reference_path=metadata.speaker_reference_path,
            )
            self.save_active_model_state(new_state)

        return metadata

    def rollback_to_zero_shot(self, user_id: str) -> ActiveModelState:
        """Safely revert active voice model to Zero-Shot XTTS."""
        state = ActiveModelState(
            user_id=user_id,
            primary_model="zero_shot",
            fallback_model="zero_shot",
            active_round=None,
            checkpoint_path=None,
            config_path=None,
            vocab_path=None,
            speaker_reference_path=None,
        )
        self.save_active_model_state(state)
        return state
