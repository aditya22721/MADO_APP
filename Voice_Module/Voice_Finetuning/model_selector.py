"""
Model Selection Engine for Voice_Finetuning.

Determines whether a fine-tuned XTTS candidate model should be:
    - PROMOTED to become the user's primary voice model
    - REJECTED, retaining Zero-Shot XTTS as primary and recommending
      collection of an additional batch of recordings.

Decision Criteria:
    1. Hard Minimum Quality Acceptance:
       - Candidate speaker similarity >= MIN_SPEAKER_SIMILARITY (default: 0.72)
       - Candidate WER <= MAX_WER (default: 0.20)
       - Candidate quality score >= MIN_QUALITY_SCORE (default: 0.70)
       - Max clipping ratio <= 0.02
    2. Meaningful Improvement Over Zero-Shot:
       - Speaker similarity improvement >= MIN_IMPROVEMENT_MARGIN (default: +0.02)
       - Candidate WER does not significantly regress compared to zero-shot
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .config import (
    BATCH_INCREMENT,
    MAX_WER,
    MIN_IMPROVEMENT_MARGIN,
    MIN_QUALITY_SCORE,
    MIN_SPEAKER_SIMILARITY,
)
from .evaluator import ComparativeEvaluationReport


class ModelDecision(str, Enum):
    """Model selection outcome."""
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


@dataclass
class SelectionResult:
    """Structured result of model promotion/rejection decision."""
    user_id: str
    decision: ModelDecision
    passed_minimum_criteria: bool
    meaningfully_better_than_zero_shot: bool
    rejection_reasons: List[str] = field(default_factory=list)
    next_action: str = ""
    recommended_additional_recordings: int = 0
    candidate_speaker_similarity: float = 0.0
    zero_shot_speaker_similarity: float = 0.0
    speaker_similarity_delta: float = 0.0
    candidate_wer: float = 0.0
    zero_shot_wer: float = 0.0
    wer_delta: float = 0.0
    candidate_quality_score: float = 0.0
    zero_shot_quality_score: float = 0.0


class ModelSelector:
    """
    Evaluates candidate vs baseline and issues promotion/rejection decisions.
    """

    def __init__(
        self,
        min_speaker_similarity: float = MIN_SPEAKER_SIMILARITY,
        max_wer: float = MAX_WER,
        min_improvement_margin: float = MIN_IMPROVEMENT_MARGIN,
        min_quality_score: float = MIN_QUALITY_SCORE,
    ):
        self.min_speaker_similarity = min_speaker_similarity
        self.max_wer = max_wer
        self.min_improvement_margin = min_improvement_margin
        self.min_quality_score = min_quality_score

    def evaluate_decision(
        self,
        report: ComparativeEvaluationReport
    ) -> SelectionResult:
        """
        Evaluate comparative report and decide whether to promote the candidate model.
        """
        cand = report.candidate_report
        zs = report.zero_shot_report

        rejection_reasons: List[str] = []

        # 1. Hard Minimum Quality Checks
        passed_min = True

        if cand.mean_speaker_similarity < self.min_speaker_similarity:
            passed_min = False
            rejection_reasons.append(
                f"Speaker similarity {cand.mean_speaker_similarity:.3f} is below minimum threshold ({self.min_speaker_similarity:.3f})"
            )

        if cand.mean_wer > self.max_wer:
            passed_min = False
            rejection_reasons.append(
                f"Word Error Rate {cand.mean_wer:.3f} exceeds maximum allowed error ({self.max_wer:.3f})"
            )

        if cand.mean_quality_score < self.min_quality_score:
            passed_min = False
            rejection_reasons.append(
                f"Audio quality score {cand.mean_quality_score:.3f} is below minimum ({self.min_quality_score:.3f})"
            )

        if cand.max_clipping_ratio > 0.02:
            passed_min = False
            rejection_reasons.append(
                f"Excessive clipping observed in validation output: {cand.max_clipping_ratio * 100:.1f}%"
            )

        # 2. Meaningful Improvement Over Zero-Shot Checks
        better_than_zs = True

        if report.speaker_similarity_delta < self.min_improvement_margin:
            better_than_zs = False
            rejection_reasons.append(
                f"Speaker similarity improvement ({report.speaker_similarity_delta:+.3f}) is below minimum margin ({self.min_improvement_margin:+.3f})"
            )

        if cand.mean_wer > zs.mean_wer + 0.05:
            better_than_zs = False
            rejection_reasons.append(
                f"Intelligibility degraded: Candidate WER ({cand.mean_wer:.3f}) is worse than Zero-Shot ({zs.mean_wer:.3f})"
            )

        # Final Decision
        if passed_min and better_than_zs:
            decision = ModelDecision.PROMOTED
            next_action = "Promote fine-tuned checkpoint to active primary model. Keep zero-shot XTTS as fallback."
            rec_recordings = 0
        else:
            decision = ModelDecision.REJECTED
            next_action = f"Candidate rejected. Keep zero-shot XTTS as primary. Collect additional {BATCH_INCREMENT} valid recordings before retraining."
            rec_recordings = BATCH_INCREMENT

        return SelectionResult(
            user_id=report.user_id,
            decision=decision,
            passed_minimum_criteria=passed_min,
            meaningfully_better_than_zero_shot=better_than_zs,
            rejection_reasons=rejection_reasons,
            next_action=next_action,
            recommended_additional_recordings=rec_recordings,
            candidate_speaker_similarity=cand.mean_speaker_similarity,
            zero_shot_speaker_similarity=zs.mean_speaker_similarity,
            speaker_similarity_delta=report.speaker_similarity_delta,
            candidate_wer=cand.mean_wer,
            zero_shot_wer=zs.mean_wer,
            wer_delta=report.wer_delta,
            candidate_quality_score=cand.mean_quality_score,
            zero_shot_quality_score=zs.mean_quality_score,
        )
