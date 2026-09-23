"""
Multi-Signal Evaluation Engine for Voice_Finetuning.

Evaluates synthesized speech using three independent objective signals:
    1. Speaker Similarity:
       Cosine similarity computed via ECAPA-TDNN speaker encoder against
       the reference speaker embeddings.
    2. Content Intelligibility:
       Faster-Whisper transcription of synthesized audio compared against
       expected validation text (Word Error Rate & Normalized Similarity).
    3. Audio Stability & Quality:
       Absence of NaN/Inf, low clipping, reasonable silence ratio, and
       expected duration ratio.

Both Zero-Shot Baseline and Fine-Tuned Candidate are evaluated on the exact
same validation sentences.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import soundfile as sf
import torch

from Voice_Input.speech_to_text import speech_to_text
from Voice_Recognition.speaker_encoder import SpeakerEncoder
from Voice_Recognition.voice_profile_manager import VoiceProfileManager

from .utils import analyze_audio_signal, compute_cer, compute_text_similarity, compute_wer


@dataclass
class SentenceEvaluationResult:
    """Evaluation result for a single synthesized validation sentence."""
    sentence_id: str
    expected_text: str
    generated_audio_path: Path
    transcribed_text: str
    wer: float
    cer: float
    text_similarity: float
    speaker_similarity: float
    clipping_ratio: float
    silence_ratio: float
    duration_seconds: float
    has_nan_inf: bool
    quality_score: float  # Composite quality index [0.0 - 1.0]


@dataclass
class ModelEvaluationReport:
    """Aggregated evaluation report across all validation sentences."""
    model_type: str  # "zero_shot" or "fine_tuned"
    user_id: str
    num_sentences: int
    mean_speaker_similarity: float
    mean_wer: float
    mean_cer: float
    mean_text_similarity: float
    mean_quality_score: float
    max_clipping_ratio: float
    mean_silence_ratio: float
    sentence_results: List[SentenceEvaluationResult] = field(default_factory=list)


@dataclass
class ComparativeEvaluationReport:
    """Direct comparison between Zero-Shot baseline and Fine-Tuned candidate."""
    user_id: str
    zero_shot_report: ModelEvaluationReport
    candidate_report: ModelEvaluationReport
    speaker_similarity_delta: float
    wer_delta: float  # Negative is better (lower error)
    quality_score_delta: float
    candidate_is_better: bool


class Evaluator:
    """
    Orchestrates multi-signal evaluation for synthesized audio files.
    """

    def __init__(
        self,
        speaker_encoder: Optional[SpeakerEncoder] = None,
        profile_manager: Optional[VoiceProfileManager] = None,
        stt_func: Optional[Callable[[Union[str, Path]], Path]] = None,
    ):
        self._speaker_encoder = speaker_encoder
        self._profile_manager = profile_manager
        self._stt_func = stt_func or speech_to_text

    @property
    def speaker_encoder(self) -> SpeakerEncoder:
        if self._speaker_encoder is None:
            self._speaker_encoder = SpeakerEncoder()
        return self._speaker_encoder

    @property
    def profile_manager(self) -> VoiceProfileManager:
        if self._profile_manager is None:
            self._profile_manager = VoiceProfileManager()
        return self._profile_manager

    def compute_speaker_similarity(
        self,
        generated_audio_path: Path,
        user_id: str
    ) -> float:
        """
        Compute mean cosine similarity between generated audio embedding and
        the target user's active reference embeddings.
        """
        # Encode generated audio
        gen_emb = self.speaker_encoder.encode_and_normalize(generated_audio_path)

        # Load target user reference embeddings
        ref_embeddings = self.profile_manager.load_all_reference_embeddings(user_id)
        if not ref_embeddings:
            # Fallback: if no stored embeddings, encode user audio files
            audio_files = self.profile_manager.get_active_reference_audio_paths(user_id)
            if audio_files:
                ref_embeddings = [
                    self.speaker_encoder.encode_and_normalize(Path(p))
                    for p in audio_files if Path(p).exists()
                ]

        if not ref_embeddings:
            raise ValueError(f"No reference embeddings or audio available for user '{user_id}'")

        # Compute cosine similarity against all references
        similarities = []
        for ref_emb in ref_embeddings:
            ref_norm = torch.nn.functional.normalize(ref_emb, p=2, dim=0).to(gen_emb.device)
            sim = torch.dot(gen_emb, ref_norm).item()
            similarities.append(sim)

        mean_sim = float(sum(similarities) / len(similarities))
        return mean_sim

    def evaluate_sentence(
        self,
        sentence_id: str,
        expected_text: str,
        generated_audio_path: Path,
        user_id: str
    ) -> SentenceEvaluationResult:
        """
        Evaluate a single synthesized sentence audio file across all signals.
        """
        if not generated_audio_path.exists():
            raise FileNotFoundError(f"Generated audio file not found: {generated_audio_path}")

        # 1. Signal & Stability analysis
        signal_info = analyze_audio_signal(generated_audio_path)

        # 2. Content Intelligibility via Faster-Whisper
        transcript_file = self._stt_func(generated_audio_path)
        transcribed_text = Path(transcript_file).read_text(encoding="utf-8").strip()

        wer = compute_wer(expected_text, transcribed_text)
        cer = compute_cer(expected_text, transcribed_text)
        text_sim = compute_text_similarity(expected_text, transcribed_text)

        # 3. Speaker Similarity via ECAPA-TDNN
        speaker_sim = self.compute_speaker_similarity(generated_audio_path, user_id)

        # 4. Composite Quality Index [0.0 - 1.0]
        # Penalizes clipping, excessive silence, and NaN/Inf
        quality_penalty = 0.0
        if signal_info["has_nan_inf"]:
            quality_penalty += 0.50
        quality_penalty += min(0.30, signal_info["clipping_ratio"] * 10.0)
        if signal_info["silence_ratio"] > 0.50:
            quality_penalty += min(0.20, (signal_info["silence_ratio"] - 0.50) * 0.5)

        base_score = 1.0 - quality_penalty
        quality_score = max(0.0, min(1.0, base_score))

        return SentenceEvaluationResult(
            sentence_id=sentence_id,
            expected_text=expected_text,
            generated_audio_path=generated_audio_path,
            transcribed_text=transcribed_text,
            wer=wer,
            cer=cer,
            text_similarity=text_sim,
            speaker_similarity=speaker_sim,
            clipping_ratio=signal_info["clipping_ratio"],
            silence_ratio=signal_info["silence_ratio"],
            duration_seconds=signal_info["duration"],
            has_nan_inf=signal_info["has_nan_inf"],
            quality_score=quality_score,
        )

    def evaluate_model_generations(
        self,
        model_type: str,
        user_id: str,
        validation_items: List[Dict[str, str]],
        generated_audio_paths: Dict[str, Path]
    ) -> ModelEvaluationReport:
        """
        Evaluate a complete batch of synthesized validation sentences for a model.
        """
        sentence_results: List[SentenceEvaluationResult] = []

        for item in validation_items:
            s_id = item.get("id", item.get("text", "")[:10])
            expected_text = item["text"]
            audio_path = generated_audio_paths[s_id]

            res = self.evaluate_sentence(
                sentence_id=s_id,
                expected_text=expected_text,
                generated_audio_path=audio_path,
                user_id=user_id
            )
            sentence_results.append(res)

        n = len(sentence_results)
        if n == 0:
            raise ValueError("No sentence evaluation results were produced.")

        mean_spk_sim = sum(r.speaker_similarity for r in sentence_results) / n
        mean_wer = sum(r.wer for r in sentence_results) / n
        mean_cer = sum(r.cer for r in sentence_results) / n
        mean_text_sim = sum(r.text_similarity for r in sentence_results) / n
        mean_quality = sum(r.quality_score for r in sentence_results) / n
        max_clipping = max(r.clipping_ratio for r in sentence_results)
        mean_silence = sum(r.silence_ratio for r in sentence_results) / n

        return ModelEvaluationReport(
            model_type=model_type,
            user_id=user_id,
            num_sentences=n,
            mean_speaker_similarity=float(mean_spk_sim),
            mean_wer=float(mean_wer),
            mean_cer=float(mean_cer),
            mean_text_similarity=float(mean_text_sim),
            mean_quality_score=float(mean_quality),
            max_clipping_ratio=float(max_clipping),
            mean_silence_ratio=float(mean_silence),
            sentence_results=sentence_results,
        )

    def compare_reports(
        self,
        zero_shot_report: ModelEvaluationReport,
        candidate_report: ModelEvaluationReport,
        min_margin: float = 0.02
    ) -> ComparativeEvaluationReport:
        """
        Compare fine-tuned candidate against zero-shot baseline.
        """
        spk_delta = candidate_report.mean_speaker_similarity - zero_shot_report.mean_speaker_similarity
        wer_delta = candidate_report.mean_wer - zero_shot_report.mean_wer
        qual_delta = candidate_report.mean_quality_score - zero_shot_report.mean_quality_score

        # Candidate is better if speaker similarity is improved by margin OR
        # speaker similarity is equal/better and WER is lower without quality degradation
        is_better = (
            spk_delta >= min_margin and
            candidate_report.mean_wer <= zero_shot_report.mean_wer + 0.05 and
            candidate_report.mean_quality_score >= zero_shot_report.mean_quality_score - 0.05
        )

        return ComparativeEvaluationReport(
            user_id=candidate_report.user_id,
            zero_shot_report=zero_shot_report,
            candidate_report=candidate_report,
            speaker_similarity_delta=float(spk_delta),
            wer_delta=float(wer_delta),
            quality_score_delta=float(qual_delta),
            candidate_is_better=is_better,
        )
