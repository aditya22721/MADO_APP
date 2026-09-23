"""
Unit Tests for Voice_Finetuning ModelSelector.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Voice_Finetuning.evaluator import ComparativeEvaluationReport, ModelEvaluationReport
from Voice_Finetuning.model_selector import ModelDecision, ModelSelector


class TestModelSelector(unittest.TestCase):

    def setUp(self):
        self.selector = ModelSelector(
            min_speaker_similarity=0.72,
            max_wer=0.20,
            min_improvement_margin=0.02,
            min_quality_score=0.70,
        )

        self.zs_report = ModelEvaluationReport(
            model_type="zero_shot",
            user_id="user_test",
            num_sentences=5,
            mean_speaker_similarity=0.740,
            mean_wer=0.120,
            mean_cer=0.040,
            mean_text_similarity=0.880,
            mean_quality_score=0.850,
            max_clipping_ratio=0.0,
            mean_silence_ratio=0.10,
        )

    def test_promotion_when_exceeding_baseline(self):
        cand_report = ModelEvaluationReport(
            model_type="fine_tuned",
            user_id="user_test",
            num_sentences=5,
            mean_speaker_similarity=0.790,  # +0.050 improvement (>= 0.02)
            mean_wer=0.100,                 # lower error
            mean_cer=0.030,
            mean_text_similarity=0.900,
            mean_quality_score=0.850,
            max_clipping_ratio=0.0,
            mean_silence_ratio=0.10,
        )

        comp = ComparativeEvaluationReport(
            user_id="user_test",
            zero_shot_report=self.zs_report,
            candidate_report=cand_report,
            speaker_similarity_delta=0.050,
            wer_delta=-0.020,
            quality_score_delta=0.0,
            candidate_is_better=True,
        )

        res = self.selector.evaluate_decision(comp)
        self.assertEqual(res.decision, ModelDecision.PROMOTED)
        self.assertTrue(res.passed_minimum_criteria)
        self.assertTrue(res.meaningfully_better_than_zero_shot)
        self.assertEqual(len(res.rejection_reasons), 0)

    def test_rejection_on_negligible_improvement(self):
        cand_report = ModelEvaluationReport(
            model_type="fine_tuned",
            user_id="user_test",
            num_sentences=5,
            mean_speaker_similarity=0.745,  # only +0.005 improvement (< 0.02 margin)
            mean_wer=0.120,
            mean_cer=0.040,
            mean_text_similarity=0.880,
            mean_quality_score=0.850,
            max_clipping_ratio=0.0,
            mean_silence_ratio=0.10,
        )

        comp = ComparativeEvaluationReport(
            user_id="user_test",
            zero_shot_report=self.zs_report,
            candidate_report=cand_report,
            speaker_similarity_delta=0.005,
            wer_delta=0.0,
            quality_score_delta=0.0,
            candidate_is_better=False,
        )

        res = self.selector.evaluate_decision(comp)
        self.assertEqual(res.decision, ModelDecision.REJECTED)
        self.assertFalse(res.meaningfully_better_than_zero_shot)
        self.assertTrue(any("below minimum margin" in r for r in res.rejection_reasons))

    def test_rejection_on_quality_failure(self):
        cand_report = ModelEvaluationReport(
            model_type="fine_tuned",
            user_id="user_test",
            num_sentences=5,
            mean_speaker_similarity=0.800,  # High similarity
            mean_wer=0.350,                 # High error (> 0.20 max)
            mean_cer=0.150,
            mean_text_similarity=0.650,
            mean_quality_score=0.600,       # Low quality
            max_clipping_ratio=0.05,        # Clipping
            mean_silence_ratio=0.10,
        )

        comp = ComparativeEvaluationReport(
            user_id="user_test",
            zero_shot_report=self.zs_report,
            candidate_report=cand_report,
            speaker_similarity_delta=0.060,
            wer_delta=0.230,
            quality_score_delta=-0.250,
            candidate_is_better=False,
        )

        res = self.selector.evaluate_decision(comp)
        self.assertEqual(res.decision, ModelDecision.REJECTED)
        self.assertFalse(res.passed_minimum_criteria)


if __name__ == "__main__":
    unittest.main()
