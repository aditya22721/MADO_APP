"""
Unit Tests for Voice_Finetuning Evaluator and Utilities.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Voice_Finetuning.evaluator import (
    ComparativeEvaluationReport,
    Evaluator,
    ModelEvaluationReport,
)
from Voice_Finetuning.utils import (
    compute_cer,
    compute_levenshtein_distance,
    compute_text_similarity,
    compute_wer,
    normalize_text,
)


class TestEvaluatorAndUtils(unittest.TestCase):

    def test_text_normalization(self):
        raw = "Hello, WORLD! 123... How are you?"
        norm = normalize_text(raw)
        self.assertEqual(norm, "hello world 123 how are you")

    def test_wer_exact_match(self):
        ref = "the quick brown fox"
        hyp = "the quick brown fox"
        self.assertEqual(compute_wer(ref, hyp), 0.0)
        self.assertEqual(compute_text_similarity(ref, hyp), 1.0)

    def test_wer_substitution_and_deletion(self):
        ref = "please take your medication right after lunch"
        hyp = "please take your pills right after"
        # 1 substitution (medication -> pills), 1 deletion (lunch)
        # distance = 2, len(ref) = 7 -> wer = 2/7
        wer = compute_wer(ref, hyp)
        self.assertAlmostEqual(wer, 2 / 7, places=3)

    def test_comparative_report_logic(self):
        evaluator = Evaluator()

        zs_report = ModelEvaluationReport(
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

        # Candidate with higher speaker similarity and lower WER
        good_cand_report = ModelEvaluationReport(
            model_type="fine_tuned",
            user_id="user_test",
            num_sentences=5,
            mean_speaker_similarity=0.790,
            mean_wer=0.080,
            mean_cer=0.020,
            mean_text_similarity=0.920,
            mean_quality_score=0.860,
            max_clipping_ratio=0.0,
            mean_silence_ratio=0.09,
        )

        comp = evaluator.compare_reports(zs_report, good_cand_report, min_margin=0.02)
        self.assertTrue(comp.candidate_is_better)
        self.assertAlmostEqual(comp.speaker_similarity_delta, +0.050, places=3)
        self.assertAlmostEqual(comp.wer_delta, -0.040, places=3)


if __name__ == "__main__":
    unittest.main()
