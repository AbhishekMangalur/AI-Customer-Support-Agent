"""
Unit tests for evaluation metrics, LLM judge, and calibration calculations.
"""

import pytest
from src.eval.metrics import compute_intent_metrics, compute_escalation_metrics, compute_n_gram_bleu, compute_text_quality_metrics
from src.eval.llm_judge import LLMJudgeEvaluator
from src.eval.calibration import compute_judge_calibration


def test_intent_metrics():
    y_true = ["TECHNICAL_TROUBLESHOOTING", "ACCOUNT_ACCESS_SECURITY", "BILLING_SUBSCRIPTION"]
    y_pred = ["TECHNICAL_TROUBLESHOOTING", "ACCOUNT_ACCESS_SECURITY", "BILLING_SUBSCRIPTION"]
    metrics = compute_intent_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_escalation_metrics():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, False]
    metrics = compute_escalation_metrics(y_true, y_pred)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["false_negatives_count"] == 1


def test_bleu_and_text_metrics():
    ref = ["Go to Settings > Wi-Fi > Forget Network and reconnect."]
    cand = ["Go to Settings > Wi-Fi > Forget Network and reconnect."]
    metrics = compute_text_quality_metrics(ref, cand)
    assert metrics["bleu"] > 0.8
    assert metrics["cosine_similarity"] > 0.9


def test_judge_calibration():
    human = [5.0, 4.0, 3.0, 2.0, 1.0]
    judge = [5.0, 4.0, 3.0, 2.0, 1.0]
    calib = compute_judge_calibration(human, judge)
    assert calib["pearson_r"] == 1.0
    assert calib["mae"] == 0.0
    assert calib["exact_match_pct"] == 100.0


def test_llm_judge():
    evaluator = LLMJudgeEvaluator()
    res = evaluator.evaluate_reply(
        customer_text="My iPhone screen cracked",
        ground_truth_intent="HARDWARE_REPAIR_WARRANTY",
        ground_truth_escalate=True,
        ground_truth_reason="Physical screen repair required",
        reference_reply="Book repair at support.apple.com/repair",
        candidate_reply="Book repair at support.apple.com/repair",
        agent_action="ESCALATE"
    )
    assert "overall_score" in res
    assert res["overall_score"] >= 3.0
