"""
Unit tests for Apple Support Agent components and regression checks.
"""

import pytest
from src.agent.intent_classifier import IntentClassifier
from src.agent.retriever import ResolutionRetriever, normalize_grounding_score
from src.agent.escalation_router import EscalationRouter
from src.agent.generator import GroundedReplyGenerator
from src.agent.support_agent import AppleSupportAgent


# --- ISSUE 1 REGRESSION TESTS: Grounding Score Normalization ---

def test_normalize_grounding_score_cases():
    """Verify normalization clamping to [0.00, 1.00] with 2 decimal places."""
    assert normalize_grounding_score(0) == 0.00
    assert normalize_grounding_score(0.36) == 0.36
    assert normalize_grounding_score(0.75) == 0.75
    assert normalize_grounding_score(1.00) == 1.00
    assert normalize_grounding_score(1.15) == 1.00  # Clamped above 1.0
    assert normalize_grounding_score(-0.50) == 0.00 # Clamped negative


def test_retriever_score_clamping():
    """Verify retriever outputs similarity scores strictly within [0.00, 1.00]."""
    retriever = ResolutionRetriever()
    results = retriever.retrieve("I dropped my iPhone 15 Pro Max and the back glass shattered into pieces.", intent="HARDWARE_REPAIR_WARRANTY", top_k=3)
    for res in results:
        score = res["similarity_score"]
        assert 0.00 <= score <= 1.00
        assert score != 1.15


# --- ISSUE 2 REGRESSION TESTS: Issue-Specific Escalation Reasons ---

def test_escalation_reason_shattered_glass_does_not_contain_charger_pin():
    """Verify shattered glass escalation reason mentions physical damage ONLY and NOT charger pin or swollen battery."""
    router = EscalationRouter()
    res = router.evaluate(
        customer_text="I dropped my iPhone 15 Pro Max and the back glass shattered into pieces.",
        intent="HARDWARE_REPAIR_WARRANTY",
        retrieved_context=[]
    )
    reason = res["escalation_reason"].lower()
    assert res["action"] == "ESCALATE"
    assert "physical damage" in reason or "repair" in reason
    assert "charger port pin" not in reason
    assert "pin" not in reason
    assert "swollen" not in reason
    assert "smoke" not in reason


def test_escalation_reason_broken_charger_pin():
    """Verify broken charger pin escalation reason mentions pin ONLY and NOT shattered glass or swollen battery."""
    router = EscalationRouter()
    res = router.evaluate(
        customer_text="My iPad charger port has a broken pin stuck inside it.",
        intent="HARDWARE_REPAIR_WARRANTY",
        retrieved_context=[]
    )
    reason = res["escalation_reason"].lower()
    assert res["action"] == "ESCALATE"
    assert "charger pin" in reason or "broken charger pin" in reason
    assert "shattered" not in reason
    assert "glass" not in reason
    assert "swollen" not in reason


def test_escalation_reason_swollen_battery():
    """Verify swollen battery escalation reason mentions swollen/expanding battery safety hazard."""
    router = EscalationRouter()
    res = router.evaluate(
        customer_text="The battery on my iPhone has swollen up and is pushing the screen outward!",
        intent="HARDWARE_REPAIR_WARRANTY",
        retrieved_context=[]
    )
    reason = res["escalation_reason"].lower()
    assert res["action"] == "ESCALATE"
    assert "swollen" in reason or "expanding" in reason
    assert "charger port pin" not in reason
    assert "shattered" not in reason


def test_normal_battery_drain_auto_handle_no_escalation_reason():
    """Verify normal battery drain is AUTO_HANDLE with no escalation reason."""
    agent = AppleSupportAgent()
    res = agent.process_message("My iPhone battery drops from 80% to 10% in less than an hour after updating.")
    assert res["intent"] == "TECHNICAL_TROUBLESHOOTING"
    assert res["action"] == "AUTO_HANDLE"
    assert res["escalation_reason"] == ""
    assert "critical hardware safety hazard" not in res["reply_draft"].lower()
    assert "stop charging" not in res["reply_draft"].lower()


def test_master_agent_grounding_score_range():
    """Verify master agent output has normalized retrieval grounding score in [0.00, 1.00]."""
    agent = AppleSupportAgent()
    res = agent.process_message("My card was charged three times for the same Apple Music subscription this month!")
    assert 0.00 <= res["retrieval_grounding_score"] <= 1.00
