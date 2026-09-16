"""
Trivial Baseline Model.
Predicts majority intent, returns static template response, and auto-handles all messages.
"""

from typing import Dict, Any


class TrivialBaselineAgent:
    """Trivial baseline model for comparison."""

    def __init__(self, majority_intent: str = "TECHNICAL_TROUBLESHOOTING"):
        self.majority_intent = majority_intent
        self.static_reply = "Thanks for reaching out! Please send us a DM with your details so we can help."

    def process_message(self, customer_text: str) -> Dict[str, Any]:
        """Process query using trivial baseline logic."""
        return {
            "customer_text": customer_text,
            "intent": self.majority_intent,
            "intent_confidence": 1.0,
            "action": "AUTO_HANDLE",
            "escalate": False,
            "escalation_reason": "Trivial baseline auto-handles all queries.",
            "confidence_score": 0.5,
            "retrieved_context": [],
            "reply_draft": self.static_reply
        }
