"""
Simple Baseline Model.
Zero-shot LLM / simple keyword model without RAG context retrieval or safety-tuned escalation rules.
"""

from typing import Dict, Any


class SimpleBaselineAgent:
    """Simple baseline model without RAG context grounding."""

    def __init__(self):
        pass

    def process_message(self, customer_text: str) -> Dict[str, Any]:
        """Process query using simple baseline logic."""
        text_lower = customer_text.lower()

        # Simple keyword intent guessing
        if "wifi" in text_lower or "battery" in text_lower or "screen" in text_lower or "app" in text_lower:
            intent = "TECHNICAL_TROUBLESHOOTING"
        elif "password" in text_lower or "apple id" in text_lower or "locked" in text_lower:
            intent = "ACCOUNT_ACCESS_SECURITY"
        elif "charge" in text_lower or "refund" in text_lower or "billing" in text_lower:
            intent = "BILLING_SUBSCRIPTION"
        elif "order" in text_lower or "shipping" in text_lower:
            intent = "ORDER_SHIPPING_DELIVERY"
        elif "repair" in text_lower or "broken" in text_lower:
            intent = "HARDWARE_REPAIR_WARRANTY"
        elif "update" in text_lower or "ios" in text_lower:
            intent = "SOFTWARE_UPDATE_OS"
        else:
            intent = "GENERAL_INQUIRY_FEEDBACK"

        # Simple threshold escalation: only escalate if word 'stolen' or 'locked' is in text
        escalate = ("stolen" in text_lower or "locked" in text_lower)
        action = "ESCALATE" if escalate else "AUTO_HANDLE"
        reason = "Simple keyword escalation trigger." if escalate else "No escalation keywords detected."

        # Un-grounded generic LLM-style reply draft
        reply_draft = f"Hello! We understand you are having an issue with {intent.lower().replace('_', ' ')}. Please check your settings or visit support.apple.com for more info."

        return {
            "customer_text": customer_text,
            "intent": intent,
            "intent_confidence": 0.7,
            "action": action,
            "escalate": escalate,
            "escalation_reason": reason,
            "confidence_score": 0.6,
            "retrieved_context": [],
            "reply_draft": reply_draft
        }
