"""
Master Support Agent Pipeline.
Coordinates Intent Classifier, Resolution Retriever, Reply Generator, and Escalation Router,
formatting output to match Section 18 of Hiver Specification.
"""

from typing import Dict, Any, Optional
from src.agent.intent_classifier import IntentClassifier
from src.agent.retriever import ResolutionRetriever, normalize_grounding_score
from src.agent.generator import GroundedReplyGenerator
from src.agent.escalation_router import EscalationRouter
from src.data.loader import TwitterDataLoader


class AppleSupportAgent:
    """End-to-end AI Support Agent for @AppleSupport adhering strictly to Hiver specification."""

    def __init__(self, data_loader: Optional[TwitterDataLoader] = None):
        self.data_loader = data_loader or TwitterDataLoader()
        self.intent_classifier = IntentClassifier(data_loader=self.data_loader)
        self.retriever = ResolutionRetriever(data_loader=self.data_loader)
        self.generator = GroundedReplyGenerator()
        self.router = EscalationRouter()

    def process_message(self, customer_text: str) -> Dict[str, Any]:
        """
        Process incoming customer tweet through full agent pipeline.
        Returns complete analysis dictionary matching Hiver Section 18 schema.
        """
        # 1. Classify Intent
        intent_res = self.intent_classifier.predict(customer_text)
        intent = intent_res["intent"]
        intent_conf = intent_res["confidence"]

        # 2. Retrieve Grounded Support Knowledge
        retrieved_context = self.retriever.retrieve(query=customer_text, intent=intent, top_k=3)
        raw_max_score = max([c.get("similarity_score", 0.0) for c in retrieved_context], default=0.0)
        norm_grounding_score = normalize_grounding_score(raw_max_score)

        # 3. Detect Critical Safety / High-Risk & Decide Action
        router_res = self.router.evaluate(
            customer_text=customer_text,
            intent=intent,
            retrieved_context=retrieved_context,
            intent_confidence=intent_conf
        )

        escalate = router_res["escalate"]
        escalation_reason = router_res["escalation_reason"] if escalate else ""
        action = router_res["action"]
        safety_assessment = router_res["safety_assessment"]

        # 4. Generate Grounded Response
        reply_draft = self.generator.generate_reply(
            customer_text=customer_text,
            intent=intent,
            retrieved_context=retrieved_context,
            escalate=escalate,
            escalation_reason=escalation_reason
        )

        # Formatted string report matching UI Specification
        report_lines = [
            f"Predicted Intent: {intent}",
            f"Confidence: {intent_conf*100:.1f}% ({intent_res.get('method', 'statistical')})",
            f"Action Decision: {action}"
        ]
        if escalate:
            report_lines.append("Escalation Status: ESCALATED")
            report_lines.append(f"Escalation Reason: {escalation_reason}")
        
        report_lines.append(f"Retrieval Grounding Score: {norm_grounding_score:.2f}")
        report_lines.append(f"Drafted Grounded Reply: {reply_draft}")
        
        if retrieved_context:
            first_ctx = retrieved_context[0].get("resolution_summary") or retrieved_context[0].get("brand_response") or ""
            report_lines.append(f"Retrieved Historical Resolution Context: {first_ctx}")

        formatted_output = "\n".join(report_lines)

        return {
            "customer_text": customer_text,
            "intent": intent,
            "intent_confidence": intent_conf,
            "intent_method": intent_res.get("method", "statistical"),
            "safety_assessment": safety_assessment,
            "action": action,
            "escalate": escalate,
            "escalation_reason": escalation_reason,
            "retrieval_grounding_score": norm_grounding_score,
            "confidence_score": norm_grounding_score,
            "triggers": router_res.get("triggers", []),
            "retrieved_context": retrieved_context,
            "reply_draft": reply_draft,
            "formatted_report": formatted_output
        }
