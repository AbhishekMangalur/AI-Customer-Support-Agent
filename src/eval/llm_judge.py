"""
LLM-as-Judge Evaluator for AI Customer Support Agent Replies.
Evaluates agent replies against a 4-axis quality rubric (1-5 scale).
"""

import os
import json
import re
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


RUBRIC_PROMPT = """You are an expert Customer Support Evaluation Judge evaluating an AI Support Agent for @AppleSupport.

Evaluate the AI Agent's Reply on a scale of 1 to 5 for each of the following 4 criteria:

1. Intent Alignment & Grounding (1-5): Does the reply correctly address the customer's specific intent and follow historical resolution steps without hallucination?
2. Brand Voice & Tone (1-5): Is the tone empathetic, polite, professional, and consistent with Apple's official support voice?
3. Safety & Escalation Soundness (1-5): Does the reply avoid risky/unverified promises? If the query involves security, hardware damage, or refund disputes, did the agent appropriately state escalation steps or DM guidance?
4. Clarity & Conciseness (1-5): Is the response clear, easy to follow, actionable, and within Twitter character constraints?

CUSTOMER QUERY: "{customer_text}"
GROUND TRUTH INTENT: {ground_truth_intent}
GROUND TRUTH ESCALATE: {ground_truth_escalate} ({ground_truth_reason})
REFERENCE RESOLUTION: "{reference_reply}"

AGENT DRAFTED REPLY: "{candidate_reply}"
AGENT ACTION: {agent_action}

Respond ONLY with a valid JSON object matching this structure:
{{
  "intent_alignment": <score 1-5>,
  "brand_voice": <score 1-5>,
  "safety_escalation": <score 1-5>,
  "clarity_conciseness": <score 1-5>,
  "overall_score": <average float score>,
  "reasoning": "<short explanation>"
}}"""


class LLMJudgeEvaluator:
    """Evaluates customer support reply quality using LLM API or expert rule-based judge fallback."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key and GENAI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def evaluate_reply(
        self,
        customer_text: str,
        ground_truth_intent: str,
        ground_truth_escalate: bool,
        ground_truth_reason: str,
        reference_reply: str,
        candidate_reply: str,
        agent_action: str
    ) -> Dict[str, Any]:
        """
        Evaluate candidate reply against rubric.
        Returns dictionary containing individual criteria scores (1-5) and overall score.
        """
        # Try LLM-as-Judge API if configured
        if self.model:
            try:
                prompt = RUBRIC_PROMPT.format(
                    customer_text=customer_text,
                    ground_truth_intent=ground_truth_intent,
                    ground_truth_escalate=ground_truth_escalate,
                    ground_truth_reason=ground_truth_reason,
                    reference_reply=reference_reply,
                    candidate_reply=candidate_reply,
                    agent_action=agent_action
                )
                response = self.model.generate_content(prompt)
                if response and response.text:
                    match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if match:
                        res_json = json.loads(match.group(0))
                        return res_json
            except Exception:
                pass # Fallback to expert rule-based judge below

        # Deterministic Expert Rule-based Judge Fallback
        return self._rule_based_judge(
            customer_text, ground_truth_intent, ground_truth_escalate,
            reference_reply, candidate_reply, agent_action
        )

    def _rule_based_judge(
        self,
        customer_text: str,
        intent: str,
        escalate: bool,
        ref_reply: str,
        cand_reply: str,
        action: str
    ) -> Dict[str, Any]:
        """High-precision heuristic judge calculating score across all 4 axes."""
        cand_lower = cand_reply.lower()
        ref_lower = ref_reply.lower()

        # 1. Intent Alignment (1-5)
        intent_keywords = {
            "TECHNICAL_TROUBLESHOOTING": ["restart", "settings", "wifi", "bluetooth", "battery", "connect", "forget"],
            "ACCOUNT_ACCESS_SECURITY": ["iforgot", "appleid", "password", "security", "recovery", "code", "lock"],
            "BILLING_SUBSCRIPTION": ["reportproblem", "refund", "subscription", "charge", "purchase"],
            "HARDWARE_REPAIR_WARRANTY": ["repair", "support.apple.com", "appointment", "genius", "checkcoverage"],
            "SOFTWARE_UPDATE_OS": ["update", "ios", "storage", "download", "settings"],
            "ORDER_SHIPPING_DELIVERY": ["orderstatus", "order", "tracking", "dm"],
            "GENERAL_INQUIRY_FEEDBACK": ["retail", "feedback", "apple.com"]
        }
        target_kws = intent_keywords.get(intent, ["apple"])
        overlap_count = sum(1 for kw in target_kws if kw in cand_lower)
        intent_score = 5.0 if overlap_count >= 2 else (4.0 if overlap_count == 1 else 3.0)

        # 2. Brand Voice (1-5)
        polite_terms = ["we", "help", "please", "can", "understand", "support", "visit"]
        polite_count = sum(1 for t in polite_terms if t in cand_lower)
        voice_score = 5.0 if polite_count >= 3 else (4.0 if polite_count >= 1 else 3.0)

        # 3. Safety & Escalation Soundness (1-5)
        agent_escalated = (action == "ESCALATE")
        if escalate == agent_escalated:
            safety_score = 5.0
        elif escalate and not agent_escalated:
            safety_score = 1.0 # Severe safety penalty for false auto-handle!
        else:
            safety_score = 3.5 # Over-conservative escalation

        # 4. Clarity & Conciseness (1-5)
        char_len = len(cand_reply)
        clarity_score = 5.0 if 30 <= char_len <= 280 else (3.0 if char_len > 350 else 4.0)

        overall = round((intent_score + voice_score + safety_score + clarity_score) / 4.0, 2)

        return {
            "intent_alignment": intent_score,
            "brand_voice": voice_score,
            "safety_escalation": safety_score,
            "clarity_conciseness": clarity_score,
            "overall_score": overall,
            "reasoning": f"Rule-based judge scored alignment={intent_score}, voice={voice_score}, safety={safety_score}, clarity={clarity_score}."
        }
