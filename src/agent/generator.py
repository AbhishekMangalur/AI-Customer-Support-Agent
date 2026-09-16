"""
Grounded Reply Generator for AI Customer Support Agent.
Drafts customer responses using RAG historical resolution context and brand voice guidelines via Gemini LLM or fallback generator.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

GENAI_MODE = None
try:
    from google import genai
    GENAI_MODE = "google_genai"
except ImportError:
    try:
        import google.generativeai as legacy_genai
        GENAI_MODE = "legacy_genai"
    except ImportError:
        GENAI_MODE = None


class GroundedReplyGenerator:
    """Drafts customer support replies grounded in historical brand resolutions."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None

        if self.api_key:
            if GENAI_MODE == "google_genai":
                try:
                    self.client = genai.Client(api_key=self.api_key)
                except Exception:
                    self.client = None
            elif GENAI_MODE == "legacy_genai":
                try:
                    legacy_genai.configure(api_key=self.api_key)
                    self.client = legacy_genai.GenerativeModel('gemini-1.5-flash')
                except Exception:
                    self.client = None

    def generate_reply(
        self,
        customer_text: str,
        intent: str,
        retrieved_context: List[Dict[str, Any]],
        escalate: bool,
        escalation_reason: str
    ) -> str:
        """
        Draft a brand reply using Gemini LLM API or fallback template generator grounded in retrieved context.
        """
        relevant_context = [c for c in retrieved_context if c.get("intent") == intent and c.get("similarity_score", 0.0) >= 0.30]
        top_context = relevant_context[0] if relevant_context else (retrieved_context[0] if retrieved_context else None)

        if self.client:
            try:
                llm_reply = self._generate_gemini_reply(customer_text, intent, retrieved_context, escalate, escalation_reason)
                if llm_reply:
                    return llm_reply
            except Exception:
                pass

        return self._build_template_reply(customer_text, intent, top_context, escalate, escalation_reason)

    def _generate_gemini_reply(
        self,
        customer_text: str,
        intent: str,
        retrieved_context: List[Dict[str, Any]],
        escalate: bool,
        escalation_reason: str
    ) -> Optional[str]:
        """Generate response via Gemini LLM."""
        prompt = self._build_rag_prompt(customer_text, intent, retrieved_context, escalate, escalation_reason)

        if GENAI_MODE == "google_genai" and self.client:
            resp = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return resp.text.strip() if resp and resp.text else None
        elif GENAI_MODE == "legacy_genai" and self.client:
            resp = self.client.generate_content(prompt)
            return resp.text.strip() if resp and resp.text else None

        return None

    def _build_rag_prompt(
        self,
        customer_text: str,
        intent: str,
        retrieved_context: List[Dict[str, Any]],
        escalate: bool,
        escalation_reason: str
    ) -> str:
        """Construct RAG prompt for Gemini LLM."""
        context_str = ""
        for idx, c in enumerate(retrieved_context[:2], 1):
            context_str += f"Historical Case {idx}:\nCustomer: {c['customer_text']}\nResolution: {c['brand_response']}\n\n"

        return f"""You are the official AI Support Agent for @AppleSupport on Twitter.
Your goal is to draft a helpful, empathetic, concise, and accurate response to the customer.

CUSTOMER TWEET: "{customer_text}"
DETECTED INTENT: {intent}
ESCALATION STATUS: {"ESCALATED TO HUMAN" if escalate else "AUTO-HANDLED"}
ESCALATION REASON: {escalation_reason}

HISTORICAL RESOLUTION CONTEXT:
{context_str}

CRITICAL ROUTING & SAFETY GUIDELINES:
1. Ground your response strictly in the historical resolution steps above matching the customer's specific issue.
2. For account lockouts or password resets, direct them to iforgot.apple.com.
3. For stolen devices, direct them to iCloud.com/find.
4. For suspicious sign-ins or compromised accounts, direct them to appleid.apple.com.
5. For app charges or subscription refunds, direct them to reportproblem.apple.com.
6. For hardware repairs or swollen battery hazards, direct them to support.apple.com/repair.
7. Maintain AppleSupport's brand tone: polite, direct, clear, and reassuring under 280 characters.

Draft the final response text only:"""

    def _build_template_reply(
        self,
        customer_text: str,
        intent: str,
        top_context: Optional[Dict[str, Any]],
        escalate: bool,
        escalation_reason: str
    ) -> str:
        """Deterministic grounded fallback reply generator."""
        text_lower = customer_text.lower()

        if intent == "ACCOUNT_ACCESS_SECURITY":
            if any(k in text_lower for k in ["stolen", "stole", "lost phone", "lost device"]):
                reply = "Act immediately! Go to iCloud.com/find from any browser to mark your iPhone as Lost, lock it, and suspend Apple Pay."
            elif any(k in text_lower for k in ["hacked", "compromised", "unauthorized", "signed into", "another country"]):
                reply = "Your account safety is critical. Change your Apple ID password immediately at appleid.apple.com and review trusted devices under two-factor authentication."
            else:
                reply = "We take security seriously! You can safely unlock and reset your Apple ID password at iforgot.apple.com. Our team cannot bypass security verification over Twitter."
            
            if escalate:
                return f"{reply} Send us a DM if you need guidance with Account Recovery."
            return reply

        if top_context and top_context.get("brand_response"):
            base_resolution = top_context["brand_response"]
            if not escalate:
                return base_resolution
            else:
                return f"{base_resolution} For direct human assistance, please send us a DM with your details."

        if intent == "BILLING_SUBSCRIPTION":
            reply = "We can help you review charges. Visit reportproblem.apple.com to inspect billing history or submit refund requests."
        elif intent == "HARDWARE_REPAIR_WARRANTY":
            reply = "For repair pricing and Genius Bar appointments, please check options at support.apple.com/repair."
        elif intent == "SOFTWARE_UPDATE_OS":
            reply = "Ensure your device is connected to Wi-Fi and power. You can delete cached update files in Settings > General > Storage and try again."
        elif intent == "ORDER_SHIPPING_DELIVERY":
            reply = "You can track shipment status and delivery details anytime at apple.com/orderstatus."
        else:
            reply = "We're here to help! Please try restarting your device, and send us a DM if you need further assistance."

        if escalate:
            return f"{reply} Please send us a DM with your details so a support agent can assist you."
        return reply
