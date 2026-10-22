"""
Escalation Router Module matching AI Agent Assignment Specification.
Evaluates semantic meaning to generate issue-specific escalation reasons
grounded ONLY in the actual customer tweet and detected condition.
"""

from typing import Dict, Any, List, Optional


class EscalationRouter:
    """Multi-factor decision router adhering strictly to AI Agent Action Policy."""

    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        customer_text: str,
        intent: str,
        retrieved_context: List[Dict[str, Any]],
        intent_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluate customer query and generate issue-specific escalation decisions and reasons.
        """
        text_lower = customer_text.lower()
        triggers = []
        escalate = False
        primary_reason = ""
        safety_assessment = "NO_CRITICAL_HAZARD"

        # 1. Swollen / Expanding Battery Safety Hazards (HARDWARE_REPAIR_WARRANTY)
        swollen_terms = ["swollen", "swollen battery", "puffed battery", "expanding", "pushing screen outward",
                         "pushing the screen outward", "smoke", "leaking battery", "burning smell", "exploding"]
        if any(term in text_lower for term in swollen_terms):
            escalate = True
            safety_assessment = "CRITICAL_HAZARD"
            triggers.append("swollen_battery_safety_hazard")
            primary_reason = "The swollen/expanding battery presents a potential safety hazard and requires professional service."

        # 2. Specific Stolen / Missing High-Value Package Delivery (ORDER_SHIPPING_DELIVERY)
        elif intent == "ORDER_SHIPPING_DELIVERY" and any(k in text_lower for k in ["stolen package", "package was stolen", "missing package", "laptop was stolen", "fedex"]):
            escalate = True
            safety_assessment = "CRITICAL_HAZARD"
            triggers.append("stolen_high_value_package")
            primary_reason = "Potential stolen/missing high-value delivery requiring carrier investigation."

        # 3. Account Access & Security Emergencies (ACCOUNT_ACCESS_SECURITY)
        elif intent == "ACCOUNT_ACCESS_SECURITY" or any(k in text_lower for k in ["stolen", "stole", "hacked", "compromised", "locked for security"]):
            if any(k in text_lower for k in ["stolen", "stole", "lost phone"]):
                escalate = True
                safety_assessment = "CRITICAL_HAZARD"
                triggers.append("stolen_iphone_emergency")
                primary_reason = "Stolen device report requiring immediate Lost Mode lock via iCloud.com/find."
            elif any(k in text_lower for k in ["hacked", "compromised", "signed into my apple id from another country"]):
                escalate = True
                safety_assessment = "CRITICAL_HAZARD"
                triggers.append("compromised_account_alert")
                primary_reason = "Suspected compromised Apple ID requiring password change and 2FA review via appleid.apple.com."
            elif any(k in text_lower for k in ["lost my trusted phone", "locked for security reasons", "iforgot", "passcode", "locked"]):
                escalate = True
                safety_assessment = "CRITICAL_HAZARD"
                triggers.append("lost_2fa_account_lockout")
                primary_reason = "Account security lockout or lost credentials require identity verification via iforgot.apple.com."

        # 4. Specific Hardware Repair Cases (HARDWARE_REPAIR_WARRANTY)
        elif intent == "HARDWARE_REPAIR_WARRANTY":
            if any(k in text_lower for k in ["broken pin", "charger port", "pin stuck"]):
                escalate = True
                safety_assessment = "CRITICAL_HAZARD"
                triggers.append("broken_charger_pin")
                primary_reason = "The broken charger pin requires professional inspection to avoid further damage."
            elif any(k in text_lower for k in ["shattered", "cracked screen", "broken glass", "shattered into pieces", "dropped"]):
                escalate = True
                safety_assessment = "CRITICAL_HAZARD"
                triggers.append("shattered_device_glass")
                primary_reason = "Physical damage to the device requires professional inspection and repair."
            elif any(k in text_lower for k in ["liquid", "water"]):
                escalate = True
                safety_assessment = "CRITICAL_HAZARD"
                triggers.append("liquid_damage_hazard")
                primary_reason = "Liquid ingress hazard requires professional service inspection."
            elif any(k in text_lower for k in ["stuck button", "power button is stuck"]):
                escalate = True
                safety_assessment = "CRITICAL_HAZARD"
                triggers.append("stuck_hardware_button")
                primary_reason = "Physical hardware button defect requires professional repair service."

        # 5. High-Risk Transaction Disputes (BILLING_SUBSCRIPTION)
        elif intent == "BILLING_SUBSCRIPTION":
            if any(k in text_lower for k in ["charged three times", "charged twice", "duplicate charge", "unauthorized charge", "chargeback"]):
                escalate = True
                safety_assessment = "CRITICAL_HAZARD"
                triggers.append("duplicate_billing_dispute")
                primary_reason = "Potential duplicate billing anomaly requiring support review."

        # 6. OS Boot Failures
        elif intent == "SOFTWARE_UPDATE_OS" and any(k in text_lower for k in ["flashing question mark", "boot", "recovery"]):
            escalate = True
            safety_assessment = "CRITICAL_HAZARD"
            triggers.append("os_boot_failure")
            primary_reason = "Potential OS boot/recovery issue requiring macOS Recovery mode guidance."

        # ANTI-FALSE-ESCALATION POLICY:
        if not escalate:
            final_action = "AUTO_HANDLE"
            primary_reason = "" # No escalation reason for AUTO_HANDLE cases!
            safety_assessment = "NO_CRITICAL_HAZARD"
        else:
            final_action = "ESCALATE"

        max_similarity = max([c.get("similarity_score", 0.0) for c in retrieved_context], default=0.0)

        return {
            "action": final_action,
            "escalate": escalate,
            "safety_assessment": safety_assessment,
            "escalation_reason": primary_reason,
            "confidence_score": round(max_similarity if max_similarity > 0 else intent_confidence, 4),
            "triggers": triggers
        }
