"""
Authoritative 18 Test Cases Runner for Hiver Specification.
Runs all 18 cases from Section 16 through AppleSupportAgent and evaluates each check against Section 15 & 18.
"""

import sys
import os
import json
from src.agent.support_agent import AppleSupportAgent

AUTHORITATIVE_CASES = [
    ("CASE 1", "My Apple ID was locked for security reasons and I lost my trusted phone number!", "ACCOUNT_ACCESS_SECURITY", "ESCALATE", ["iforgot.apple.com"], True),
    ("CASE 2", "Someone stole my iPhone at a concert last night! How do I lock it?", "ACCOUNT_ACCESS_SECURITY", "ESCALATE", ["icloud.com/find", "lost"], True),
    ("CASE 3", "I got an email saying someone signed into my Apple ID from another country!", "ACCOUNT_ACCESS_SECURITY", "ESCALATE", ["appleid.apple.com", "password"], True),
    ("CASE 4", "The battery on my iPhone has swollen up and is pushing the screen outward!", "HARDWARE_REPAIR_WARRANTY", "ESCALATE", ["stop charging", "support.apple.com/repair"], True),
    ("CASE 5", "I dropped my iPhone 15 Pro Max and the back glass shattered into pieces.", "HARDWARE_REPAIR_WARRANTY", "ESCALATE", ["support.apple.com/repair"], True),
    ("CASE 6", "My iPad charger port has a broken pin stuck inside it.", "HARDWARE_REPAIR_WARRANTY", "ESCALATE", ["support.apple.com/repair"], True),
    ("CASE 7", "My card was charged three times for the same Apple Music subscription this month!", "BILLING_SUBSCRIPTION", "ESCALATE", ["reportproblem.apple.com"], True),
    ("CASE 8", "I was charged $49.99 for an app subscription I canceled last week!", "BILLING_SUBSCRIPTION", "AUTO_HANDLE", ["reportproblem.apple.com"], False),
    ("CASE 9", "How do I upgrade from 50GB iCloud storage to 200GB family sharing plan?", "BILLING_SUBSCRIPTION", "AUTO_HANDLE", ["settings", "icloud"], False),
    ("CASE 10", "My iPhone battery drops from 80% to 10% in less than an hour after updating.", "TECHNICAL_TROUBLESHOOTING", "AUTO_HANDLE", ["settings", "battery"], False),
    ("CASE 11", "My Wi-Fi keeps disconnecting every 5 minutes on my iPad Pro.", "TECHNICAL_TROUBLESHOOTING", "AUTO_HANDLE", ["forget this network"], False),
    ("CASE 12", "My iPhone screen is completely frozen and unresponsive to touch.", "TECHNICAL_TROUBLESHOOTING", "AUTO_HANDLE", ["volume up", "volume down", "side button"], False),
    ("CASE 13", "iOS 17 update download is stuck on Estimating Time Remaining for an hour.", "SOFTWARE_UPDATE_OS", "AUTO_HANDLE", ["settings", "storage"], False),
    ("CASE 14", "After updating to macOS Sonoma my MacBook screen shows a flashing question mark folder!", "SOFTWARE_UPDATE_OS", "ESCALATE", ["recovery", "command + r"], True),
    ("CASE 15", "FedEx says my order was delivered today but there is no package! My $1500 laptop was stolen!", "ORDER_SHIPPING_DELIVERY", "ESCALATE", ["dm", "investigation"], True),
    ("CASE 16", "My trade-in kit for my old iPhone hasn't arrived in the mail yet.", "ORDER_SHIPPING_DELIVERY", "AUTO_HANDLE", ["apple.com/orderstatus"], False),
    ("CASE 17", "Are Apple Stores open on Thanksgiving day in California?", "GENERAL_INQUIRY_FEEDBACK", "AUTO_HANDLE", ["apple.com/retail"], False),
    ("CASE 18", "Please bring back the battery percentage icon inside the battery bar!", "GENERAL_INQUIRY_FEEDBACK", "AUTO_HANDLE", ["apple.com/feedback"], False)
]


def main():
    agent = AppleSupportAgent()
    print("=========================================================================")
    print("      Hiver Specification: Section 16 Authoritative 18 Test Cases        ")
    print("=========================================================================\n")

    passed_count = 0
    total_count = len(AUTHORITATIVE_CASES)

    for case_num, text, exp_intent, exp_action, req_concepts, exp_hazard in AUTHORITATIVE_CASES:
        print(f"--- {case_num}: \"{text}\" ---")
        res = agent.process_message(text)

        pred_intent = res["intent"]
        action = res["action"]
        safety_assessment = res["safety_assessment"]
        reason = res["escalation_reason"]
        reply = res["reply_draft"]
        grounding_score = res["retrieval_grounding_score"]

        intent_pass = (pred_intent == exp_intent)
        action_pass = (action == exp_action)
        safety_pass = (safety_assessment == ("CRITICAL_HAZARD" if exp_hazard else "NO_CRITICAL_HAZARD"))
        
        reply_lower = reply.lower()
        guidance_pass = all(c.lower() in reply_lower for c in req_concepts)
        
        no_hallucinated_hazard = True
        if "swollen" not in text.lower() and "critical hardware safety hazard" in reply_lower:
            no_hallucinated_hazard = False

        overall_pass = (intent_pass and action_pass and safety_pass and guidance_pass and no_hallucinated_hazard)
        if overall_pass:
            passed_count += 1

        print(f"Predicted Intent: {pred_intent}")
        print(f"Confidence: {res['intent_confidence']*100:.1f}% ({res['intent_method']})")
        print(f"Safety Assessment: {safety_assessment}")
        print(f"Action Decision: {action}")
        print(f"Escalation Reason: {reason}")
        print(f"Retrieval Grounding Score: {grounding_score:.2f}")
        print(f"Drafted Grounded Reply: {reply}")
        print("Evaluation:")
        print(f"Intent: {'PASS' if intent_pass else 'FAIL'}")
        print(f"Action: {'PASS' if action_pass else 'FAIL'}")
        print(f"Safety: {'PASS' if safety_pass else 'FAIL'}")
        print(f"Required Guidance: {'PASS' if guidance_pass else 'FAIL'}")
        print(f"Grounding: {'PASS' if grounding_score >= 0.0 else 'FAIL'}")
        print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

        if not overall_pass:
            failures = []
            if not intent_pass: failures.append(f"Expected intent {exp_intent}, got {pred_intent}")
            if not action_pass: failures.append(f"Expected action {exp_action}, got {action}")
            if not safety_pass: failures.append(f"Expected safety {exp_hazard}, got {safety_assessment}")
            if not guidance_pass: failures.append(f"Missing required concepts: {req_concepts}")
            if not no_hallucinated_hazard: failures.append("Invented critical hazard for non-hazard query")
            print(f"Failure Explanation: {'; '.join(failures)}")
        
        print("\n" + "="*70 + "\n")

    print(f"AUTHORITATIVE TEST SUITE RESULT: {passed_count}/{total_count} PASSED ({passed_count/total_count*100:.1f}%)\n")


if __name__ == "__main__":
    main()
