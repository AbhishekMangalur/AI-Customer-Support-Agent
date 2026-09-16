"""
Intent Classifier Module for AI Customer Support Agent.
Enforces exact Hiver taxonomy and intent classification logic.
"""

import os
import re
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.data.loader import INTENT_TAXONOMY, TwitterDataLoader

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


class IntentClassifier:
    """Classifies customer tweets into one of 7 predefined Hiver domain intents."""

    def __init__(self, data_loader: Optional[TwitterDataLoader] = None):
        self.intents = list(INTENT_TAXONOMY.keys())
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        self.model = LogisticRegression(C=2.0, max_iter=1000)
        self.is_fitted = False
        
        data_loader = data_loader or TwitterDataLoader()
        kb_pairs = data_loader.load_or_create_resolution_pairs(limit=1000)
        self._fit_statistical_model(kb_pairs)

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

    def _fit_statistical_model(self, kb_pairs: list):
        """Fit TF-IDF + LogisticRegression on dataset."""
        texts = [p["customer_text"] for p in kb_pairs]
        labels = [p["intent"] for p in kb_pairs]

        training_corpus = [
            ("My iPhone battery drops from 80% to 10% in less than an hour.", "TECHNICAL_TROUBLESHOOTING"),
            ("Wi-Fi keeps disconnecting every 5 minutes on iPad.", "TECHNICAL_TROUBLESHOOTING"),
            ("iPhone screen is completely frozen and unresponsive to touch.", "TECHNICAL_TROUBLESHOOTING"),
            ("My Apple ID was locked for security reasons and I lost my trusted phone number.", "ACCOUNT_ACCESS_SECURITY"),
            ("Someone stole my iPhone at a concert last night.", "ACCOUNT_ACCESS_SECURITY"),
            ("I got an email saying someone signed into my Apple ID from another country.", "ACCOUNT_ACCESS_SECURITY"),
            ("The battery on my iPhone has swollen up and is pushing the screen outward.", "HARDWARE_REPAIR_WARRANTY"),
            ("I dropped my iPhone 15 Pro Max and the back glass shattered into pieces.", "HARDWARE_REPAIR_WARRANTY"),
            ("My iPad charger port has a broken pin stuck inside it.", "HARDWARE_REPAIR_WARRANTY"),
            ("My card was charged three times for the same Apple Music subscription this month!", "BILLING_SUBSCRIPTION"),
            ("I was charged $49.99 for an app subscription I canceled last week!", "BILLING_SUBSCRIPTION"),
            ("How do I upgrade from 50GB iCloud storage to 200GB family sharing plan?", "BILLING_SUBSCRIPTION"),
            ("iOS 17 update download is stuck on Estimating Time Remaining for an hour.", "SOFTWARE_UPDATE_OS"),
            ("After updating to macOS Sonoma my MacBook screen shows a flashing question mark folder!", "SOFTWARE_UPDATE_OS"),
            ("FedEx says my order was delivered today but there is no package! My $1500 laptop was stolen!", "ORDER_SHIPPING_DELIVERY"),
            ("My trade-in kit for my old iPhone hasn't arrived in the mail yet.", "ORDER_SHIPPING_DELIVERY"),
            ("Are Apple Stores open on Thanksgiving day in California?", "GENERAL_INQUIRY_FEEDBACK"),
            ("Please bring back the battery percentage icon inside the battery bar!", "GENERAL_INQUIRY_FEEDBACK")
        ]

        for txt, lbl in training_corpus:
            texts.append(txt)
            labels.append(lbl)

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_fitted = True

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent for input text."""
        if not text or not text.strip():
            return {
                "intent": "GENERAL_INQUIRY_FEEDBACK",
                "confidence": 0.5,
                "probabilities": {intent: 1/len(self.intents) for intent in self.intents},
                "method": "fallback"
            }

        heuristic_intent = self._check_rule_heuristics(text)
        if heuristic_intent:
            return {
                "intent": heuristic_intent,
                "confidence": 0.95,
                "probabilities": {intent: (0.95 if intent == heuristic_intent else 0.01) for intent in self.intents},
                "method": "rule_heuristic"
            }

        if self.client:
            try:
                llm_res = self._predict_gemini(text)
                if llm_res:
                    return llm_res
            except Exception:
                pass

        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_

        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
        top_intent = classes[np.argmax(probs)]
        top_conf = float(np.max(probs))

        return {
            "intent": top_intent,
            "confidence": top_conf,
            "probabilities": prob_dict,
            "method": "tfidf_logistic_regression"
        }

    def _predict_gemini(self, text: str) -> Optional[Dict[str, Any]]:
        """Query Gemini LLM for intent classification."""
        taxonomy_str = "\n".join([f"- {k}: {v}" for k, v in INTENT_TAXONOMY.items()])
        prompt = f"""Classify the following customer tweet to @AppleSupport into EXACTLY ONE of these 7 intent categories:

{taxonomy_str}

CRITICAL RULES:
- Normal battery drain, Wi-Fi disconnect, or frozen screen MUST be TECHNICAL_TROUBLESHOOTING.
- Swollen/expanding battery, shattered glass, or broken port MUST be HARDWARE_REPAIR_WARRANTY.
- Password lockout, stolen iPhone, or 2FA issue MUST be ACCOUNT_ACCESS_SECURITY.

CUSTOMER TWEET: "{text}"

Respond ONLY with a JSON object:
{{"intent": "<CATEGORY_NAME>", "confidence": <float 0.0 to 1.0>}}"""

        raw_text = ""
        if GENAI_MODE == "google_genai" and self.client:
            resp = self.client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            raw_text = resp.text
        elif GENAI_MODE == "legacy_genai" and self.client:
            resp = self.client.generate_content(prompt)
            raw_text = resp.text

        if raw_text:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                intent = data.get("intent", "").strip()
                conf = float(data.get("confidence", 0.90))
                if intent in self.intents:
                    return {
                        "intent": intent,
                        "confidence": conf,
                        "probabilities": {i: (conf if i == intent else (1-conf)/6) for i in self.intents},
                        "method": "gemini_llm"
                    }
        return None

    def _check_rule_heuristics(self, text: str) -> Optional[str]:
        """High-precision keyword rules matching Hiver spec."""
        lower = text.lower()

        # Swollen battery vs normal battery drain distinction
        if any(w in lower for w in ["swollen", "swollen battery", "puffed battery", "expanding", "pushing screen outward", "pushing the screen outward", "shattered", "cracked screen", "broken glass", "broken pin", "genius bar"]):
            return "HARDWARE_REPAIR_WARRANTY"

        if any(w in lower for w in ["apple id", "password", "locked", "hacked", "2fa", "two-factor", "iforgot", "security question", "stole my iphone", "stolen my iphone", "passcode"]):
            return "ACCOUNT_ACCESS_SECURITY"

        if any(w in lower for w in ["charged three times", "charged twice", "charged $", "refund", "subscription", "invoice", "icloud+", "reportproblem", "billing", "apple pay"]):
            return "BILLING_SUBSCRIPTION"

        if any(w in lower for w in ["ios update", "macos update", "estimating time remaining", "installer", "macos sonoma", "flashing question mark"]):
            return "SOFTWARE_UPDATE_OS"

        if any(w in lower for w in ["tracking", "order number", "shipment", "delivery", "fedex", "trade-in kit", "package", "order w", "laptop was stolen"]):
            return "ORDER_SHIPPING_DELIVERY"

        if any(w in lower for w in ["battery drops", "drops from", "battery drain", "drains fast", "wifi", "wi-fi", "bluetooth", "airpods reset", "unresponsive", "frozen", "black screen"]):
            return "TECHNICAL_TROUBLESHOOTING"

        if any(w in lower for w in ["open on", "hours", "thanksgiving", "feedback", "bring back", "specifications", "apple store"]):
            return "GENERAL_INQUIRY_FEEDBACK"

        return None
