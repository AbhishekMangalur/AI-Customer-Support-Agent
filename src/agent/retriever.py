"""
Resolution Retriever (RAG Engine) for AI Customer Support Agent.
Retrieves historical resolution pairs for incoming customer queries with normalized grounding scores.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data.loader import TwitterDataLoader


def normalize_grounding_score(score: float) -> float:
    """
    Normalizes any retrieval grounding score to the range [0.00, 1.00].
    Clamps negative values to 0.00 and values >= 1.00 to 1.00.
    Returns float rounded to 2 decimal places.
    """
    if score is None or np.isnan(score):
        return 0.00
    val = float(score)
    if val <= 0.0:
        return 0.00
    if val >= 1.0:
        return 1.00
    return round(val, 2)


class ResolutionRetriever:
    """RAG retriever indexing historical @AppleSupport resolution pairs."""

    def __init__(self, data_loader: Optional[TwitterDataLoader] = None):
        self.data_loader = data_loader or TwitterDataLoader()
        self.kb_pairs = self.data_loader.load_or_create_resolution_pairs(limit=1000)
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.texts = [p["customer_text"] for p in self.kb_pairs]
        
        if self.texts:
            self.doc_vectors = self.vectorizer.fit_transform(self.texts)
        else:
            self.doc_vectors = None

    def retrieve(self, query: str, intent: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar historical resolutions for a customer query.
        Calculates normalized similarity scores strictly in [0.00, 1.00].
        """
        if not query or not self.texts or self.doc_vectors is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]

        results = []
        for idx, sim_score in enumerate(similarities):
            item = self.kb_pairs[idx]
            
            raw_score = float(sim_score)
            norm_score = normalize_grounding_score(raw_score)
                
            results.append({
                "id": item["id"],
                "intent": item.get("intent"),
                "customer_text": item["customer_text"],
                "brand_response": item["brand_response"],
                "resolution_summary": item.get("resolution_summary", item["brand_response"]),
                "raw_similarity_score": round(raw_score, 4),
                "similarity_score": norm_score
            })

        # Sort descending by normalized similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
