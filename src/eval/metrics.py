"""
Metrics calculation module for AI Support Agent evaluation.
Computes classification metrics (Accuracy, Precision, Recall, F1), escalation metrics, and text generation metrics (ROUGE, BLEU, Cosine Similarity).
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute intent classification Accuracy, Precision, Recall, Macro F1, and Weighted F1."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "weighted_f1": round(float(w_f1), 4),
        "per_class_report": report
    }


def compute_escalation_metrics(y_true_escalate: List[bool], y_pred_escalate: List[bool]) -> Dict[str, Any]:
    """Compute binary escalation Precision, Recall, F1, and Safety False Negative Rate."""
    acc = accuracy_score(y_true_escalate, y_pred_escalate)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_escalate, y_pred_escalate, average='binary', pos_label=True, zero_division=0)

    # False negative count (high risk cases incorrectly auto-handled)
    false_negatives = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if t and not p)
    fnr = false_negatives / sum(y_true_escalate) if sum(y_true_escalate) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "false_negative_rate": round(float(fnr), 4),
        "false_negatives_count": false_negatives
    }


def compute_n_gram_bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """Compute smoothed sentence BLEU score."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not cand_tokens:
        return 0.0

    p_ns = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(zip(*[ref_tokens[i:] for i in range(n)]))
        cand_ngrams = Counter(zip(*[cand_tokens[i:] for i in range(n)]))

        overlap = sum((cand_ngrams & ref_ngrams).values())
        total = max(len(cand_tokens) - n + 1, 1)
        p_n = (overlap + 1.0) / (total + 1.0) # Add-1 smoothing
        p_ns.append(p_n)

    # Brevity penalty
    c = len(cand_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / max(c, 1))

    log_p_sum = sum(math.log(p) for p in p_ns) / max_n
    return round(bp * math.exp(log_p_sum), 4)


def compute_text_quality_metrics(references: List[str], candidates: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and Cosine Semantic Similarity."""
    bleu_scores = [compute_n_gram_bleu(ref, cand) for ref, cand in zip(references, candidates)]
    avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0

    rouge_1, rouge_2, rouge_l = 0.0, 0.0, 0.0
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        r1s, r2s, rls = [], [], []
        for ref, cand in zip(references, candidates):
            scores = scorer.score(ref, cand)
            r1s.append(scores['rouge1'].fmeasure)
            r2s.append(scores['rouge2'].fmeasure)
            rls.append(scores['rougeL'].fmeasure)
        rouge_1 = float(np.mean(r1s))
        rouge_2 = float(np.mean(r2s))
        rouge_l = float(np.mean(rls))

    # Cosine Similarity via TF-IDF
    vectorizer = TfidfVectorizer().fit(references + candidates)
    ref_vecs = vectorizer.transform(references)
    cand_vecs = vectorizer.transform(candidates)

    cos_sims = []
    for i in range(len(references)):
        sim = cosine_similarity(ref_vecs[i], cand_vecs[i])[0][0]
        cos_sims.append(sim)

    avg_cos_sim = float(np.mean(cos_sims)) if cos_sims else 0.0

    return {
        "bleu": round(avg_bleu, 4),
        "rouge1": round(rouge_1, 4),
        "rouge2": round(rouge_2, 4),
        "rougeL": round(rouge_l, 4),
        "cosine_similarity": round(avg_cos_sim, 4)
    }
