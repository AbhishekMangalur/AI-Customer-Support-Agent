"""
Judge Calibration & Human Agreement Module.
Calculates statistical agreement (Cohen's Kappa, Pearson r, MAE, Exact Match %)
between Human evaluators and LLM-as-Judge scores.
"""

import numpy as np
from typing import List, Dict, Any
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


def compute_judge_calibration(
    human_scores: List[float],
    judge_scores: List[float]
) -> Dict[str, Any]:
    """
    Compute statistical calibration metrics comparing Human ratings against LLM Judge ratings.
    """
    if not human_scores or not judge_scores or len(human_scores) != len(judge_scores):
        return {
            "cohens_kappa": 0.0,
            "pearson_r": 0.0,
            "spearman_r": 0.0,
            "mae": 0.0,
            "exact_match_pct": 0.0,
            "within_1_pct": 0.0
        }

    h_arr = np.array(human_scores, dtype=float)
    j_arr = np.array(judge_scores, dtype=float)

    # 1. Pearson Correlation
    p_corr, _ = pearsonr(h_arr, j_arr)
    p_corr = float(p_corr) if not np.isnan(p_corr) else 0.0

    # 2. Spearman Correlation
    s_corr, _ = spearmanr(h_arr, j_arr)
    s_corr = float(s_corr) if not np.isnan(s_corr) else 0.0

    # 3. Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(h_arr - j_arr)))

    # 4. Discrete Binned Cohen's Kappa (discretized to nearest integer score 1 to 5)
    h_binned = np.clip(np.round(h_arr), 1, 5).astype(int)
    j_binned = np.clip(np.round(j_arr), 1, 5).astype(int)
    
    kappa = float(cohen_kappa_score(h_binned, j_binned, weights='quadratic'))

    # 5. Exact and Within-1 Score Percentages
    exact = float(np.mean(h_binned == j_binned) * 100)
    within_1 = float(np.mean(np.abs(h_arr - j_arr) <= 1.0) * 100)

    return {
        "cohens_kappa_quadratic": round(kappa, 4),
        "pearson_r": round(p_corr, 4),
        "spearman_r": round(s_corr, 4),
        "mae": round(mae, 4),
        "exact_match_pct": round(exact, 2),
        "within_1_pct": round(within_1, 2),
        "sample_size": len(human_scores)
    }
