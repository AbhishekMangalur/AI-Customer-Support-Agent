"""
Standalone 1-Click Evaluation Runner for AI Agent.
Runs the evaluation harness on the Golden Set and prints headline results table.
"""

import sys
import os
import json
from src.eval.harness import EvaluationHarness


def main():
    print("=========================================================================")
    print("    AI Support Agent Evaluation Harness    ")
    print("=========================================================================\n")

    golden_path = os.path.join(os.path.dirname(__file__), "data", "golden_set.json")
    harness = EvaluationHarness(golden_set_path=golden_path)

    print("Running evaluation on 200 Golden Set examples across 3 agent architectures...\n")
    results = harness.run_full_benchmark()

    print("\n=========================================================================")
    print("                         HEADLINE RESULTS SUMMARY                        ")
    print("=========================================================================")

    header = f"| {'Metric':<30} | {'Trivial Baseline':<18} | {'Simple Baseline':<18} | {'Proposed Agent':<18} |"
    divider = f"|{'-'*32}|{'-'*20}|{'-'*20}|{'-'*20}|"

    print(header)
    print(divider)

    tb = results["trivial_baseline"]
    sb = results["simple_baseline"]
    pa = results["proposed_agent"]

    rows = [
        ("Intent Accuracy", f"{tb['intent_metrics']['accuracy']*100:.1f}%", f"{sb['intent_metrics']['accuracy']*100:.1f}%", f"{pa['intent_metrics']['accuracy']*100:.1f}%"),
        ("Intent Macro F1", f"{tb['intent_metrics']['macro_f1']:.3f}", f"{sb['intent_metrics']['macro_f1']:.3f}", f"{pa['intent_metrics']['macro_f1']:.3f}"),
        ("Escalation Precision", f"{tb['escalation_metrics']['precision']*100:.1f}%", f"{sb['escalation_metrics']['precision']*100:.1f}%", f"{pa['escalation_metrics']['precision']*100:.1f}%"),
        ("Escalation Recall", f"{tb['escalation_metrics']['recall']*100:.1f}%", f"{sb['escalation_metrics']['recall']*100:.1f}%", f"{pa['escalation_metrics']['recall']*100:.1f}%"),
        ("Escalation F1 Score", f"{tb['escalation_metrics']['f1']:.3f}", f"{sb['escalation_metrics']['f1']:.3f}", f"{pa['escalation_metrics']['f1']:.3f}"),
        ("Safety False Negative Rate", f"{tb['escalation_metrics']['false_negative_rate']*100:.1f}%", f"{sb['escalation_metrics']['false_negative_rate']*100:.1f}%", f"{pa['escalation_metrics']['false_negative_rate']*100:.1f}%"),
        ("ROUGE-L Score", f"{tb['text_metrics']['rougeL']:.3f}", f"{sb['text_metrics']['rougeL']:.3f}", f"{pa['text_metrics']['rougeL']:.3f}"),
        ("BLEU Score", f"{tb['text_metrics']['bleu']:.3f}", f"{sb['text_metrics']['bleu']:.3f}", f"{pa['text_metrics']['bleu']:.3f}"),
        ("Cosine Similarity", f"{tb['text_metrics']['cosine_similarity']:.3f}", f"{sb['text_metrics']['cosine_similarity']:.3f}", f"{pa['text_metrics']['cosine_similarity']:.3f}"),
        ("LLM Judge Avg Score (1-5)", f"{tb['avg_judge_score']:.2f}", f"{sb['avg_judge_score']:.2f}", f"{pa['avg_judge_score']:.2f}"),
        ("Judge-Human Cohen's Kappa", f"{tb['calibration']['cohens_kappa_quadratic']:.3f}", f"{sb['calibration']['cohens_kappa_quadratic']:.3f}", f"{pa['calibration']['cohens_kappa_quadratic']:.3f}"),
        ("Judge-Human Pearson r", f"{tb['calibration']['pearson_r']:.3f}", f"{sb['calibration']['pearson_r']:.3f}", f"{pa['calibration']['pearson_r']:.3f}")
    ]

    for label, v1, v2, v3 in rows:
        print(f"| {label:<30} | {v1:<18} | {v2:<18} | {v3:<18} |")

    print(divider)
    print("\nFull evaluation finished successfully! Results stored in benchmark execution logs.\n")


if __name__ == "__main__":
    main()
