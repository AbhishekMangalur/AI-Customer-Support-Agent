"""
Automated Evaluation Harness for AI Customer Support Agent.
Benchmarks Trivial Baseline, Simple Baseline, and Proposed Agent on Golden Set.
"""

import os
import json
from typing import Dict, Any, List, Optional
from src.data.loader import TwitterDataLoader
from src.agent.support_agent import AppleSupportAgent
from src.baselines.trivial_baseline import TrivialBaselineAgent
from src.baselines.simple_baseline import SimpleBaselineAgent
from src.eval.metrics import compute_intent_metrics, compute_escalation_metrics, compute_text_quality_metrics
from src.eval.llm_judge import LLMJudgeEvaluator
from src.eval.calibration import compute_judge_calibration


class EvaluationHarness:
    """Benchmark runner for AI Agent Specification."""

    def __init__(self, golden_set_path: str = "data/golden/golden_set.jsonl"):
        if not os.path.exists(golden_set_path) and os.path.exists("data/golden_set.json"):
            golden_set_path = "data/golden_set.json"
        self.golden_set_path = golden_set_path
        self.golden_set = self._load_golden_set()
        self.judge = LLMJudgeEvaluator()

    def _load_golden_set(self) -> List[Dict[str, Any]]:
        """Load hand-labelled Golden Set in JSONL or JSON format."""
        if not os.path.exists(self.golden_set_path):
            raise FileNotFoundError(f"Golden set file not found at {self.golden_set_path}")
        
        if self.golden_set_path.endswith(".jsonl"):
            items = []
            with open(self.golden_set_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line.strip()))
            return items
        else:
            with open(self.golden_set_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def evaluate_agent(self, agent_name: str, agent_instance: Any, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run agent instance on Golden Set and compute complete metrics.
        """
        samples = self.golden_set[:limit] if limit else self.golden_set

        y_true_intent = []
        y_pred_intent = []

        y_true_escalate = []
        y_pred_escalate = []

        reference_replies = []
        candidate_replies = []

        human_scores = []
        judge_scores = []
        per_sample_details = []

        print(f"Evaluating {agent_name} on {len(samples)} samples...")

        for sample in samples:
            c_text = sample["customer_text"]
            gt_intent = sample["ground_truth_intent"]
            gt_escalate = sample["ground_truth_escalate"]
            gt_reason = sample.get("ground_truth_escalation_reason", "")
            ref_reply = sample["reference_reply"]
            human_score = sample.get("human_quality_score", 5.0)

            # Process with agent
            res = agent_instance.process_message(c_text)

            pred_intent = res["intent"]
            pred_escalate = res["escalate"]
            cand_reply = res["reply_draft"]
            action = res["action"]

            # Store predictions
            y_true_intent.append(gt_intent)
            y_pred_intent.append(pred_intent)

            y_true_escalate.append(gt_escalate)
            y_pred_escalate.append(pred_escalate)

            reference_replies.append(ref_reply)
            candidate_replies.append(cand_reply)
            human_scores.append(human_score)

            # Score with Judge
            judge_res = self.judge.evaluate_reply(
                customer_text=c_text,
                ground_truth_intent=gt_intent,
                ground_truth_escalate=gt_escalate,
                ground_truth_reason=gt_reason,
                reference_reply=ref_reply,
                candidate_reply=cand_reply,
                agent_action=action
            )

            judge_score = judge_res["overall_score"]
            judge_scores.append(judge_score)

            per_sample_details.append({
                "id": sample["id"],
                "customer_text": c_text,
                "ground_truth_intent": gt_intent,
                "predicted_intent": pred_intent,
                "ground_truth_escalate": gt_escalate,
                "predicted_escalate": pred_escalate,
                "action": action,
                "reference_reply": ref_reply,
                "candidate_reply": cand_reply,
                "human_score": human_score,
                "judge_score": judge_score,
                "judge_breakdown": judge_res
            })

        # Calculate metrics
        intent_metrics = compute_intent_metrics(y_true_intent, y_pred_intent)
        escalation_metrics = compute_escalation_metrics(y_true_escalate, y_pred_escalate)
        text_metrics = compute_text_quality_metrics(reference_replies, candidate_replies)
        calibration_metrics = compute_judge_calibration(human_scores, judge_scores)

        avg_judge_score = float(sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0

        return {
            "agent_name": agent_name,
            "sample_count": len(samples),
            "intent_metrics": intent_metrics,
            "escalation_metrics": escalation_metrics,
            "text_metrics": text_metrics,
            "avg_judge_score": round(avg_judge_score, 2),
            "calibration": calibration_metrics,
            "sample_results": per_sample_details
        }

    def run_full_benchmark(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Run benchmark across all 3 agent architectures."""
        trivial_agent = TrivialBaselineAgent()
        simple_agent = SimpleBaselineAgent()
        proposed_agent = AppleSupportAgent()

        trivial_res = self.evaluate_agent("Trivial Baseline (Majority + Static)", trivial_agent, limit=limit)
        simple_res = self.evaluate_agent("Simple Baseline (Zero-Shot + Keywords)", simple_agent, limit=limit)
        proposed_res = self.evaluate_agent("Proposed Agent (RAG + Escalation Router)", proposed_agent, limit=limit)

        return {
            "trivial_baseline": trivial_res,
            "simple_baseline": simple_res,
            "proposed_agent": proposed_res
        }
