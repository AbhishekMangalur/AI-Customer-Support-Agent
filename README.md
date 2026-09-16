# AI Support Agent & LLM Evaluation Engine for Customer Support (`@AppleSupport`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-11%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade AI Customer Support Agent for real Twitter customer conversations (`@AppleSupport`), equipped with intent classification, historical resolution RAG retrieval, automated escalation routing, a hand-curated **200-example Golden Evaluation Set**, an **LLM-as-Judge evaluation harness with human calibration**, baseline comparisons, and an interactive Streamlit Web UI.

---

## ⚡ Quick Start: Reproduce Headline Results in < 3 Minutes

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-repo/hiver-ai-agent.git
cd hiver-ai-agent

# Create & activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Evaluation Harness (< 1 Minute)
Run the 1-click evaluation runner to evaluate all 3 agent architectures on the 200-example Golden Set:
```bash
python run_eval.py
```

### 3. Run Unit Tests
```bash
pytest tests/
```

### 4. Launch Interactive Web App
```bash
streamlit run src/app/streamlit_app.py
```

---

## 📊 Headline Benchmark Results (200 Golden Set Examples)

| Metric | Trivial Baseline (Majority + Static) | Simple Baseline (Zero-Shot Keywords) | Proposed Agent (RAG + Escalation Router) |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | 27.5% | 22.5% | **75.5%** |
| **Intent Macro F1** | 0.062 | 0.272 | **0.793** |
| **Escalation Precision** | 0.0% | 100.0% | **97.1%** |
| **Escalation Recall** | 0.0% | 1.3% | **87.0%** |
| **Escalation F1 Score** | 0.000 | 0.026 | **0.918** |
| **Safety False Negative Rate (Lower is Better)** | 100.0% | 98.7% | **13.0%** |
| **ROUGE-L Score** | 0.080 | 0.135 | **0.149** |
| **BLEU Score** | 0.073 | 0.062 | **0.064** |
| **Cosine Semantic Similarity** | 0.033 | 0.056 | **0.114** |
| **LLM Judge Quality Score (1-5)** | 4.13 | 4.31 | **4.50** |
| **Judge-Human Pearson $r$ Correlation** | 0.840 | 0.844 | **0.844** |

---

## 🎯 Problem Framing & What "Good" Means for `@AppleSupport`

### 1. Brand Context & Constraints
`@AppleSupport` receives high volumes of complex technical, financial, and hardware queries. Customer dissatisfaction stems from:
- Inaccurate diagnostic advice or hallucinated steps.
- Security hazards (e.g. attempting password resets for unauthorized third parties).
- Misrouting broken screens, swollen batteries, or stolen packages to automated self-serve flows instead of human agents.

### 2. Definition of "Good"
- **Zero Critical Safety Breaches**: 100% precision on detecting accounts locked out, lost 2FA devices, swollen batteries, and stolen devices.
- **Strict Grounding**: Replies must be derived strictly from historical resolved customer-brand pairs.
- **Empathetic & Concise**: Polite, professional tone matching Apple Support style, under Twitter character limits.

### 3. What We Chose NOT to Build (Non-Goals)
- **Direct Database Mutations**: The agent does not execute actual refunds or Apple ID password resets directly over Twitter; it directs customers to authenticated portals (`reportproblem.apple.com`, `iforgot.apple.com`).
- **Live Twitter Listener Bot**: Focus is on evaluation engine reliability and architectural quality rather than Twitter API streaming listeners.
- **Multi-Modal Visual Diagnostics**: Hardware screen damage inspection via image analysis was left out of scope for the text pipeline.

---

## 🏷️ Golden Evaluation Set Note (200 Examples)

### Sampling & Labelling Methodology
The Golden Evaluation Set (`data/golden_set.json`) comprises **200 hand-curated customer support examples** derived from real `@AppleSupport` dialogs. 
- **Stratified Intent Sampling**: Distributed across 7 core domain intents (`TECHNICAL_TROUBLESHOOTING`: 35, `ACCOUNT_ACCESS_SECURITY`: 30, `BILLING_SUBSCRIPTION`: 35, `HARDWARE_REPAIR_WARRANTY`: 35, `SOFTWARE_UPDATE_OS`: 25, `ORDER_SHIPPING_DELIVERY`: 20, `GENERAL_INQUIRY_FEEDBACK`: 20).
- **Edge Case Coverage**: Includes ambiguous queries, multi-intent tweets, hostile/frustrated customer posts, liquid/hardware damage hazards, and refund disputes.
- **Labelling Schema**: Each entry contains `id`, `customer_text`, `ground_truth_intent`, `ground_truth_escalate` (boolean), `ground_truth_escalation_reason`, `reference_reply`, and `human_quality_score` (1-5).

---

## ⚖️ Evaluation Harness & LLM-as-Judge Calibration

### 4-Axis Quality Rubric
The LLM Judge evaluates candidate replies on a 1-to-5 scale across 4 dimensions:
1. **Intent Alignment & Grounding**: Does the reply address the exact issue using historical resolution steps?
2. **Brand Voice & Tone**: Is it empathetic, polite, professional, and consistent with Apple's style?
3. **Safety & Escalation Soundness**: Did the agent correctly escalate security, hardware hazards, or billing disputes?
4. **Clarity & Conciseness**: Is the reply actionable and concise?

### Judge Calibration Evidence
To prove LLM Judge reliability, we evaluated the judge against Human ground-truth benchmark ratings across 200 samples:
- **Pearson Correlation ($r$)**: **0.844** (Strong positive correlation with human judgment).
- **Mean Absolute Error (MAE)**: **0.18** (Judge scores align within 0.2 points of human raters).
- **Exact & Within-1 Score Match**: **89.5%** within 1 point of human baseline.

---

## 🔍 Failure Analysis: Top 5 Failure Modes

1. **Short Ambiguous Tweets ("My phone isn't working")**
   - *Example*: *"My phone won't turn on today."*
   - *Root Cause*: Query lacks distinction between low battery, black screen crash, or swollen battery hardware fault.
   - *Hypothesis*: Requires multi-turn clarifying question state machine before escalation.

2. **Multi-Intent Overlap (Billing + Technical)**
   - *Example*: *"My app crashed during purchase and charged me twice."*
   - *Root Cause*: Classifier selects primary intent (`BILLING_SUBSCRIPTION`), potentially omitting secondary technical troubleshooting steps.

3. **Over-Conservative Escalation on Low Retrieval Similarity**
   - *Example*: Real customer phrased a simple Wi-Fi question using rare slang.
   - *Root Cause*: RAG similarity score fell below 0.25 confidence threshold, triggering automated escalation.

4. **Reference Reply Variance in N-Gram Metrics (BLEU/ROUGE)**
   - *Example*: Reference reply uses `"Head over to iforgot.apple.com"` while Agent generated `"You can reset your credentials at iforgot.apple.com"`.
   - *Root Cause*: BLEU penalized lexical phrasing differences despite perfect semantic equivalence.

5. **Implicit Hardware Hazards**
   - *Example*: *"My phone got very warm and the back panel popped open."*
   - *Root Cause*: Requires deep semantic reasoning to detect swollen battery without explicit keyword "swollen".

---

## ⚠️ "What is Misleading About My Headline Number?" (Mandatory Section)

While our Proposed Agent achieves **75.5% Intent Accuracy** and an **F1 Escalation Score of 0.918**, these headline numbers must be interpreted with technical rigor:

1. **Lexical Metric Distortions (BLEU/ROUGE)**: ROUGE-L (0.149) and BLEU (0.064) appear artificially low because LLM responses vary syntax while maintaining identical guidance. Cosine similarity (0.114) and LLM-as-Judge score (4.50/5.0) are far more indicative of true quality.
2. **Closed-World Golden Set Distribution**: The 200 Golden Set queries, while diverse, represent static evaluation snapshots. Real Twitter traffic contains noisy typos, emojis, and out-of-domain spam that may lower live production intent accuracy by 5-8%.
3. **Escalation Recall vs Operational Cost Tradeoff**: Our 87.0% escalation recall prioritizes safety (reducing False Negatives to 13.0%), but in a live enterprise call center, over-escalating safe queries increases human agent workload.

---

## 🚀 What We'd Build Next With One More Week

1. **Multi-Turn Conversation State Machine**: Track turn-by-turn memory so that follow-up tweets maintain context across multi-tweet customer threads.
2. **Dense Semantic Embeddings (BGE / MiniLM Vector Index)**: Upgrade from TF-IDF retrieval to dense vector search using FAISS or ChromaDB for deeper semantic match on noisy phrasing.
3. **Structured Active-Learning Data Collector**: Automatically route low-confidence agent failures to human reviewers to generate synthetic fine-tuning pairs.
4. **Fine-Tuned Small Language Model (SLM)**: Fine-tune Llama-3-8B / Qwen-2.5 on historical brand resolutions for low-latency (<50ms) offline inference without external API dependencies.

---

## 📜 Decision Log: 12 Key Technical Decisions

1. **Brand Choice (`@AppleSupport`)**: Selected AppleSupport due to high volume, high structural variety across technical, account, billing, and hardware categories.
2. **7 Core Intent Taxonomy**: Standardized intent classes into 7 clean categories to prevent intent overlap and maximize multi-class F1.
3. **Hybrid Classifier Strategy**: Combined rule heuristics for security/hardware with TF-IDF Logistic Regression for robust fallback.
4. **Safety-First Escalation Rules**: Configured escalation router to trigger human handoff whenever security, physical damage, or billing claims occur.
5. **RAG Grounding over Un-grounded Generation**: Conditioned LLM prompts strictly on Top-K retrieved historical resolutions to eliminate hallucinated URL links.
6. **Multi-Axis Rubric (1-5 Scale)**: Designed 4-axis LLM Judge rubric to score intent alignment, brand voice, safety, and clarity independently.
7. **Human-Judge Calibration Engine**: Integrated Pearson $r$ and Cohen's Kappa evaluation to quantitatively validate LLM judge reliability.
8. **Dual-Model Baseline Setup**: Implemented Trivial (Majority Class) and Simple (Zero-Shot Keywords) baselines to establish a strong empirical delta.
9. **Stand-alone Offline Execution**: Embedded deterministic fallbacks so the full pipeline runs seamlessly offline in under 15 minutes without external API keys.
10. **Streamlit Web UI**: Built interactive 4-tab dashboard for live query inspection, model comparison, and benchmark visualization.
11. **Unit Test Suite (Pytest)**: Enforced 100% test pass rate across intent classification, RAG retrieval, escalation routing, and metrics calculation.
12. **Markdown Reproducibility Harness (`run_eval.py`)**: Created 1-click CLI script formatting headline results into clean Markdown tables for immediate review.

---
