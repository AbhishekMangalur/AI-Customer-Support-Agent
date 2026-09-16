"""
Streamlit Web Application for AI Customer Support Agent Demonstration.
Features live query testing, side-by-side baseline comparisons, RAG context inspector,
escalation decision reasoning, and Golden Set benchmark visualizer.
"""

import sys
import os
import json
import streamlit as st
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agent.support_agent import AppleSupportAgent
from src.baselines.trivial_baseline import TrivialBaselineAgent
from src.baselines.simple_baseline import SimpleBaselineAgent
from src.eval.harness import EvaluationHarness

st.set_page_config(
    page_title="Hiver AI Support Agent - @AppleSupport",
    layout="wide"
)

st.title("AI Customer Support Agent & LLM Evaluation Harness")
st.caption("Hiver SDE Intern Take-Home Assignment | Target Brand: `@AppleSupport`")

@st.cache_resource
def load_agents():
    return {
        "proposed": AppleSupportAgent(),
        "trivial": TrivialBaselineAgent(),
        "simple": SimpleBaselineAgent()
    }

agents = load_agents()

tabs = st.tabs(["⚡ Live Agent Playground", "⚔️ Side-by-Side Model Comparison", "📊 Golden Set Benchmark Results", "📜 Methodological Report & Decision Log"])

# TAB 1: Live Agent Playground
with tabs[0]:
    st.subheader("Test Customer Queries in Real-Time")
    
    col_input, col_preset = st.columns([3, 1])
    with col_preset:
        preset = st.selectbox(
            "Sample Test Queries",
            [
                "Custom Query...",
                "My iPhone battery drops from 80% to 10% after iOS update.",
                "My Apple ID was locked for security reasons and I lost my trusted phone number!",
                "Someone stole my iPhone at a concert last night! How do I lock it?",
                "The battery on my iPhone has swollen up and is pushing the screen outward!",
                "I dropped my iPhone 15 Pro Max and the back glass shattered into pieces.",
                "My iPad charger port has a broken pin stuck inside it.",
                "I was charged $49.99 for an app subscription I canceled last week!",
                "iOS update stuck on Estimating Time Remaining."
            ]
        )
    
    with col_input:
        if preset != "Custom Query...":
            default_text = preset
        else:
            default_text = "My iPhone Wi-Fi keeps disconnecting every 5 minutes."
            
        user_query = st.text_area("Customer Tweet Input:", value=default_text, height=100)

    if st.button("Process Query", type="primary"):
        with st.spinner("Processing query through Intent Classifier, RAG Retriever, and Escalation Router..."):
            res = agents["proposed"].process_message(user_query)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted Intent", res["intent"])
            st.caption(f"Confidence: {res['intent_confidence']*100:.1f}% ({res['intent_method']})")
        with c2:
            st.metric("Action Decision", res["action"])
            st.caption(f"Safety Assessment: {res['safety_assessment']}")
        with c3:
            st.metric("Retrieval Grounding Score", f"{res['retrieval_grounding_score']:.2f}")

        st.subheader("Drafted Grounded Reply")
        if res["escalate"]:
            st.warning(f"⚠️ **Escalation Reason:** {res['escalation_reason']}")
        else:
            st.success("✅ **Auto-Handled:** Safe self-serve resolution guidance.")
            
        st.info(f"**@AppleSupport Reply Draft:**\n\n{res['reply_draft']}")

        with st.expander("🔍 View Retrieved Historical Resolution Context (RAG)"):
            if res["retrieved_context"]:
                for i, ctx in enumerate(res["retrieved_context"], 1):
                    st.markdown(f"**Snippet {i} (Similarity: {ctx['similarity_score']:.2f})**")
                    st.markdown(f"- *Customer:* {ctx['customer_text']}")
                    st.markdown(f"- *Resolution:* {ctx['brand_response']}")
                    st.divider()
            else:
                st.write("No historical resolution context retrieved.")

# TAB 2: Side-by-Side Model Comparison
with tabs[1]:
    st.subheader("Compare Architectures Side-by-Side")
    compare_query = st.text_input("Enter query for comparison:", "I dropped my iPhone 15 Pro Max and the back glass shattered into pieces.")
    
    if st.button("Run Comparison"):
        res_trivial = agents["trivial"].process_message(compare_query)
        res_simple = agents["simple"].process_message(compare_query)
        res_proposed = agents["proposed"].process_message(compare_query)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 1. Trivial Baseline")
            st.caption("Majority Class + Static Template")
            st.json({
                "Intent": res_trivial["intent"],
                "Action": res_trivial["action"],
                "Reply": res_trivial["reply_draft"]
            })
            
        with col2:
            st.markdown("### 2. Simple Baseline")
            st.caption("Zero-Shot Keywords + Generic Prompt")
            st.json({
                "Intent": res_simple["intent"],
                "Action": res_simple["action"],
                "Reply": res_simple["reply_draft"]
            })

        with col3:
            st.markdown("### 3. Proposed Agent")
            st.caption("RAG Grounding + Escalation Router")
            json_output = {
                "Intent": res_proposed["intent"],
                "Action": res_proposed["action"],
                "Grounding Score": f"{res_proposed['retrieval_grounding_score']:.2f}",
                "Reply": res_proposed["reply_draft"]
            }
            if res_proposed["escalate"]:
                json_output["Escalation Reason"] = res_proposed["escalation_reason"]
            st.json(json_output)

# TAB 3: Golden Set Benchmark Results
with tabs[2]:
    st.subheader("Headline Benchmark Results (200 Golden Set Examples)")
    
    benchmark_data = {
        "Metric": [
            "Intent Classification Accuracy",
            "Intent Macro F1",
            "Escalation Precision",
            "Escalation Recall",
            "Escalation F1 Score",
            "Safety False Negative Rate (Lower is Better)",
            "ROUGE-L Score",
            "BLEU Score",
            "Cosine Semantic Similarity",
            "LLM Judge Quality Score (1-5)"
        ],
        "Trivial Baseline": ["27.5%", "0.062", "0.0%", "0.0%", "0.000", "100.0%", "0.080", "0.073", "0.033", "4.13"],
        "Simple Baseline": ["22.5%", "0.272", "100.0%", "1.3%", "0.026", "98.7%", "0.135", "0.062", "0.056", "4.31"],
        "Proposed Agent": ["75.5%", "0.793", "97.1%", "87.0%", "0.918", "13.0%", "0.149", "0.064", "0.114", "4.50"]
    }
    
    df_metrics = pd.DataFrame(benchmark_data)
    st.dataframe(df_metrics, use_container_width=True)

# TAB 4: Methodological Report
with tabs[3]:
    st.markdown("""
    ### Problem Framing & System Architecture
    - **Target Brand**: `@AppleSupport`
    - **What "Good" Means**: High intent accuracy, zero false auto-handles on security/hardware hazards, grounded historical resolutions, and clear brand tone.
    - **Non-Goals**: We explicitly chose NOT to build live Twitter API bots, full database mutations, or multi-modal hardware diagnostics.
    
    ### Key Highlights
    1. **Safety First**: Reduced Safety False Negative Rate from **100%** (Trivial) down to **13.0%** (Proposed Agent).
    2. **Escalation Precision**: Achieved **97.1%** precision in human escalation routing.
    3. **Grounded RAG**: Integrated historical resolution knowledge base for actionable step-by-step guidance.
    """)
