import time
import pandas as pd
import streamlit as st

from src.rag import CEDDRAGPipeline
from src.db import init_db, log_interaction, update_feedback, get_recent_logs, get_analytics_summary

st.set_page_config(
    page_title="Hong Kong CEDD GS - Section 15 RAG",
    page_icon="🏗️",
    layout="wide"
)

# Initialize database
init_db()

# Cache the RAG pipeline
@st.cache_resource(show_spinner="Loading Hybrid Search Index & Embeddings...")
def load_rag_pipeline():
    return CEDDRAGPipeline(k=4)

rag = load_rag_pipeline()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Navigation Tabs
tab_chat, tab_monitor = st.tabs(["💬 Engineering Q&A", "📊 Observability & Monitoring"])

# ---------------------------------------------------------
# TAB 1: Chat & Engineering Assistant
# ---------------------------------------------------------
with tab_chat:
    st.title("🏗️ Hong Kong CEDD Specification Assistant")
    st.caption("Grounded RAG querying **General Specification for Civil Engineering Works (Section 15: Steel Reinforcement)**")

    # Sidebar: sample engineering questions
    st.sidebar.header("Sample Test Queries")
    st.sidebar.caption("Click any question below to test:")
    sample_queries = [
        "What are the temperature requirements when bending steel reinforcement?",
        "What particulars must be submitted for Category A reinforcement?",
        "What are the testing requirements for reinforcement connectors?",
        "How should steel reinforcement be stored on site to prevent damage?"
    ]
    for idx, sq in enumerate(sample_queries):
        if st.sidebar.button(sq, key=f"sample_btn_{idx}"):
            st.session_state.pending_query = sq

    # Render existing chat history
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show latency and feedback buttons if assistant response
            if msg["role"] == "assistant":
                st.caption(f"⚡ Latency: {msg.get('latency', 0.0):.2f}s | Log ID: #{msg.get('log_id', '?')}")
                
                current_fb = msg.get("feedback", 0)
                log_id = msg.get("log_id")
                
                if current_fb == 0 and log_id is not None:
                    col1, col2, _ = st.columns([1, 1, 8])
                    with col1:
                        if st.button("👍", key=f"up_{msg_idx}_{log_id}", help="Mark as accurate"):
                            update_feedback(log_id, feedback=1, notes="User marked accurate")
                            msg["feedback"] = 1
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"down_{msg_idx}_{log_id}", help="Mark as inaccurate"):
                            update_feedback(log_id, feedback=-1, notes="User marked inaccurate")
                            msg["feedback"] = -1
                            st.rerun()
                elif current_fb == 1:
                    st.caption("✅ Marked as **accurate** (👍)")
                elif current_fb == -1:
                    st.caption("⚠️ Marked as **inaccurate** (👎)")

    # Read user input
    chat_input_val = st.chat_input("Ask a technical requirement from CEDD Section 15...")
    active_query = None

    if chat_input_val:
        active_query = chat_input_val
    elif st.session_state.pending_query:
        active_query = st.session_state.pending_query
        st.session_state.pending_query = None

    # Process new query
    if active_query:
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching clauses and generating grounded answer..."):
                start_time = time.time()
                result = rag.answer_query(active_query)
                latency = time.time() - start_time

                answer = result["answer"]
                sources = result["source_documents"]

                # Extract only the clause ID, removing duplicates so the logs look clean
                retrieved_clauses = list(dict.fromkeys([
                    d.metadata.get('clause_id', 'Clause')
                    for d in sources
                ]))

                log_id = log_interaction(
                    question=active_query,
                    answer=answer,
                    retrieved_clauses=retrieved_clauses,
                    latency_seconds=latency,
                    model_name="nvidia/nemotron-3.5-lightning:free"
                )

                st.markdown(answer)
                st.caption(f"⚡ Latency: {latency:.2f}s | Log ID: #{log_id}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "latency": latency,
            "log_id": log_id,
            "feedback": 0
        })
        st.rerun()

# ---------------------------------------------------------
# TAB 2: Observability & Monitoring Dashboard
# ---------------------------------------------------------
with tab_monitor:
    st.title("📊 Production Observability & Monitoring")
    st.caption("Live telemetry persisted in local SQLite database (`rag_monitor.db`)")

    summary = get_analytics_summary()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Invocations", summary["total_queries"])
    m2.metric("Mean Latency", f"{summary['avg_latency']}s")
    m3.metric("Positive Ratings (👍)", summary["positive_feedback"])
    m4.metric("Negative Ratings (👎)", summary["negative_feedback"])

    st.write("---")
    st.subheader("Recent Query Execution Logs")

    recent_logs = get_recent_logs(limit=50)
    if recent_logs:
        df = pd.DataFrame(recent_logs)
        display_cols = ["id", "timestamp", "question", "retrieved_clauses", "latency_seconds", "feedback"]
        df_display = df[[c for c in display_cols if c in df.columns]]
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No query logs recorded yet.")

    if st.button("🔄 Refresh Telemetry"):
        st.rerun()