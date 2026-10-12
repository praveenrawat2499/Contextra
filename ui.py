import importlib
import os
import streamlit as st

import src.cache
import src.search

importlib.reload(src.cache)
importlib.reload(src.search)

from src.chat_history import (
    clear_all_sessions,
    create_new_session_id,
    delete_session,
    load_all_sessions,
    load_session,
    save_session,
)
from src.data_loader import load_all_documents
from src.diagnostics import run_system_diagnostics
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

# Set page config
st.set_page_config(
    page_title="Contextra Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Main title styling */
.main-title {
    background: linear-gradient(135deg, #FF4B4B 0%, #852DF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #8A8D9A;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.status-healthy { background: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid #2ecc71; }
.status-degraded { background: rgba(241, 196, 15, 0.15); color: #f1c40f; border: 1px solid #f1c40f; }
.status-unready { background: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 1px solid #e74c3c; }

/* Custom button hover transitions */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #852DF6 0%, #FF4B4B 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    transition: all 0.3s ease;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(133, 45, 246, 0.4);
}
</style>
""",
    unsafe_allow_html=True,
)


# Cached search engine factory
@st.cache_resource
def get_rag_search(
    llm_model: str,
    enable_expansion: bool,
    enable_reranking: bool,
    enable_semantic_cache: bool = True,
):
    return RAGSearch(
        llm_model=llm_model,
        enable_expansion=enable_expansion,
        enable_reranking=enable_reranking,
        enable_semantic_cache=enable_semantic_cache,
    )


# App layout header
st.markdown('<h1 class="main-title">🤖 Contextra Chatbot</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Production-grade Retrieval-Augmented Generation with Hybrid Search, Citations & Guardrails</p>',
    unsafe_allow_html=True,
)

# Initialize Session State for Chat Management
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = load_all_sessions()

if "current_session_id" not in st.session_state:
    if st.session_state.all_sessions:
        st.session_state.current_session_id = next(iter(st.session_state.all_sessions))
        st.session_state.messages = st.session_state.all_sessions[st.session_state.current_session_id].get("messages", [])
    else:
        new_sid = create_new_session_id()
        st.session_state.current_session_id = new_sid
        st.session_state.messages = []
elif "messages" not in st.session_state:
    curr_data = load_session(st.session_state.current_session_id)
    st.session_state.messages = curr_data.get("messages", []) if curr_data else []

# Sidebar configurations, conversations, and diagnostics
with st.sidebar:
    st.markdown("## 💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True):
        new_sid = create_new_session_id()
        st.session_state.current_session_id = new_sid
        st.session_state.messages = []
        st.rerun()

    # Load and display saved sessions
    saved_sessions = load_all_sessions()
    st.session_state.all_sessions = saved_sessions

    if saved_sessions:
        st.caption("Saved History (from Memory & Disk):")
        for sid, sdata in list(saved_sessions.items())[:10]:
            is_active = (sid == st.session_state.current_session_id)
            title = sdata.get("title", "New Conversation")
            display_title = (title[:22] + "...") if len(title) > 22 else title
            btn_label = f"👉 {display_title}" if is_active else f"💬 {display_title}"

            c_btn, c_del = st.columns([0.8, 0.2])
            if c_btn.button(btn_label, key=f"session_btn_{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.session_state.messages = sdata.get("messages", [])
                st.rerun()

            if c_del.button("🗑️", key=f"session_del_{sid}", help="Delete this conversation"):
                delete_session(sid)
                if st.session_state.current_session_id == sid:
                    new_sid = create_new_session_id()
                    st.session_state.current_session_id = new_sid
                    st.session_state.messages = []
                st.session_state.all_sessions = load_all_sessions()
                st.rerun()

        if st.button("🧹 Clear All Chats", use_container_width=True, help="Wipe all saved chat history"):
            clear_all_sessions()
            new_sid = create_new_session_id()
            st.session_state.current_session_id = new_sid
            st.session_state.messages = []
            st.session_state.all_sessions = {}
            st.rerun()

    st.markdown("---")
    st.markdown("## 🔍 System Readiness")
    diag = run_system_diagnostics()
    status_label = diag["overall_status"].upper()
    badge_class = f"status-{diag['overall_status']}"
    st.markdown(
        f'<span class="status-badge {badge_class}">Status: {status_label}</span>',
        unsafe_allow_html=True,
    )

    vec_count = diag["components"]["vector_store"].get("vector_count", 0)
    model_tag = diag["config_summary"]["embedding_model"].split("/")[-1]
    st.caption(f"📊 **Indexed Chunks:** `{vec_count}` | 🧠 **Embeddings:** `{model_tag}`")

    st.markdown("---")
    st.markdown("## ⚙️ Configuration")

    llm_model = st.selectbox(
        "Large Language Model",
        options=["z-ai/glm-5.3-flash", "nvidia/llama-3.1-nemotron-70b-instruct"],
        index=0,
        help="Select which LLM to query on NVIDIA NIM",
    )

    top_k = st.slider("Retrieve Count (K)", min_value=1, max_value=10, value=5)

    enable_expansion = st.checkbox(
        "Query Expansion",
        value=False,
        help="Generate alternative search query variations to boost recall",
    )
    enable_reranking = st.checkbox(
        "Cross-Encoder Reranker",
        value=True,
        help="Re-score candidates using TinyBERT cross-encoder for higher precision",
    )
    enable_semantic_cache = st.checkbox(
        "Semantic Vector Cache",
        value=True,
        help="Tier-2 semantic caching: answers paraphrased and semantically equivalent questions via vector similarity without invoking LLM",
    )

    st.markdown("---")
    st.markdown("## 📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload new documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_ext = file_name.split(".")[-1].lower()

            save_dir = "data/pdf" if file_ext == "pdf" else "data/text_files"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file_name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success(f"Successfully uploaded {len(uploaded_files)} file(s)!")

    # Rebuild Index Button
    if st.button("🔄 Rebuild Vector Store"):
        with st.spinner("Parsing documents and rebuilding FAISS database..."):
            docs = load_all_documents("data")
            if not docs:
                st.error("No documents found in the data directory!")
            else:
                store = FaissVectorStore("faiss_store")
                store.build_from_documents(docs)
                st.cache_resource.clear()
                st.success("FAISS index successfully rebuilt and loaded!")
                st.rerun()

    # Query Cache Metrics
    rag_cached = get_rag_search(
        llm_model=llm_model,
        enable_expansion=enable_expansion,
        enable_reranking=enable_reranking,
        enable_semantic_cache=enable_semantic_cache,
    )
    if not hasattr(rag_cached, "get_cache_stats"):
        st.cache_resource.clear()
        st.rerun()

    cache_stats = rag_cached.get_cache_stats()
    with st.expander("⚡ Query Cache", expanded=False):
        c_hit, c_miss = st.columns(2)
        c_hit.metric("Cache Hits", cache_stats.get("hits", 0))
        c_miss.metric("Cache Misses", cache_stats.get("misses", 0))
        exact_h = cache_stats.get("exact_hits", 0)
        sem_h = cache_stats.get("semantic_hits", 0)
        st.caption(
            f"🎯 **Hit Rate:** `{cache_stats.get('hit_rate_pct', 0.0)}%` "
            f"*(⚡ Exact: `{exact_h}` | 🧠 Semantic: `{sem_h}`)*\n\n"
            f"🚫 **LLM Calls Saved:** `{cache_stats.get('llm_calls_avoided', 0)}`\n\n"
            f"🏷️ **Index Version:** `{cache_stats.get('index_version', 'N/A')}`"
        )
        if st.button("🧹 Clear Query Cache", use_container_width=True):
            if hasattr(rag_cached, "clear_cache"):
                rag_cached.clear_cache()
            st.success("Query cache cleared!")
            st.rerun()

# Display Chat History with Citations
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        citations = message.get("citations")
        if citations:
            with st.expander(f"📚 Verified Citations ({len(citations)})"):
                for c in citations:
                    st.markdown(
                        f"**[{c['citation_index']}]** `{c['source']}` *(Chunk: `{c['chunk_id'][:10]}`)*\n"
                        f"> \"{c['snippet']}\""
                    )

# User Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    curr_sid = st.session_state.current_session_id
    curr_data = load_session(curr_sid)
    session_title = curr_data.get("title") if curr_data else prompt[:35]
    if not session_title or session_title == "New Conversation":
        session_title = prompt[:35]

    save_session(
        session_id=curr_sid,
        title=session_title,
        messages=st.session_state.messages,
    )
    st.session_state.all_sessions = load_all_sessions()

    # Query RAG Engine and display response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating grounded answer..."):
            try:
                rag_engine = get_rag_search(
                    llm_model=llm_model,
                    enable_expansion=enable_expansion,
                    enable_reranking=enable_reranking,
                    enable_semantic_cache=enable_semantic_cache,
                )

                rag_response = rag_engine.generate_grounded_answer(
                    query=prompt,
                    top_k=top_k,
                    chat_history=st.session_state.messages[:-1],
                )

                st.write(rag_response.answer)
                is_cached = bool(rag_response.metadata.get("cache_hit", False))
                cache_tier = rag_response.metadata.get("cache_tier", "exact")
                latency_val = rag_response.metadata.get("latency_ms", 0.0)
                sim_score = rag_response.metadata.get("similarity")
                if is_cached:
                    if cache_tier == "semantic" and sim_score:
                        st.caption(f"⚡ *Served from semantic vector cache ({latency_val:.1f} ms — similarity {sim_score*100:.1f}%)*")
                    else:
                        st.caption(f"⚡ *Served from exact cache ({latency_val:.1f} ms — LLM avoided)*")
                elif latency_val > 0:
                    st.caption(f"🧠 *Generated via LLM ({latency_val:.0f} ms)*")

                serial_citations = []
                if rag_response.citations:
                    with st.expander(f"📚 Verified Citations ({len(rag_response.citations)})"):
                        for c in rag_response.citations:
                            st.markdown(
                                f"**[{c.citation_index}]** `{c.source}` *(Chunk: `{c.chunk_id[:10]}`)*\n"
                                f"> \"{c.snippet}\""
                            )
                            serial_citations.append(
                                {
                                    "citation_index": c.citation_index,
                                    "source": c.source,
                                    "chunk_id": c.chunk_id,
                                    "snippet": c.snippet,
                                }
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": rag_response.answer,
                        "status": rag_response.status,
                        "citations": serial_citations,
                        "cache_hit": is_cached,
                    }
                )

                save_session(
                    session_id=curr_sid,
                    title=session_title,
                    messages=st.session_state.messages,
                )
                st.session_state.all_sessions = load_all_sessions()

            except Exception as e:
                st.error(f"Error querying RAG engine: {e}")
