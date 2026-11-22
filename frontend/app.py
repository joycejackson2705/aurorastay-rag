import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860")   # Flask API (same Codespace, host network)

st.set_page_config(page_title="AuroraStay Policy Assistant", page_icon="🏨", layout="wide")
st.title("🏨 AuroraStay Policy Assistant")
st.caption("Internal tool for front-desk and reservations staff. Answers come only from AuroraStay policy documents.")

# Sidebar: retrieval and generation settings
with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve (k)", 1, 8, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.slider("Max tokens", 100, 800, 400, 50)
    mode = st.radio("Response type", ["Answer only", "Answer + Relevant Chunks"], index=1)

query = st.text_input("Ask a policy question", placeholder="e.g. What is the pet fee at ASH-CHI Loop?")

if st.button("Submit", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        endpoint = "/v1/answer" if mode == "Answer only" else "/v1/answer_with_relevant_chunks"
        payload = {"query": query, "k": k, "model_name": "gpt-4o", "temperature": temperature,
                   "top_p": 1.0, "max_tokens": max_tokens}
        try:
            with st.spinner("Searching the policies..."):
                resp = requests.post(BACKEND_URL + endpoint, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            st.subheader("Answer")
            st.markdown(data["answer"])
            for i, ch in enumerate(data.get("relevant_chunks", []), 1):
                with st.expander(f"Source {i}: {ch['document']} | {ch['section']}"):
                    st.markdown(ch["content"])
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the RAG API: {e}")
