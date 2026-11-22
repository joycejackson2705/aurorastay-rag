import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL = "intfloat/e5-base-v2"

# Load the saved FAISS index once, when the API starts
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
vectorstore = FAISS.load_local(os.path.join(BASE_DIR, "faiss_index"), embedding_model,
                               allow_dangerous_deserialization=True)   # safe: we created this index ourselves

with open(os.path.join(BASE_DIR, "system_prompt.txt")) as f:
    SYSTEM_PROMPT = f.read()


def retrieve(query, k=4, vectorstore=vectorstore):
    """Return the k policy chunks most similar to the query."""
    return vectorstore.similarity_search(query, k=k)


def format_context(retrieved_chunks):
    """Number the chunks so the answer can cite them as [Source n]."""
    return "\n\n".join(f"[Source {i}] {d.page_content}" for i, d in enumerate(retrieved_chunks, 1))


def generate(query, retrieved_chunks, model_name="gpt-4o", temperature=0, top_p=1.0, max_tokens=400):
    """Answer the query using only the retrieved chunks."""
    llm = ChatOpenAI(model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                     api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
                     max_retries=3, timeout=60)
    messages = [SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"### Context\n{format_context(retrieved_chunks)}\n\n### Question\n{query}")]
    return llm.invoke(messages).content


def rag(query, k=4, model_name="gpt-4o", temperature=0, top_p=1.0, max_tokens=400, vectorstore=vectorstore):
    """Full RAG pipeline: retrieve, then generate."""
    retrieved_chunks = retrieve(query, k, vectorstore)
    answer = generate(query, retrieved_chunks, model_name, temperature, top_p, max_tokens)
    return answer, retrieved_chunks
