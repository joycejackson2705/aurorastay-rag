from flask import Flask, request, jsonify
from rag_utils import retrieve, rag

app = Flask(__name__)


def chunk_to_dict(doc):
    return {"document": doc.metadata.get("title"), "section": doc.metadata.get("section"),
            "source_file": doc.metadata.get("source"), "content": doc.page_content}


def read_params(data):
    return dict(k=int(data.get("k", 4)), model_name=data.get("model_name", "gpt-4o"),
                temperature=float(data.get("temperature", 0)), top_p=float(data.get("top_p", 1.0)),
                max_tokens=int(data.get("max_tokens", 400)))


@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Aurora Stay RAG API!"


@app.route("/v1/relevant_chunks", methods=["POST"])
def relevant_chunks():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    chunks = retrieve(query, k=int(data.get("k", 4)))
    return jsonify({"query": query, "relevant_chunks": [chunk_to_dict(d) for d in chunks]})


@app.route("/v1/answer", methods=["POST"])
def answer():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    ans, _ = rag(query, **read_params(data))
    return jsonify({"query": query, "answer": ans})


@app.route("/v1/answer_with_relevant_chunks", methods=["POST"])
def answer_with_relevant_chunks():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    ans, chunks = rag(query, **read_params(data))
    return jsonify({"query": query, "answer": ans, "relevant_chunks": [chunk_to_dict(d) for d in chunks]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
