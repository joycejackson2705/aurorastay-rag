FROM python:3.13-slim
WORKDIR /app
COPY . .
# CPU-only PyTorch (much smaller than the default GPU build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
# Download the embedding model at build time so the container starts offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 7860
CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:app"]
