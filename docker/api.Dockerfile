FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --break-system-packages

COPY src/ ./src/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY data/chroma_db/ ./data/chroma_db/

EXPOSE 8000

CMD ["uvicorn", "src.python.api.main:app", "--host", "0.0.0.0", "--port", "8000"]