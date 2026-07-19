# WD14 tagger sidecar — CPU-only ONNX inference.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Model downloads to /cache on first start (mount a volume to persist it).
ENV HF_HOME=/cache
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
