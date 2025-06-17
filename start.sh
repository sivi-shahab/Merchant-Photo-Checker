#!/usr/bin/env bash
set -e

# 1. Start Ollama server in background
ollama serve --host "$OLLAMA_HOST" --port "$OLLAMA_PORT" &

echo "Waiting for Ollama to be ready..."
# 2. Wait until Ollama API responds
for i in {1..60}; do
  if curl -sSf "http://localhost:$OLLAMA_PORT/api/tags" >/dev/null; then
    echo "Ollama is up"
    break
  fi
  sleep 1
done

# 3. Pull the model (caches locally)
echo "Pulling Ollama model $OLLAMA_MODEL..."
ollama pull "$OLLAMA_MODEL"

# 4. Launch the FastAPI backend
echo "Starting FastAPI..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000