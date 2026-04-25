#!/bin/bash

# --- CONFIGURATION ---
PORT=8000
HOST="0.0.0.0"
CONDA_ENV="genai"

# Ensure we are in the script's directory
cd "$(dirname "$0")"

echo "🚀 Starting Apple Multi-Agent RAG System..."

# Check for API Key
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  WARNING: GROQ_API_KEY is not set in your environment."
    echo "Please run: export GROQ_API_KEY='your_key_here' before starting."
    # We don't exit here, as the app has its own internal checks
fi

# Set Python Path to include current directory for absolute imports
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

echo "📦 Using Conda Environment: $CONDA_ENV"
echo "🌐 URL: http://localhost:$PORT"
echo "------------------------------------------------"

# Run Uvicorn via Conda with explicit environment variable passing
conda run -n $CONDA_ENV --no-capture-output \
    env GROQ_API_KEY="$GROQ_API_KEY" \
    uvicorn app:app \
    --host $HOST \
    --port $PORT \
    --reload \
    --log-level info
