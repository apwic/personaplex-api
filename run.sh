#!/bin/bash
# Run PersonaPlex API locally

cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

echo "Starting PersonaPlex API..."
echo "API will be available at http://localhost:8000"
echo ""
echo "Endpoints:"
echo "  GET  /health    - Health check"
echo "  GET  /models    - Model status"
echo "  POST /inference - Speech-to-speech (requires Bearer token)"
echo ""

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
