#!/bin/bash
cd "$(dirname "$0")"
export NUMBA_CACHE_DIR="$(pwd)/.numba_cache"
echo "Starting web app at http://localhost:8765"
python3 -m uvicorn server:app --host 127.0.0.1 --port 8765 --reload
