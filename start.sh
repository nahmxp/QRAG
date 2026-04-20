#!/usr/bin/env bash
# Run QRAG — activates venv and launches Streamlit
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run app.py --server.port 8501 --server.headless false
