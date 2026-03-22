# Insulin Resistance Prediction API

FastAPI-based ML + RAG + LLM system.

## Features
- ML prediction (3 models)
- SHAP explainability
- FAISS RAG
- Groq LLM explanation

## Run locally
```bash
pip install -r requirements.txt
uvicorn api.huggingface:app --reload
