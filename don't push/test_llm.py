from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
import shap
import requests

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# Load ML model
intermediate_model = joblib.load("../model/intermediate_model.pkl")
explainer = shap.TreeExplainer(intermediate_model)

# Load embeddings + FAISS ONCE (very important)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "./",
    embedding,
    allow_dangerous_deserialization=True
)


# Input schema
class PatientData(BaseModel):
    Age: float
    Sex: float
    BMI: float
    Waist: float
    Glucose: float
    Triglycerides: float


@app.get("/")
def home():
    return {"message": "API running"}


@app.post("/predict")
def predict(data: PatientData):
    # ---------------- ML PART ----------------
    TyG = np.log((data.Triglycerides * data.Glucose) / 2)

    feature_names = ["Age", "Sex", "BMI", "Waist", "Glucose", "Triglycerides", "TyG Index"]

    features = np.array([[
        data.Age,
        data.Sex,
        data.BMI,
        data.Waist,
        data.Glucose,
        data.Triglycerides,
        TyG
    ]])

    prediction = intermediate_model.predict(features)[0]
    prob = intermediate_model.predict_proba(features)[0][1]

    shap_values = explainer(features)
    values = shap_values.values[0, :, 1]

    contributions = list(zip(feature_names, values))
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    top_factors = [
        f"{name} increased risk (impact {round(float(val * 100), 3)})"
        if val > 0 else f"{name} decreased risk (impact {round(float(val * 100), 3)})"
        for name, val in contributions[:5]  # take top 5 only
    ]

    label = "Insulin Resistant" if prediction == 1 else "Normal"
    risk_category = "Low Risk" if prob< 0.31 else "Moderate Risk" if prob < 0.61 else "High Risk"

    result = {
        "prediction": int(prediction),
        "label": label,
        "risk_category": risk_category,
        "risk_probability": round(prob * 100, 2),
        "top_risk_factors": top_factors
    }

    # ---------------- RAG QUERY (DYNAMIC) ----------------

    # Extract keywords from top factors
    keywords = " ".join([f.split()[0] for f in top_factors])

    query = f"{label} causes effects {keywords}"

    docs = db.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in docs])

    # ---------------- PROMPT ----------------

    prompt = f"""You are a medical assistant generating a short explanation for a health report.



Input:
- Condition: {label}
- Risk Probability: {round(prob * 100, 2)}%
- Key Factors: {contributions[:3][0]}

Output:
A short, clean explanation in 6 sentences.
"""

    # ---------------- OLLAMA ----------------

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }
    )

    llm_output = response.json()["response"]

    # ---------------- FINAL RESPONSE ----------------

    return {
        "prediction": result["prediction"],
        "label": result["label"],
        "risk_category": result["risk_category"],
        "risk_probability": result["risk_probability"],
        "top_risk_factors": result["top_risk_factors"],
        "explanation": llm_output
    }