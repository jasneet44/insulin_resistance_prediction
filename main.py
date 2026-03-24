from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
import shap
from groq import Groq
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# ✅ FIXED BASE DIR (points to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- GLOBAL PLACEHOLDERS ----------------
models = {}
explainers = {}
db = None


# ---------------- LOAD EVERYTHING ON STARTUP ----------------
@app.on_event("startup")
def load_everything():
    global models, explainers, db

    print("🚀 Loading models...")

    # ---------------- LOAD MODELS ----------------
    models = {
        "basic": joblib.load(os.path.join(BASE_DIR, "models", "basic_model.pkl")),
        "intermediate": joblib.load(os.path.join(BASE_DIR, "models", "intermediate_model.pkl")),
        "advanced": joblib.load(os.path.join(BASE_DIR, "models", "advanced_model.pkl"))
    }

    print("✅ Models loaded")

    # ---------------- SHAP ----------------
    explainers = {
        name: shap.TreeExplainer(model)
        for name, model in models.items()
    }

    print("✅ SHAP explainers ready")

    # ---------------- LOAD RAG ----------------
    try:
        print("📚 Loading FAISS...")

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.load_local(
            os.path.join(BASE_DIR, "rag"),
            embedding,
            allow_dangerous_deserialization=True
        )

        print("✅ FAISS loaded")

    except Exception as e:
        print("❌ RAG failed to load:", str(e))
        db = None


# ---------------- INPUT SCHEMA ----------------
class PatientData(BaseModel):
    model_type: str

    Age: float
    Sex: float
    BMI: float
    Waist: float

    Glucose: float | None = None
    Triglycerides: float | None = None
    HDL: float | None = None
    Exercise: float | None = None


@app.get("/")
def home():
    return {"message": "API running"}


# ---------------- FEATURE BUILDER ----------------
def build_features(data: PatientData):
    model_type = data.model_type

    if model_type == "basic":
        feature_names = ["Age", "Sex", "BMI", "Waist"]
        features = np.array([[data.Age, data.Sex, data.BMI, data.Waist]])

    elif model_type == "intermediate":
        TyG = np.log((data.Triglycerides * data.Glucose) / 2)
        feature_names = ["Age", "Sex", "BMI", "Waist", "Glucose", "Triglycerides", "TyG Index"]
        features = np.array([[data.Age, data.Sex, data.BMI, data.Waist,
                              data.Glucose, data.Triglycerides, TyG]])

    elif model_type == "advanced":
        TyG = np.log((data.Triglycerides * data.Glucose) / 2)
        TG_HDL_ratio = data.Triglycerides / data.HDL
        feature_names = ["Age", "Sex", "BMI", "Waist", "Glucose", "Triglycerides",
                         "TyG Index", "HDL", "Exercise", "TG_HDL_ratio"]
        features = np.array([[data.Age, data.Sex, data.BMI, data.Waist,
                              data.Glucose, data.Triglycerides, TyG,
                              data.HDL, data.Exercise, TG_HDL_ratio]])
    else:
        raise ValueError("Invalid model_type")

    return features, feature_names


# ---------------- MAIN API ----------------
@app.post("/predict")
def predict(data: PatientData):

    if data.model_type not in models:
        return {"error": "Invalid model_type. Choose basic/intermediate/advanced"}

    model = models[data.model_type]
    explainer = explainers[data.model_type]

    features, feature_names = build_features(data)

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    shap_values = explainer(features)

    values = shap_values.values[0, :, 1] if len(shap_values.values.shape) == 3 else shap_values.values[0]

    contributions = list(zip(feature_names, values))
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    top_factors = [
        f"{name} increased risk (impact {round(float(val * 100), 3)})"
        if val > 0 else f"{name} decreased risk (impact {round(float(val * 100), 3)})"
        for name, val in contributions[:5]
    ]

    label = "Insulin Resistant" if prediction == 1 else "Normal"
    risk_category = (
        "Low Risk" if prob < 0.31 else
        "Moderate Risk" if prob < 0.61 else
        "High Risk"
    )

    result = {
        "prediction": int(prediction),
        "label": label,
        "risk_category": risk_category,
        "risk_probability": round(prob * 100, 2),
        "top_risk_factors": top_factors
    }

    # ---------------- RAG ----------------
    context = ""
    if db is not None:
        keywords = " ".join([f.split()[0] for f in top_factors])
        query = f"{label} causes effects {keywords}"

        docs = db.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in docs])

    # ---------------- PROMPT ----------------
    prompt = f"""
Patient Condition: {label}
Risk Level: {risk_category} ({round(prob * 100, 2)}%)

Top Risk Factors:
{", ".join([f[0] for f in contributions[:3]])}

Medical Context:
{context}

Explain this to a non-medical person in 3-4 simple sentences.
Mention what was the major factor contributing to the outcome.
Include causes, meaning, and what the patient should understand.
Suggest Recommendations and Lifestyle changes.
"""

    # ---------------- LLM ----------------
    chat_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a medical assistant explaining insulin resistance in simple terms."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    llm_output = chat_response.choices[0].message.content

    return {
        **result,
        "explanation": llm_output
    }