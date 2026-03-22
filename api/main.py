from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
import shap

app = FastAPI()

intermediate_model = joblib.load("../model/intermediate_model.pkl")
explainer = shap.TreeExplainer(intermediate_model)


@app.get("/")
def home():
    return {"message": "API running"}


class PatientData(BaseModel):
    Age: float
    Sex: float
    BMI: float
    Waist: float
    Glucose: float
    Triglycerides: float



@app.post("/predict")
def predict(data: PatientData):
    TyG = np.log((data.Triglycerides * data.Glucose) / 2)
    feature_names=["Age","Sex","BMI","Waist","Glucose","Triglycerides","TyG Index"]
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
    # SHAP explanation
    shap_values = explainer(features)

    values = shap_values.values[0, :, 1]

    contributions = list(zip(feature_names, values))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    top_factors = [
        f"{name} increased risk (impact {round(float(val*100), 3)})"
        if val > 0 else f"{name} decreased risk (impact {round(float(val*100),3)})"
        for name, val in contributions[:]
    ]
    #print(values)
    #print(values.shape)
    label = "Insulin Resistant" if prediction == 1 else "Normal"

    #print(top_factors)

    return {
        "prediction": int(prediction),
        "label": label,
        "risk_probability": round(prob * 100, 2),
        "top_risk_factors": top_factors
    }
