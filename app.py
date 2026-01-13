from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic.functional_validators import field_validator
import numpy as np
import logging

# ----------------------------------
# CONFIG
# ----------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DATAIKÔS - Prédiction de Réussite Étudiante")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MODEL

THETA = np.array([
    -6.60598966, -0.38887618, -0.05527884, 0.24046097,
    14.34727348, -0.66947624, 0.0156859, -0.17815602,
    0.32959119, -0.01474053, 0.08425352, -0.45180327,
    0.02356171, -0.2706575, -0.9303213
])

MIN_VALS = np.array([
    16., 0., 1., 7., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
])

MAX_VALS = np.array([
    30., 1., 5., 17.05, 3., 1., 2., 2., 1., 2., 418., 758., 230., 6.
])

# SCHEMA
class StudentData(BaseModel):
    Age: float
    Gender: int
    Level: int
    GPA: float
    Teaching_Quality: float
    Lab_Sessions: int
    Structured_Plan: int
    Living_Situation: int
    Sleep_Hours_Daily: float
    Physical_Activity: int
    Success_Factors_Len: int
    Improvement_Suggestions_Len: int
    Study_Hours_Weekly: int
    Class_Regularity: float

    @field_validator("Age")
    def age_ok(cls, v):
        if not 16 <= v <= 30:
            raise ValueError("Âge invalide")
        return v

    @field_validator("GPA")
    def gpa_ok(cls, v):
        if not 0 <= v <= 20:
            raise ValueError("GPA invalide")
        return v

# UTILS
def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

# ROUTES API
@app.post("/api/predict")
def predict(data: StudentData):
    try:
        X = np.array([[
            data.Age, data.Gender, data.Level, data.GPA,
            data.Teaching_Quality, data.Lab_Sessions,
            data.Structured_Plan, data.Living_Situation,
            data.Sleep_Hours_Daily, data.Physical_Activity,
            data.Success_Factors_Len, data.Improvement_Suggestions_Len,
            data.Study_Hours_Weekly, data.Class_Regularity
        ]])

        # Normalisation
        X_scaled = (X - MIN_VALS) / (MAX_VALS - MIN_VALS + 1e-8)
        X_b = np.c_[np.ones(1), X_scaled]

        prob = sigmoid(X_b @ THETA)[0]
        prediction = int(prob >= 0.5)

        
        # RECOMMANDATIONS IA
        
        recommendations = []

        if data.GPA < 12:
            recommendations.append(
                " Votre GPA est faible : augmentez vos heures d’étude et adoptez une méthode plus structurée."
            )

        if data.Sleep_Hours_Daily < 7:
            recommendations.append(
                "Sommeil insuffisant : visez 7 à 9 heures par nuit pour améliorer la concentration."
            )

        if data.Study_Hours_Weekly < 20:
            recommendations.append(
                "Heures d’étude faibles : planifiez au moins 20 heures d’étude par semaine."
            )

        if data.Physical_Activity < 1:
            recommendations.append(
                "Activité physique insuffisante : 30 minutes d’exercice, 3 fois par semaine, réduisent le stress."
            )

        if data.Class_Regularity < 4:
            recommendations.append(
                "Régularité en classe à améliorer : assistez régulièrement aux cours pour de meilleurs résultats."
            )

        if data.Success_Factors_Len < 50:
            recommendations.append(
                "Développez davantage vos facteurs de succès pour renforcer votre motivation."
            )

        if data.Improvement_Suggestions_Len < 50:
            recommendations.append(
                "Ajoutez plus de suggestions d’amélioration pour mieux progresser."
            )

        if not recommendations:
            recommendations.append(
                "Excellent profil ! Continuez avec ces bonnes habitudes."
            )

        return {
            "prediction": prediction,
            "probability": round(float(prob), 4),
            "recommendations": recommendations[:5]
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Erreur interne")


# FRONTEND
app.mount("/", StaticFiles(directory="static", html=True), name="static")
