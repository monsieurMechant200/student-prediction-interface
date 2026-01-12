```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DATAIKÔS - Prédiction de Réussite Étudiante")

# CORS (restreindre en prod à votre domaine frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Changez à ["http://votre-frontend.com"] en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle hardcodé (theta + min/max)
THETA = np.array([-6.60598966, -0.38887618, -0.05527884, 0.24046097, 14.34727348, -0.66947624, 0.0156859,
                  -0.17815602, 0.32959119, -0.01474053, 0.08425352, -0.45180327, 0.02356171, -0.2706575 , -0.9303213
])

MIN_VALS = np.array([16., 0., 1., 7., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
MAX_VALS = np.array([30., 1., 5., 17.05, 3., 1., 2., 2., 1., 2., 418., 758., 230., 6.])

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

    @field_validator('Age')
    def validate_age(cls, v):
        if not 16 <= v <= 30:
            raise ValueError('Âge doit être entre 16 et 30')
        return v

    @field_validator('GPA')
    def validate_gpa(cls, v):
        if not 0 <= v <= 20:
            raise ValueError('GPA doit être entre 0 et 20')
        return v

    # Ajoutez d'autres validateurs si nécessaire...
    @field_validator('Level')
    def validate_level(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Niveau doit être entre 1 et 5')
        return v

    @field_validator('Teaching_Quality')
    def validate_teaching_quality(cls, v):
        if not 0 <= v <= 3:
            raise ValueError('Qualité d\'enseignement doit être entre 0 et 3')
        return v

    @field_validator('Lab_Sessions')
    def validate_lab_sessions(cls, v):
        if v not in [0, 1]:
            raise ValueError('Sessions de labo doit être 0 ou 1')
        return v

    @field_validator('Structured_Plan')
    def validate_structured_plan(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Méthode d\'étude doit être entre 0 et 2')
        return v

    @field_validator('Living_Situation')
    def validate_living_situation(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Situation de vie doit être entre 0 et 2')
        return v

    @field_validator('Sleep_Hours_Daily')
    def validate_sleep_hours_daily(cls, v):
        if not 0 <= v <= 24:
            raise ValueError('Heures de sommeil doit être entre 0 et 24')
        return v

    @field_validator('Physical_Activity')
    def validate_physical_activity(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Activité physique doit être entre 0 et 2')
        return v

    @field_validator('Study_Hours_Weekly')
    def validate_study_hours_weekly(cls, v):
        if not 0 <= v <= 168:
            raise ValueError('Heures d\'étude par semaine doit être entre 0 et 168')
        return v

    @field_validator('Class_Regularity')
    def validate_class_regularity(cls, v):
        if not 0 <= v <= 6:
            raise ValueError('Régularité en classe doit être entre 0 et 6')
        return v

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

@app.get("/")
def root():
    return {"message": "API DATAIKÔS en ligne"}

@app.post("/predict")
def predict(data: StudentData):
    try:
        features = np.array([[
            data.Age, data.Gender, data.Level, data.GPA, data.Teaching_Quality,
            data.Lab_Sessions, data.Structured_Plan, data.Living_Situation,
            data.Sleep_Hours_Daily, data.Physical_Activity, data.Success_Factors_Len,
            data.Improvement_Suggestions_Len, data.Study_Hours_Weekly, data.Class_Regularity
        ]])
        # Normalisation
        scaled = (features - MIN_VALS) / (MAX_VALS - MIN_VALS + 1e-8)
        X_b = np.c_[np.ones(scaled.shape[0]), scaled]
        # Prédiction
        prob = sigmoid(np.dot(X_b, THETA))[0]
        prediction = int(prob >= 0.5)
        logger.info(f"Prédiction effectuée : {prediction} avec probabilité {prob}")
        # Recommandations personnalisées (priorisées et limitées à 5)
        recommendations = []
        if data.GPA < 12:
            recommendations.append("Votre GPA est bas : augmentez vos heures d'étude à 25-30h/semaine et revoyez vos méthodes.")
        if data.Sleep_Hours_Daily < 7:
            recommendations.append("Sommeil insuffisant : visez 7-9h par nuit pour booster votre concentration et performance.")
        if data.Study_Hours_Weekly < 20:
            recommendations.append("Étudiez plus : planifiez des sessions structurées en groupe pour atteindre 20h minimum.")
        if data.Physical_Activity < 1:
            recommendations.append("Activité physique faible : intégrez 30min d'exercice 3x/semaine pour réduire le stress.")
        if data.Class_Regularity < 4:
            recommendations.append("Régularité en classe à améliorer : assistez à au moins 90% des cours pour mieux suivre.")
        if data.Success_Factors_Len < 50:
            recommendations.append("Développez vos facteurs de succès : listez plus de stratégies personnelles pour renforcer votre motivation.")
        if data.Improvement_Suggestions_Len < 50:
            recommendations.append("Pensez à plus d'améliorations : identifiez des axes comme le sommeil ou l'organisation.")
        if not recommendations:
            recommendations.append("Profil excellent ! Maintenez vos habitudes pour assurer une réussite continue.")
        # Limiter à 5 max et prioriser (ex. : trier par criticité si besoin)
        recommendations = recommendations[:5]
        return {
            "prediction": prediction,
            "probability": round(float(prob), 4),
            "recommendations": recommendations
        }
    except ValueError as ve:
        logger.warning(f"Validation échouée : {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erreur interne : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction")

# Servir les fichiers statiques (pour le frontend HTML si intégré)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```
