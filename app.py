#!/usr/bin/env python
# coding: utf-8

# In[4]:


# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="DATAIKÔS - Prédiction Réussite Étudiante")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle hardcodé (theta + min/max)
THETA = np.array([-6.60598966, -0.38887618, -0.05527884,  0.24046097, 14.34727348, -0.66947624,  0.0156859,
                  -0.17815602,  0.32959119, -0.01474053, 0.08425352, -0.45180327,  0.02356171, -0.2706575 , -0.9303213
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

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

@app.get("/")
def root():
    return {"message": "API DATAIKÔS en ligne"}

@app.post("/predict")
def predict(data: StudentData):
    features = np.array([[
        data.Age, data.Gender, data.Level, data.GPA, data.Teaching_Quality,
        data.Lab_Sessions, data.Structured_Plan, data.Living_Situation,
        data.Sleep_Hours_Daily, data.Physical_Activity,
        data.Success_Factors_Len, data.Improvement_Suggestions_Len,
        data.Study_Hours_Weekly, data.Class_Regularity
    ]])

    scaled = (features - MIN_VALS) / (MAX_VALS - MIN_VALS + 1e-8)
    X_b = np.c_[np.ones(scaled.shape[0]), scaled]
    prob = sigmoid(np.dot(X_b, THETA))[0]

    return {
        "prediction": int(prob >= 0.5),
        "probability": round(float(prob), 4)
    }

# Servir le frontend
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# In[ ]:
