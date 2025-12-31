#  Prédiction de Réussite Étudiante par IA

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)](https://fastapi.tiangolo.com/)


**DATAIKÔS** est une application web légère qui utilise un modèle de **régression logistique** entraîné pour prédire si un étudiant réussira son année scolaire en fonction de ses habitudes, performance académique antérieure et facteurs personnels.

Le modèle a été comparé avec plusieurs fonctions de perte (Binary Cross Entropy, Focal Loss, Hinge Loss) et la version **Binary Cross Entropy** a donné les meilleures performances (Accuracy ≈ 89% sur le jeu de test).

Ce projet inclut :
- Une **API FastAPI** avec prédiction en temps réel
- Un **frontend statique** beau et responsive (HTML/CSS/JS pur)
- Tout le modèle **hardcodé** (pas de dépendances ML lourdes comme scikit-learn)

##  Démo locale

Une fois le serveur lancé, ouvrez votre navigateur et allez à :

 **[Tester le modèle de prediction ici](https://student-prediction-interface.onrender.com/static/index.html)**

Vous pourrez remplir le formulaire et obtenir instantanément la prédiction de réussite avec la probabilité associée.

##  Aperçu du modèle

- **Algorithme** : Régression logistique (implémentation from scratch + optimisation par descente de gradient)
- **Fonction de perte** : Binary Cross Entropy (meilleure performance)
- **Features** (14 au total) :
  - Âge, Genre, Niveau d’études, Situation de vie
  - GPA (moyenne), Qualité d’enseignement, Participation aux labs, Plan structuré
  - Heures de sommeil quotidiennes, Activité physique
  - Longueur du texte "Facteurs de succès" et "Suggestions d’amélioration"
  - Heures d’étude hebdomadaires, Régularité en classe

- **Performances sur le jeu de test** :
  - Accuracy : ~89%
  - F1-Score : ~82%

##  Installation et lancement
#### A defaut de tout simplement cliquer sur le lien pour utilisation, faites ceci:

### Prérequis
- Python 3.8 ou supérieur
- `pip` à jour

### Étapes

1. **Clonez le dépôt**

```bash
git clone https://github.com/votre-username/dataikos.git
cd dataikos
```

2. **(Recommandé) Créez un environnement virtuel**

```bash
python -m venv venv
source venv/bin/activate    # Sur Windows : venv\Scripts\activate
```

3. **Installez les dépendances**

```bash
pip install -r requirements.txt
```

Le fichier `requirements.txt` contient seulement :
- fastapi
- uvicorn
- numpy
- pydantic

4. **Lancez le serveur**

```bash
uvicorn app:app --reload
```

5. **Ouvrez l’application**

Allez à l’URL suivante dans votre navigateur :

 **http://127.0.0.1:8000/static/index.html**

## Structure du projet

```
dataikos/
├── Modele/
│   └── classification_comparison.png          # image montrant les correspondance entre les differentes fonctions coûts
│   └── dataset_final.csv         # Dataset recupéré après enquêtes
│   └── generateur.ypnb         # Fnotebook de generation des données
│   └── regression logistique complete.ipynb         # notebook d'apprentissage complet avec toutes les fonctions coûts
│   └── Regression logistique.ipynb          # notebook d'apprentissage 
│   └── scaler.pkl          # fichier de conservation des min et max
│   └── synthetic_2data_1000.csv         # fichier synthetiaue contenant 1000 données
│   └── theta_bce.pkl          # fichier contenant les valeurs de theta
├── app.py                  # Backend FastAPI + modèle hardcodé
├── requirements.txt        # Dépendances minimales
├── static/
│   └── index.html          # Frontend complet (formulaire + affichage résultat)
│   └── favicon.ico          # icone necessqire qu bon fonctionnement 
└── README.md               # Ce fichier
```

##  Personnalisation

- Pour ajuster le seuil de décision (par défaut 0.5) : modifiez la ligne dans `app.py`
  ```python
  prediction = int(prob >= 0.5)  # ← changez 0.5 si besoin
  ```

##  Licence

Ce projet est sous licence **Personnelle**.

---

**DATAIKÔS** – Parce que chaque étudiant mérite de connaître ses chances de réussite 

Fais avec coeur par l'equipe David, Faysal, Prudencia, Randy et Armstrong
