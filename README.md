# AI Business Intelligence App 

Une plateforme intelligente de **Business Intelligence augmentée par l’IA**, combinant :

- Machine Learning
- IA Générative (Gemini API)
- MLOps avec MLflow & DagsHub
- Interface Web Streamlit
- Déploiement Cloud
- Système Champion Model

---

#  Contexte du Projet

Ce projet a été réalisé dans le cadre d’un examen pratique MLOps & IA Générative.

L’objectif est de développer une application complète capable :

- d’analyser des données clients,
- d’effectuer des prédictions intelligentes,
- de suivre plusieurs expérimentations ML,
- de charger automatiquement le meilleur modèle (“Champion”),
- d’intégrer une IA générative,
- puis de déployer une application web professionnelle.

---

#  Fonctionnalités

##  Machine Learning

- Nettoyage des données
- Gestion des valeurs manquantes
- Encodage des variables
- Split Train/Test
- Entraînement de plusieurs modèles :
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost

---

##  MLOps

- Tracking des expériences avec MLflow
- Stockage distant avec DagsHub
- Suivi :
  - paramètres
  - métriques
  - modèles
- Sélection automatique du meilleur modèle (Champion Model)

---

##  Application Streamlit

L’application contient :

-  Présentation du projet
-  Prédiction unitaire
-  Prédiction batch CSV
-  Dashboard & visualisation
-  Analyse IA Générative avec Gemini

---

##  IA Générative

Intégration de Gemini API pour :

- analyser automatiquement les datasets,
- générer des insights,
- produire des explications intelligentes.

---

##  Déploiement

Projet déployable sur :

- Streamlit Community Cloud

---

#  Technologies Utilisées

- Python
- Streamlit
- Scikit-learn
- MLflow
- DagsHub
- Pandas
- NumPy
- Gemini API
- Git & GitHub

---

#  Structure du Projet

AI_Business_Intelligence_App/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── customer_churn.csv
│
├── models/
│   ├── champion_model.pkl
│   └── features.json
│
├── src/
│   ├── train.py
│   ├── champion.py
│   ├── logger.py
│   └── gemini_analysis.py
│
├── logs/
├── mlruns/
├── notebooks/
├── requirements.txt
├── .env
├── .gitignore
└── README.md
---

#  Installation

##  Cloner le projet

```bash
git clone https://github.com/VOTRE_USERNAME/AI_Business_Intelligence_App.git
```

```bash
cd AI_Business_Intelligence_App
```

---

##  Créer un environnement virtuel

### Windows


python -m venv venv


venv\Scripts\activate


---

##  Installer les dépendances

pip install -r requirements.txt

---

#  Variables d’Environnement

Créer un fichier `.env` :

```env
GEMINI_API_KEY=your_api_key
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token
```

---

#  Lancer l’Application

streamlit run app/app.py

---

#  MLflow & DagsHub

## Tracking MLflow

Les expériences sont automatiquement enregistrées avec :

- accuracy
- paramètres
- modèles
- runs

---

## DagsHub

Les runs MLflow sont synchronisés avec DagsHub.

### Exemple :

- Expériences MLflow
- Champion Model
- Historique des runs

---

#  Champion Model System

Le projet implémente un système “Champion Model”.

Le meilleur modèle est automatiquement sélectionné selon :

- Accuracy maximale

Puis chargé dynamiquement dans l’application Streamlit.

---

#  Batch Prediction

L’utilisateur peut :

- uploader un CSV,
- obtenir des prédictions automatiques,
- télécharger les résultats.

---

#  IA Générative

Gemini API permet :

- l’analyse automatique des données,
- la génération d’insights métiers,
- l’explication des tendances du dataset.

---

#  Déploiement

## Streamlit Cloud

Déploiement possible via :

https://streamlit.io/cloud


---

# Liens

## GitHub

https://github.com/mootezkhadhraoui/AI-Business-Intelligence-SaaS

## DagsHub

https://dagshub.com/mootez89/ai-business-intelligence-app

## Application Déployée

https://ai-business-intelligence-saas.streamlit.app/

---

#  Résultats

- Application fonctionnelle
- Tracking MLOps
- IA Générative
- Champion Model
- Déploiement Cloud
- Interface professionnelle