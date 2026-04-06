"""
🏎️ LIGNE B : HMM META-BOT (Météo Avancée)
Détecte le régime de marché via les probabilités cachées.
Version Serveur : Sauvegarde en JSON, aucun graphique.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

LOG_FILE = "regime_log.json"

print("📥 Téléchargement des données SPY pour le HMM...")
data = yf.download("SPY", period="5y", progress=False)['Close'].ffill().dropna()

# Calcul des rendements (Returns) et de la Volatilité
returns = np.log(data / data.shift(1)).dropna()
volatility = returns.rolling(window=20).std().dropna()

# Aligner les données
returns = returns.loc[volatility.index]
X = np.column_stack([returns, volatility])

print("🧠 Entraînement du modèle HMM (Recherche des 2 régimes cachés)...")
# On force le modèle à trouver 2 régimes (Calme/Haussier vs Volatil/Baissier)
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
hmm_model.fit(X)

# Prédiction sur le dernier jour
hidden_states = hmm_model.predict(X)
current_state = hidden_states[-1]
state_probs = hmm_model.predict_proba(X)[-1]

# Interprétation du régime (le régime avec la plus haute volatilité moyenne est le Bear Market)
volatility_by_state = [np.mean(X[hidden_states == i, 1]) for i in range(2)]
bear_state = np.argmax(volatility_by_state)

if current_state == bear_state:
    regime_name = "BEAR MARKET (Danger)"
    confidence = state_probs[bear_state]
else:
    regime_name = "BULL/NORMAL MARKET (Sûr)"
    confidence = state_probs[current_state]

# --- SAUVEGARDE JSON ---
log_entry = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "HMM",
    "regime": regime_name,
    "confidence": round(float(confidence), 4)
}

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        try:
            logs = json.load(f)
        except:
            logs = []
else:
    logs = []

logs.append(log_entry)

with open(LOG_FILE, "w") as f:
    json.dump(logs, f, indent=4)

print(f"✅ HMM Terminé : {regime_name} (Confiance : {confidence:.1%}). Log mis à jour.")