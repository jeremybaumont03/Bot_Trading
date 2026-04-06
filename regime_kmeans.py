"""
🚲 LIGNE B : K-MEANS META-BOT (Météo Basique)
Détecte le régime de marché par regroupement de la volatilité.
Version Serveur : Sauvegarde en JSON, aucun graphique.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

LOG_FILE = "kmeans_log.json"

print("📥 Téléchargement des données SPY pour le K-Means...")
data = yf.download("SPY", period="5y", progress=False)['Close'].ffill().dropna()

returns = np.log(data / data.shift(1)).dropna()
volatility = returns.rolling(window=20).std().dropna()

returns = returns.loc[volatility.index]
X = np.column_stack([returns, volatility])

print("🧠 Entraînement du modèle K-Means (2 clusters)...")
kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_model.fit(X)

current_cluster = kmeans_model.predict([X[-1]])[0]

# Interprétation : Le cluster avec le centre de volatilité le plus élevé est le Bear Market
cluster_centers = kmeans_model.cluster_centers_
bear_cluster = np.argmax(cluster_centers[:, 1]) # La colonne 1 est la volatilité

if current_cluster == bear_cluster:
    regime_name = "BEAR MARKET (Danger)"
else:
    regime_name = "BULL/NORMAL MARKET (Sûr)"

# --- SAUVEGARDE JSON ---
log_entry = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "K-Means",
    "regime": regime_name
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

print(f"✅ K-Means Terminé : {regime_name}. Log mis à jour.")