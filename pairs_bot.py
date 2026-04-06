"""
⚖️ BOT DE PAIRS TRADING — PRODUCTION (Scanner Silencieux)
Améliorations :
  - Aucun graphique (Headless pour GitHub Actions)
  - Shadow Logging : Enregistre les anomalies dans pairs_log.json
  - Alertes Telegram pour les élastiques tendus
  - Connecté au Cerveau Central
"""

import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import itertools
import json
import os
import sys
import requests
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# --- 🧠 LECTURE DU CERVEAU CENTRAL ---
try:
    with open("global_settings.json", "r") as f:
        settings = json.load(f)
        
    if settings.get("master_switch_active") == False:
        print("⛔ DANGER MARCHÉ : Le Cerveau Central a désactivé ce bot.")
        sys.exit()
        
    risk_multiplier = settings.get("risk_multiplier", 1.0)
    print(f"✅ Bot autorisé. Multiplicateur de risque actuel : {risk_multiplier}x")

except FileNotFoundError:
    print("⚠️ Fichier global_settings.json introuvable. Exécution normale par défaut.")
    risk_multiplier = 1.0
  
# ── CONFIGURATION TELEGRAM ────────────────────────────────────────────────────
TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TICKERS = [
    # Mega-Cap Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    # Semiconductors
    "AMD", "INTC", "TSM", "QCOM",
    # Finance
    "JPM", "V", "BAC", "GS",
    # Consumer & Industrial
    "WMT", "JNJ", "PG", "HD", "DIS",
    # Crypto
    "BTC-USD", "ETH-USD",
    # ETFs
    "SPY", "QQQ", "IWM", "TLT", "GLD", "XLK", "XLF",
]

LOG_FILE = "pairs_log.json"

# ── FONCTION TELEGRAM ─────────────────────────────────────────────────────────
def envoyer_alerte_telegram(message):
    if not TOKEN_TELEGRAM or not CHAT_ID_TELEGRAM:
        print("ℹ️ Telegram non configuré — alerte ignorée")
        return
    try:
        url     = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
        payload = {"chat_id": CHAT_ID_TELEGRAM, "text": message, "parse_mode": "HTML"}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Erreur Telegram : {e}")

# ── MAIN PROCESS ──────────────────────────────────────────────────────────────
print(f"📥 1. Téléchargement des données pour {len(TICKERS)} actifs...")
data = yf.download(TICKERS, period="2y", progress=False)['Close'].ffill().dropna()

print(f"🧮 2. Scan des {len(list(itertools.combinations(TICKERS, 2)))} combinaisons (Test de Co-intégration)...")
all_pairs = list(itertools.combinations(TICKERS, 2))
active_signals = []

for asset_A, asset_B in all_pairs:
    price_A = data[asset_A]
    price_B = data[asset_B]
    
    score, p_value, _ = coint(price_A, price_B)
    
    # Si la relation mathématique est solide (> 95% de confiance)
    if p_value < 0.05:
        X = sm.add_constant(price_B)
        ols_model = sm.OLS(price_A, X).fit()
        hedge_ratio = ols_model.params.iloc[1]
        
        spread = price_A - (hedge_ratio * price_B)
        rolling_mean = spread.rolling(window=30).mean()
        rolling_std = spread.rolling(window=30).std()
        
        z_score = ((spread - rolling_mean) / rolling_std).iloc[-1]
        
        # On ne garde que les élastiques sur le point de craquer
        if z_score > 2.0 or z_score < -2.0:
            action = "SHORT THE SPREAD" if z_score > 2.0 else "LONG THE SPREAD"
            active_signals.append({
                "asset_A": asset_A,
                "asset_B": asset_B,
                "z_score": z_score,
                "action": action,
                "price_A": price_A.iloc[-1],
                "price_B": price_B.iloc[-1]
            })

# --- 3. SAUVEGARDE DANS LE JOURNAL JSON ---
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        try:
            logs = json.load(f)
        except:
            logs = []
else:
    logs = []

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if not active_signals:
    print("🟡 Aucun signal extrême aujourd'hui.")
    logs.append({
        "date": timestamp,
        "status": "NO_SIGNAL",
        "message": "Aucun élastique n'est tendu au-delà de 2.0 aujourd'hui."
    })
    telegram_msg = "⚖️ <b>PAIRS TRADING BOT</b>\nScan terminé : Aucun écart extrême aujourd'hui. 🟡"
else:
    # Trier les signaux par l'anomalie la plus forte
    active_signals.sort(key=lambda x: abs(x['z_score']), reverse=True)
    
    telegram_msg = f"⚖️ <b>PAIRS TRADING BOT : ALERTES !</b> ⚖️\n\n"
    for sig in active_signals:
        logs.append({
            "date": timestamp,
            "asset_A": sig["asset_A"],
            "asset_B": sig["asset_B"],
            "price_A": round(float(sig["price_A"]), 2),
            "price_B": round(float(sig["price_B"]), 2),
            "z_score": round(float(sig["z_score"]), 2),
            "action": sig["action"]
        })
        emoji = "🔴" if sig["action"] == "SHORT THE SPREAD" else "🟢"
        telegram_msg += (
            f"{emoji} <b>{sig['asset_A']} / {sig['asset_B']}</b>\n"
            f"Tension (Z-Score): <b>{sig['z_score']:.2f}</b>\n"
            f"Action: {sig['action']}\n\n"
        )

with open(LOG_FILE, "w") as f:
    json.dump(logs, f, indent=4)

# --- 4. ENVOI TELEGRAM ---
print(telegram_msg)
envoyer_alerte_telegram(telegram_msg)
print("✅ Fin du script. Fichier JSON mis à jour.")
