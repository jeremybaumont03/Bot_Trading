"""
ULTIMATE META-CONTROLLER (Darwin + Macro HMM/KMeans + Secours SPY)
Rôle : 
1. Lit le Laboratoire (HMM/KMeans) ou le SPY pour le Risque Global.
2. Lit les performances des bots pour allouer le capital (Darwin).
3. Met à jour global_settings.json pour tous les bots.
"""

import json
import os
import glob
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "global_settings.json")
HMM_FILE      = os.path.join(BASE_DIR, "regime_log.json")
KMEANS_FILE   = os.path.join(BASE_DIR, "kmeans_log.json")

TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── RÈGLES DE DÉCISION MACRO ──────────────────────────────────────────────────
REGLES = {
    ("BEAR", "BEAR"):    (0.2, 1.0, 2.0, "🔴 CRISE TOTALE — Défensif maximum"),
    ("BEAR", "NEUTRAL"): (0.3, 1.2, 1.8, "🔴 Bear confirmé — Mode survie"),
    ("BEAR", "BULL"):    (0.4, 1.5, 1.5, "🟡 Divergence — Prudence maximale"),
    ("NEUTRAL", "BEAR"): (0.5, 1.5, 1.5, "🟡 Marché incertain — Semi-défensif"),
    ("NEUTRAL", "NEUTRAL"): (0.7, 2.0, 1.5, "🟡 Marché neutre — Normal réduit"),
    ("NEUTRAL", "BULL"): (0.85, 2.0, 1.5, "🟢 Tendance positive — Normal"),
    ("BULL", "BEAR"):    (0.6, 1.8, 1.5, "🟡 Divergence — Prudence"),
    ("BULL", "NEUTRAL"): (0.9, 2.0, 1.5, "🟢 Bull confirmé — Normal+"),
    ("BULL", "BULL"):    (1.0, 2.5, 1.5, "🟢 BULL TOTAL — Plein régime"),
}
REGLE_DEFAUT = (0.5, 2.0, 1.5, "⚪ Prudence par défaut")

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def envoyer_telegram(message):
    if not TOKEN_TELEGRAM or not CHAT_ID_TELEGRAM: return
    try:
        url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID_TELEGRAM, "text": message, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e: print(f"⚠️ Telegram : {e}")

# ── LECTURE MACRO (HMM / KMEANS / SPY) ────────────────────────────────────────
def lire_fichiers_labo():
    regime_hmm, confiance_hmm, regime_kmeans = "NEUTRAL", 0.0, "NEUTRAL"
    labo_actif = False
    try:
        if os.path.exists(HMM_FILE):
            with open(HMM_FILE, "r") as f:
                data = json.load(f)
                derniere_entree = data[-1] if isinstance(data, list) else data
                regime_hmm = derniere_entree.get("regime", "NEUTRAL").upper()
                confiance_hmm = float(derniere_entree.get("confiance", 0.5))
                if "BULL" in regime_hmm or "HAUSSIER" in regime_hmm: regime_hmm = "BULL"
                elif "BEAR" in regime_hmm or "BAISSIER" in regime_hmm: regime_hmm = "BEAR"
                labo_actif = True
    except: pass

    try:
        if os.path.exists(KMEANS_FILE):
            with open(KMEANS_FILE, "r") as f:
                data = json.load(f)
                derniere_entree = data[-1] if isinstance(data, list) else data
                regime_kmeans = derniere_entree.get("regime", "NEUTRAL").upper()
                if "BULL" in regime_kmeans or "HAUSSIER" in regime_kmeans: regime_kmeans = "BULL"
                elif "BEAR" in regime_kmeans or "BAISSIER" in regime_kmeans: regime_kmeans = "BEAR"
    except: pass
    
    return labo_actif, regime_hmm, confiance_hmm, regime_kmeans

def lire_spy_fallback():
    """Analyse robuste du S&P 500 si le Labo est en panne."""
    try:
        spy = yf.download("SPY", period="2y", interval="1d", progress=False)["Close"].dropna()
        if isinstance(spy, pd.DataFrame): spy = spy.squeeze()
        prix = float(spy.iloc[-1])
        ma200 = float(spy.rolling(200).mean().iloc[-1])
        vol = float(spy.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
        
        if prix < ma200 and vol > 0.18: return "BEAR"
        elif prix > ma200 and vol < 0.15: return "BULL"
        else: return "NEUTRAL"
    except: return "NEUTRAL"

def calculer_nouveaux_params(regime_hmm, confiance_hmm, regime_kmeans):
    cle = (regime_hmm, regime_kmeans)
    risk, atr_tp, atr_sl, description = REGLES.get(cle, REGLE_DEFAUT)
    if confiance_hmm > 0 and confiance_hmm < 0.60:
        risk = min(risk, 0.7)
        description += f" (confiance HMM faible: {confiance_hmm:.0%})"
    return risk, atr_tp, atr_sl, description

# ── LECTURE DARWIN (ALLOCATION BOTS) ──────────────────────────────────────────
def calculer_darwin_allocations():
    """Analyse les fichiers portfolio_*.json pour noter et allouer le capital."""
    fichiers = glob.glob(os.path.join(BASE_DIR, "portfolio_*.json"))
    scores = {}
    total_score = 0
    
    for f in fichiers:
        nom = os.path.basename(f).replace(".json", "")
        try:
            with open(f, "r") as pf:
                data = json.load(pf)
                hist = data.get("historique", [])
                pnl_recent = sum(t.get("pnl", 0) for t in hist[-20:]) 
                score = max(0, pnl_recent + 10) # +10 pour survie initiale
                scores[nom] = score
                total_score += score
        except:
            scores[nom] = 1; total_score += 1
            
    if total_score > 0: return {bot: round(s / total_score, 4) for bot, s in scores.items()}
    else: return {bot: round(1.0 / len(scores), 4) for bot in scores.keys()}

# ── EXÉCUTION PRINCIPALE ──────────────────────────────────────────────────────
def main():
    aujourd_hui = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n🧠 MASTER BRAIN — {aujourd_hui}")
    print("═" * 60)

    # 1. Macro Analysis
    labo_actif, regime_hmm, confiance_hmm, regime_kmeans = lire_fichiers_labo()
    
    if labo_actif:
        print(f"  📡 HMM     : {regime_hmm} (confiance: {confiance_hmm:.0%})")
        print(f"  📡 K-Means : {regime_kmeans}")
        nouveau_risk, nouveau_tp, nouveau_sl, description = calculer_nouveaux_params(regime_hmm, confiance_hmm, regime_kmeans)
    else:
        regime_spy = lire_spy_fallback()
        print(f"  📡 SPY Secours : {regime_spy}")
        # On simule HMM et KMeans identiques pour utiliser les règles
        nouveau_risk, nouveau_tp, nouveau_sl, description = calculer_nouveaux_params(regime_spy, 1.0, regime_spy)
        description = f"SECours SPY Actif — {description}"

    # 2. Darwin Analysis
    allocations = calculer_darwin_allocations()

    # 3. Sauvegarde Globale
    try:
        with open(SETTINGS_FILE, "r") as f: ancien_risk = json.load(f).get("global_risk_multiplier", 1.0)
    except: ancien_risk = 1.0

    nouveaux_settings = {
        "last_update": aujourd_hui,
        "market_regime": regime_hmm if labo_actif else regime_spy,
        "global_risk_multiplier": nouveau_risk,
        "atr_tp_multiplier": nouveau_tp,
        "atr_sl_multiplier": nouveau_sl,
        "description": description,
        "bot_allocations": allocations
    }

    with open(SETTINGS_FILE, "w") as f:
        json.dump(nouveaux_settings, f, indent=4)

    print(f"\n  ✅ global_settings.json mis à jour :")
    print(f"     Risk     : {ancien_risk}x → {nouveau_risk}x")
    print(f"     Décision : {description}")
    print("\n  ⚖️ Allocations Darwin (Capital) :")
    for bot, alloc in allocations.items(): print(f"     - {bot:<20} : {alloc*100:>5.1f}%")
    print("═" * 60)

    # 4. Telegram Alert
    if abs(nouveau_risk - ancien_risk) >= 0.1 or len(allocations) > 0:
        msg = (
            f"🧠 *Master Brain — Mise à jour*\n\n"
            f"🌍 *Macro* : `{description}`\n"
            f"⚙️ *Risk Multiplier* : `{ancien_risk}x` → `{nouveau_risk}x`\n\n"
            f"🧬 *Allocations Darwin* :\n"
        )
        for bot, alloc in allocations.items(): msg += f"• {bot.replace('portfolio_','')} : `{alloc*100:.0f}%`\n"
        envoyer_telegram(msg)

if __name__ == "__main__":
    main()
