"""
MASTER BRAIN — meta_controller.py
1. Analyse la performance de chaque bot (Darwin).
2. Analyse le régime de marché (Macro via SPY).
3. Génère les ordres de capital pour toute la flotte.
"""

import json
import os
import glob
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "global_settings.json")

def obtenir_regime_macro():
    """Macro simple et robuste basée sur le S&P 500 (Trend + Volatilité)."""
    try:
        spy = yf.download("SPY", period="2y", interval="1d", progress=False)["Close"].dropna()
        if isinstance(spy, pd.DataFrame): spy = spy.squeeze()

        prix = float(spy.iloc[-1])
        ma200 = float(spy.rolling(200).mean().iloc[-1])
        vol = float(spy.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))

        # Filtre Tendance + Risque
        if prix < ma200 and vol > 0.18:
            return "BEAR"
        elif prix > ma200 and vol < 0.15:
            return "BULL"
        else:
            return "NEUTRAL"
    except Exception as e:
        print(f"⚠️ Erreur Macro fallback SPY: {e}")
        return "NEUTRAL"

def calculer_darwin_allocations():
    """Analyse les fichiers portfolio_*.json pour noter les bots."""
    fichiers = glob.glob(os.path.join(BASE_DIR, "portfolio_*.json"))
    scores = {}
    total_score = 0
    
    for f in fichiers:
        nom = os.path.basename(f).replace(".json", "")
        try:
            with open(f, "r") as pf:
                data = json.load(pf)
                hist = data.get("historique", [])
                # Prend les 20 derniers trades
                pnl_recent = sum(t.get("pnl", 0) for t in hist[-20:]) 
                
                # Darwinisme : +10 points de base pour ne tuer personne au début
                score = max(0, pnl_recent + 10) 
                scores[nom] = score
                total_score += score
        except:
            scores[nom] = 1
            total_score += 1
            
    # Transformation des scores en pourcentages
    if total_score > 0:
        allocations = {bot: round(s / total_score, 4) for bot, s in scores.items()}
    else:
        allocations = {bot: round(1.0 / len(scores), 4) for bot in scores.keys()}
        
    return allocations

def main():
    print("🧠 Master Brain démarré...")
    
    regime = obtenir_regime_macro()
    allocations = calculer_darwin_allocations()
    
    # Paramètres de risque stricts selon le régime
    params_macro = {
        "BULL": {"risk": 1.0, "desc": "Tendance haussière : Plein régime"},
        "NEUTRAL": {"risk": 0.7, "desc": "Marché incertain : Prudence normale"},
        "BEAR": {"risk": 0.3, "desc": "CRISE (BEAR) : Mode survie, Cash prioritaire"}
    }
    
    config = params_macro.get(regime, params_macro["NEUTRAL"])
    
    global_settings = {
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market_regime": regime,
        "global_risk_multiplier": config["risk"],
        "description": config["desc"],
        "bot_allocations": allocations
    }
    
    with open(SETTINGS_FILE, "w") as f:
        json.dump(global_settings, f, indent=4)
    
    print(f"✅ Master Brain mis à jour.")
    print(f"🌍 Régime Macro : {regime} | Risque Global : {config['risk']}x")
    print("⚖️ Allocations actuelles :")
    for bot, alloc in allocations.items():
        print(f"   - {bot:<20} : {alloc*100:>5.1f}%")

if __name__ == "__main__":
    main()
