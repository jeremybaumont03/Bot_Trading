"""
META CONTROLLER — PRODUCTION GRADE (v2.0)
Architecture robuste : Auto-réparation, Fallbacks, Logs propres, Validation JSON.
"""

import json
import os
import glob
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ── LOGGING & CONFIG ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🧠 MASTER BRAIN - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "global_settings.json")

# ── 1. MODULE MACRO (AVEC FALLBACK) ───────────────────────────────────────────
def obtenir_regime_macro():
    """Analyse robuste du S&P 500. Ne plante jamais, retourne NEUTRAL en cas d'erreur."""
    try:
        spy = yf.download("SPY", period="2y", interval="1d", progress=False)["Close"].dropna()
        if isinstance(spy, pd.DataFrame): 
            spy = spy.squeeze()

        if len(spy) < 200:
            logging.warning("Pas assez de données SPY. Fallback: NEUTRAL.")
            return "NEUTRAL"

        prix = float(spy.iloc[-1])
        ma200 = float(spy.rolling(200).mean().iloc[-1])
        vol = float(spy.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))

        if prix < ma200 and vol > 0.18:
            return "BEAR"
        elif prix > ma200 and vol < 0.15:
            return "BULL"
        else:
            return "NEUTRAL"
    except Exception as e:
        logging.error(f"Erreur API Yahoo Finance: {e}. Fallback: NEUTRAL.")
        return "NEUTRAL"

# ── 2. MODULE DARWIN (AUTO-RÉPARANT) ──────────────────────────────────────────
def calculer_darwin_allocations():
    """Scanne les portfolios. Si aucun portfolio n'existe, retourne un dictionnaire vide."""
    fichiers = glob.glob(os.path.join(BASE_DIR, "portfolio_*.json"))
    
    if not fichiers:
        logging.warning("Aucun fichier portfolio_*.json détecté. Phase d'initialisation.")
        return {}

    scores = {}
    total_score = 0
    
    for f in fichiers:
        nom_bot = os.path.basename(f).replace(".json", "")
        try:
            with open(f, "r") as pf:
                data = json.load(pf)
                hist = data.get("historique", [])
                
                # Récupère les 20 derniers trades en toute sécurité
                pnl_recent = sum(t.get("pnl", 0) for t in hist[-20:]) if hist else 0
                
                # Score de base : 10 points de survie
                score = max(0, pnl_recent + 10) 
                scores[nom_bot] = score
                total_score += score
        except json.JSONDecodeError:
            logging.error(f"Fichier corrompu ignoré : {nom_bot}")
            scores[nom_bot] = 1
            total_score += 1
        except Exception as e:
            logging.error(f"Erreur de lecture sur {nom_bot}: {e}")
            scores[nom_bot] = 1
            total_score += 1
            
    # Calcul des pourcentages stricts
    allocations = {}
    if total_score > 0:
        allocations = {bot: round(s / total_score, 4) for bot, s in scores.items()}
    else:
        # Égalitarisme si tous les bots sont à zéro
        parts_egales = round(1.0 / len(scores), 4)
        allocations = {bot: parts_egales for bot in scores.keys()}
        
    return allocations

# ── 3. GÉNÉRATEUR DU JSON (ATOMIC WRITE) ──────────────────────────────────────
def sauvegarder_settings(nouveau_json):
    """Sauvegarde le fichier proprement pour éviter la corruption en cours d'écriture."""
    try:
        # Écriture dans un fichier temporaire d'abord
        temp_file = SETTINGS_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(nouveau_json, f, indent=4)
        
        # Remplacement atomique (très sûr)
        os.replace(temp_file, SETTINGS_FILE)
        logging.info("global_settings.json mis à jour avec succès.")
    except Exception as e:
        logging.error(f"Impossible de sauvegarder global_settings.json : {e}")

# ── EXÉCUTION PRINCIPALE ──────────────────────────────────────────────────────
def main():
    logging.info("Démarrage de l'analyse du système...")
    
    # Étape 1 : Macro
    regime = obtenir_regime_macro()
    logging.info(f"Régime Macro détecté : {regime}")
    
    params_macro = {
        "BULL": {"risk": 1.0, "desc": "Tendance haussière - Plein régime"},
        "NEUTRAL": {"risk": 0.8, "desc": "Marché incertain - Prudence modérée"},
        "BEAR": {"risk": 0.3, "desc": "BEAR MARKET - Protection du capital max"}
    }
    config = params_macro.get(regime, params_macro["NEUTRAL"])
    
    # Étape 2 : Darwin
    allocations = calculer_darwin_allocations()
    if allocations:
        for bot, alloc in allocations.items():
            logging.info(f"Allocation Darwin -> {bot} : {alloc*100:.1f}%")
    
    # Étape 3 : Assemblage du Cerveau
    global_settings = {
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "master_switch_active": True,
        "market_regime": regime,
        "global_risk_multiplier": config["risk"],
        "description": config["desc"],
        "bot_allocations": allocations  # ⭐ LA PIÈCE MAÎTRESSE
    }
    
    sauvegarder_settings(global_settings)

if __name__ == "__main__":
    main()
