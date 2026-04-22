"""
🧠 META CONTROLLEUR v5.0 — INSTITUTIONAL GRADE + SENTIMENT ENGINE
Intègre la robustesse v4.4 (Darwin Paranoïaque, Déduplication, SPY+QQQ) 
ET les Sentiments v5.0 (VADER) en mode Shadow.
"""

import json
import os
import glob
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ── LOGGING & CHEMINS ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🧠 MASTER BRAIN - %(levelname)s - %(message)s')
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "global_settings.json")
KMEANS_FILE   = os.path.join(BASE_DIR, "kmeans_log.json")
HMM_FILE      = os.path.join(BASE_DIR, "regime_log.json")
SENTIMENT_FILE= os.path.join(BASE_DIR, "sentiment_log.json")

TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

CONFIRMATION_JOURS = 3
ALLOCATION_CAP     = 0.40  # Max 40% du capital par bot
MIN_TRADES_SHARPE  = 10
VOLATILITE_PANIQUE = 0.30

# 🔒 SHADOW MODE: Fixé à 0.00 pour observer les sentiments sans impacter le budget
SENTIMENT_FORCE    = 0.00  

PARAMS_REGIME = {
    "BULL":    {"target_risk": 1.0, "atr_tp": 2.5, "atr_sl": 1.5, "desc": "🟢 Plein régime"},
    "NEUTRAL": {"target_risk": 0.6, "atr_tp": 2.0, "atr_sl": 1.5, "desc": "🟡 Prudence — Marché mitigé"},
    "BEAR":    {"target_risk": 0.2, "atr_tp": 1.5, "atr_sl": 2.0, "desc": "🔴 Défensif — Alerte Krach"}
}

def envoyer_telegram(message):
    if TOKEN_TELEGRAM and CHAT_ID_TELEGRAM:
        try: requests.post(f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage", data={"chat_id": CHAT_ID_TELEGRAM, "text": message, "parse_mode": "Markdown"}, timeout=10)
        except: pass

# ── MODULE 1 : LECTURE LABO ───────────────────────────────────────────────────
def normaliser_regime(regime_raw):
    if regime_raw is None: return "NEUTRAL"
    r = str(regime_raw).upper()
    
    # ✅ FIX v4.4: "NORMAL" n'est plus haussier, il tombe dans le fallback NEUTRAL
    if any(x in r for x in ["BULL", "HAUSSIER", "ACHAT", "UP"]): 
        return "BULL"
    elif any(x in r for x in ["BEAR", "BAISSIER", "CRISE", "DANGER", "VOLATILE", "DOWN"]): 
        return "BEAR"
        
    return "NEUTRAL"

def lire_donnees_labo():
    lab_intel = {"kmeans": "NEUTRAL", "hmm": "NEUTRAL", "hmm_confiance": 0.5}
    for file, key in [(KMEANS_FILE, "kmeans"), (HMM_FILE, "hmm")]:
        if os.path.exists(file):
            try:
                with open(file, "r") as f: data = json.load(f)
                entree = data[-1] if isinstance(data, list) else data
                lab_intel[key] = normaliser_regime(entree.get("regime", "NEUTRAL"))
                if key == "hmm": lab_intel["hmm_confiance"] = float(entree.get("confidence", entree.get("confiance", 0.5)))
            except: pass
    return lab_intel

# ── MODULE 2 : DÉTECTION MACRO ────────────────────────────────────────────────
def detecter_regime_brut():
    try:
        # Double Radar Macro (SPY + QQQ)
        tickers = yf.download(["SPY", "QQQ"], period="2y", progress=False)["Close"].dropna()
        if isinstance(tickers.columns, pd.MultiIndex): tickers.columns = tickers.columns.get_level_values(0)
        
        spy_close = tickers["SPY"]
        qqq_close = tickers["QQQ"]

        prix    = float(spy_close.iloc[-1])
        ma200   = float(spy_close.rolling(200).mean().iloc[-1])
        mom_20j = float(spy_close.iloc[-1] / spy_close.iloc[-20] - 1) if len(spy_close) >= 20 else 0.0
        
        vol_spy = float(spy_close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
        vol_qqq = float(qqq_close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
        vol_max = max(vol_spy, vol_qqq)

        labo = lire_donnees_labo()
        hmm_regime, hmm_conf, kmeans_regime = labo["hmm"], labo["hmm_confiance"], labo["kmeans"]
        poids_hmm = hmm_conf * 2.0

        signaux_bear = 0.0
        if prix < ma200:            signaux_bear += 1.0
        if vol_max > 0.25:          signaux_bear += 1.0
        if mom_20j < -0.05:         signaux_bear += 1.0
        if kmeans_regime == "BEAR": signaux_bear += 1.0
        if hmm_regime == "BEAR":    signaux_bear += poids_hmm

        signaux_bull = 0.0
        if prix > ma200:            signaux_bull += 1.0
        if vol_max < 0.15:          signaux_bull += 1.0
        if mom_20j > 0.02:          signaux_bull += 1.0
        if kmeans_regime == "BULL": signaux_bull += 1.0
        if hmm_regime == "BULL":    signaux_bull += poids_hmm

        if signaux_bear >= 2.5:   return "BEAR", vol_max
        elif signaux_bull >= 3.0: return "BULL", vol_max
        else:                     return "NEUTRAL", vol_max
    except Exception as e:
        logging.error(f"Erreur détection macro : {e}")
        return "NEUTRAL", 0.15

# ── MODULE 3 : FILTRE DE CONFIRMATION (Dédupliqué) ────────────────────────────
def appliquer_filtre_confirmation(regime_brut, settings_actuels):
    historique = settings_actuels.get("historique_regime_brut", [])
    aujourd_hui = datetime.now().strftime("%Y-%m-%d")

    # ✅ FIX v4.4: Déduplication temporelle (Écrase si on relance le même jour)
    if historique and historique[-1].get("date") == aujourd_hui:
        historique[-1]["regime"] = regime_brut
    else:
        historique.append({"date": aujourd_hui, "regime": regime_brut})

    historique = historique[-CONFIRMATION_JOURS:]
    regimes_recents = [h["regime"] for h in historique]

    if len(historique) >= CONFIRMATION_JOURS and len(set(regimes_recents)) == 1:
        return regimes_recents[0], historique
    return settings_actuels.get("market_regime", "NEUTRAL"), historique

# ── MODULE 4 : DARWIN V4.4 (Sortino & Data Integrity) ─────────────────────────
def calculer_darwin_allocations():
    fichiers = glob.glob(os.path.join(BASE_DIR, "portfolio_*.json"))
    if not fichiers: return {}

    scores = {}
    total_score = 0

    for f in fichiers:
        nom_bot = os.path.basename(f).replace(".json", "")
        if any(x in nom_bot.lower() for x in ["backup", "tmp", "v14_safe", "v12"]): continue

        try:
            with open(f, "r") as pf: data = json.load(pf)
            
            # ✅ FIX v4.4: Filtrage Darwin paranoïaque (Seulement de vrais trades clôturés)
            trades_fermes = [
                t for t in data.get("historique", []) 
                if t.get("action") == "VENTE" 
                and "pnl" in t 
                and t.get("mise", 0) > 0
            ]
            trades_recents = trades_fermes[-20:]

            if len(trades_recents) < MIN_TRADES_SHARPE:
                score = 5.0
            else:
                # Calcul du Ratio de Sortino robuste sur les rendements
                rendements = pd.Series([t["pnl"] / t["mise"] for t in trades_recents])
                downside = rendements[rendements < 0]
                
                if len(downside) > 0 and downside.std() > 0:
                    sortino_robuste = (rendements.mean() / downside.std()) * np.sqrt(len(trades_recents))
                else:
                    sortino_robuste = (rendements.mean() / 0.01) * np.sqrt(len(trades_recents)) if rendements.mean() > 0 else 0.0
                    
                score = max(0.5, 10 + (sortino_robuste * 5))
            
            scores[nom_bot] = score
            total_score += score
        except Exception as e:
            logging.warning(f"Alerte Darwin sur {nom_bot} : {e}")
            scores[nom_bot] = 1.0; total_score += 1.0

    if total_score == 0: return {}

    # Allocation Cap et Renormalisation
    alloc_cappees = {bot: min(ALLOCATION_CAP, s / total_score) for bot, s in scores.items()}
    total_cappe = sum(alloc_cappees.values())
    
    return {bot: round(v / total_cappe, 4) for bot, v in alloc_cappees.items()} if total_cappe > 0 else alloc_cappees

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    logging.info("🚀 MASTER BRAIN v5.0 — Déploiement Final (Data Integrity + Sentiment)")
    
    try:
        with open(SETTINGS_FILE, "r") as f: settings_actuels = json.load(f)
    except: settings_actuels = {}

    regime_brut, vol_max = detecter_regime_brut()
    regime_confirme, historique = appliquer_filtre_confirmation(regime_brut, settings_actuels)
    config = PARAMS_REGIME.get(regime_confirme, PARAMS_REGIME["NEUTRAL"])

    ancien_risque = float(settings_actuels.get("global_risk_multiplier", 0.6))
    risque_base   = config["target_risk"]
    
    # 🤖 INTÉGRATION DU SENTIMENT (Mode Shadow)
    score_sentiment = 0.0
    try:
        if os.path.exists(SENTIMENT_FILE):
            with open(SENTIMENT_FILE, "r") as f: 
                score_sentiment = json.load(f).get("sentiment_lisse", 0.0)
    except: pass

    # Sécurité sur le score et application de la force
    score_sentiment = max(-1.0, min(1.0, score_sentiment))
    sentiment_multiplier = 1 + (score_sentiment * SENTIMENT_FORCE)
    
    risque_cible = risque_base * sentiment_multiplier
    risque_cible = max(0.2, min(risque_cible, 1.5)) # Cap global du risque
    
    # Risk Smoothing Asymétrique
    risque_lisse  = risque_cible if risque_cible < ancien_risque else round((0.7 * ancien_risque) + (0.3 * risque_cible), 3)

    allow_buying = True
    panic_mode = False
    
    # Kill Switch Hybride
    if regime_brut == "BEAR" and vol_max > VOLATILITE_PANIQUE:
        allow_buying, risque_lisse, panic_mode = False, 0.0, True
        logging.warning(f"🚨 PANIC MODE ACTIVÉ : Régime BEAR + Volatilité Extrême ({vol_max:.1%}).")
    elif regime_confirme == "BEAR":
        allow_buying, risque_lisse = False, 0.0
        logging.warning("🛑 ACHATS BLOQUÉS (Tendance BEAR confirmée).")

    allocations = calculer_darwin_allocations()

    global_settings = {
        "last_update"            : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_update_by"         : "meta_controller_v5.0",
        "allow_buying"           : allow_buying,
        "panic_mode"             : panic_mode,
        "market_regime"          : regime_confirme,
        "regime_brut_today"      : regime_brut,
        "global_risk_multiplier" : risque_lisse,
        "target_risk_multiplier" : risque_cible,
        "sentiment_impact"       : f"{sentiment_multiplier:.2f}x", # 🤖 Enregistré pour l'audit
        "atr_tp_multiplier"      : config["atr_tp"],
        "atr_sl_multiplier"      : config["atr_sl"],
        "description"            : config["desc"],
        "bot_allocations"        : allocations,
        "historique_regime_brut" : historique,
    }

    temp_file = SETTINGS_FILE + ".tmp"
    with open(temp_file, "w") as f: json.dump(global_settings, f, indent=4)
    os.replace(temp_file, SETTINGS_FILE)
    logging.info(f"✅ Risk exécuté : {risque_lisse} (Ancien: {ancien_risque})")
    
    ancien_regime = settings_actuels.get("market_regime", "UNKNOWN")
    if (regime_confirme != ancien_regime and ancien_regime != "UNKNOWN") or panic_mode:
        emoji = "🚨" if panic_mode else {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}.get(regime_confirme, "⚪")
        msg = (f"🧠 *Master Brain v5.0*\n\n{emoji} {'**PANIC MODE**' if panic_mode else f'`{ancien_regime}` → `{regime_confirme}`'}\n\n"
               f"📰 Sentiment Sizing : `{sentiment_multiplier:.2f}x` (Shadow Mode)\n"
               f"📉 Cible Finale : `{risque_cible:.2f}x` | 🌊 Lissé : `{risque_lisse:.2f}x`\n"
               f"🛑 Achats : `{'OUI' if allow_buying else 'NON'}`\n\n📋 {config['desc']}")
        envoyer_telegram(msg)

if __name__ == "__main__":
    main()
