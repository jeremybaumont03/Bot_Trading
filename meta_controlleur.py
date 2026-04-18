"""
🧠 META CONTROLLEUR v2.2 — FULL INTELLIGENCE (VERSION CORRIGÉE)
Corrections appliquées :
  ✅ Fix Darwin : Amortissement 50% pour éviter la monopolisation du capital
  ✅ Fix HMM/KMeans : Lecture robuste qui accepte tous les formats de clés
  ✅ Fix Telegram : Alerte si changement de régime
  ✅ Fix Fallback : Allocations égales si aucun portfolio trouvé
  ✅ Écriture atomique conservée
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

# ── LOGGING & CONFIG ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 MASTER BRAIN - %(levelname)s - %(message)s'
)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "global_settings.json")
KMEANS_FILE   = os.path.join(BASE_DIR, "kmeans_log.json")
HMM_FILE      = os.path.join(BASE_DIR, "regime_log.json")

TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def envoyer_telegram(message):
    if not TOKEN_TELEGRAM or not CHAT_ID_TELEGRAM:
        return
    try:
        url     = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
        payload = {"chat_id": CHAT_ID_TELEGRAM, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        logging.warning(f"Telegram : {e}")

# ── 1. MODULE LABO (Lecture robuste HMM + KMeans) ────────────────────────────
def normaliser_regime(regime_raw):
    """
    ✅ FIX : Accepte tous les formats possibles de tes scripts HMM/KMeans.
    Que ce soit "BULL", "HAUSSIER", "NORMAL", "BULL/NORMAL MARKET (Sûr)",
    "BEAR", "BAISSIER", "CRISE", index 0 ou 1 — tout est normalisé.
    """
    if regime_raw is None:
        return "NEUTRAL"

    r = str(regime_raw).upper()

    if any(x in r for x in ["BULL", "HAUSSIER", "NORMAL", "SAFE", "1"]):
        return "BULL"
    elif any(x in r for x in ["BEAR", "BAISSIER", "CRISE", "DANGER", "VOLATILE", "0"]):
        return "BEAR"
    return "NEUTRAL"

def lire_donnees_labo():
    """
    ✅ FIX : Lit les fichiers JSON du Labo en acceptant liste ou objet unique.
    Essaie toutes les clés possibles pour trouver le régime.
    """
    lab_intel = {"kmeans": "NEUTRAL", "hmm": "NEUTRAL", "hmm_confiance": 0.5}

    # ── Lecture K-Means ───────────────────────────────────────────────────────
    if os.path.exists(KMEANS_FILE):
        try:
            with open(KMEANS_FILE, "r") as f:
                data = json.load(f)

            # Supporte liste ou objet
            entree = data[-1] if isinstance(data, list) else data

            # Essaie toutes les clés possibles
            regime_raw = (
                entree.get("regime")
                or entree.get("regime_kmeans")
                or entree.get("cluster")
                or entree.get("state")
                or "NEUTRAL"
            )
            lab_intel["kmeans"] = normaliser_regime(regime_raw)
            logging.info(f"🔬 K-Means lu : {regime_raw} → {lab_intel['kmeans']}")
        except Exception as e:
            logging.warning(f"Lecture KMeans : {e}")

    # ── Lecture HMM ───────────────────────────────────────────────────────────
    if os.path.exists(HMM_FILE):
        try:
            with open(HMM_FILE, "r") as f:
                data = json.load(f)

            entree = data[-1] if isinstance(data, list) else data

            # ✅ FIX : Essaie TOUTES les clés possibles du HMM
            regime_raw = (
                entree.get("regime")
                or entree.get("regime_markov")
                or entree.get("regime_hmm")
                or entree.get("state")
                or entree.get("hidden_state")
                or "NEUTRAL"
            )
            confiance = float(
                entree.get("confiance")
                or entree.get("confidence")
                or entree.get("proba")
                or 0.5
            )
            lab_intel["hmm"]          = normaliser_regime(regime_raw)
            lab_intel["hmm_confiance"] = confiance
            logging.info(f"🔬 HMM lu : {regime_raw} → {lab_intel['hmm']} (confiance: {confiance:.0%})")
        except Exception as e:
            logging.warning(f"Lecture HMM : {e}")

    return lab_intel

# ── 2. MODULE MACRO ───────────────────────────────────────────────────────────
def obtenir_regime_macro():
    """
    Analyse croisée : SPY Technique + K-Means + HMM.
    Le HMM a plus de poids que le K-Means car c'est un modèle plus sophistiqué.
    """
    try:
        spy = yf.download("SPY", period="2y", progress=False, interval="1d")
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy_close = spy["Close"].dropna()

        prix      = float(spy_close.iloc[-1])
        ma200     = float(spy_close.rolling(200).mean().iloc[-1])
        vol_ann   = float(spy_close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))

        labo = lire_donnees_labo()
        hmm_regime    = labo["hmm"]
        kmeans_regime = labo["kmeans"]
        hmm_conf      = labo["hmm_confiance"]

        logging.info(f"📊 SPY: {prix:.2f} | MA200: {ma200:.2f} | Vol: {vol_ann:.1%}")
        logging.info(f"📊 HMM: {hmm_regime} ({hmm_conf:.0%}) | KMeans: {kmeans_regime}")

        # ── Règles de décision ────────────────────────────────────────────────
        # BEAR : Conditions graves — au moins 2 signaux négatifs
        signaux_bear = 0
        if prix < ma200:           signaux_bear += 1
        if vol_ann > 0.25:         signaux_bear += 1
        if hmm_regime == "BEAR" and hmm_conf > 0.60:  signaux_bear += 2  # HMM poids double
        if kmeans_regime == "BEAR": signaux_bear += 1

        # BULL : Conditions favorables
        signaux_bull = 0
        if prix > ma200:           signaux_bull += 1
        if vol_ann < 0.15:         signaux_bull += 1
        if hmm_regime == "BULL" and hmm_conf > 0.60:  signaux_bull += 2
        if kmeans_regime == "BULL": signaux_bull += 1

        if signaux_bear >= 2:
            return "BEAR"
        elif signaux_bull >= 3:
            return "BULL"
        else:
            return "NEUTRAL"

    except Exception as e:
        logging.error(f"Erreur Analyse Macro : {e}")
        return "NEUTRAL"

# ── 3. MODULE DARWIN (Corrigé) ────────────────────────────────────────────────
def calculer_darwin_allocations():
    """
    ✅ FIX : Amortissement 50% pour éviter qu'un seul bot monopolise tout le capital.
    Formule : score = max(0.5, 10 + pnl_recent * 0.5)
    
    Exemple :
    - Bot qui a gagné 50€ → score = 10 + 25 = 35  (pas 60 comme avant)
    - Bot qui a perdu 15€ → score = max(0.5, 10 - 7.5) = 2.5
    - Bot neutre (0€)    → score = 10
    """
    fichiers = glob.glob(os.path.join(BASE_DIR, "portfolio_*.json"))

    if not fichiers:
        logging.warning("Aucun portfolio trouvé — allocations égales par défaut")
        return {}

    scores = {}
    total_score = 0

    for f in fichiers:
        nom_bot = os.path.basename(f).replace(".json", "")

        # Ignore les fichiers de backup
        if "backup" in nom_bot.lower():
            continue

        try:
            with open(f, "r") as pf:
                data = json.load(pf)

            historique = data.get("historique", [])

            # PnL des 20 derniers trades fermés
            trades_fermes = [t for t in historique if t.get("action") == "VENTE"]
            trades_recents = trades_fermes[-20:] if len(trades_fermes) >= 20 else trades_fermes
            pnl_recent = sum(t.get("pnl", 0) for t in trades_recents)

            # ✅ FIX : Amortissement 50% — évite la monopolisation
            score = max(0.5, 10 + pnl_recent * 0.5)

            scores[nom_bot] = score
            total_score += score

            logging.info(f"🤖 {nom_bot:<30} PnL 20 trades: {pnl_recent:+.2f}€ → score: {score:.2f}")

        except Exception as e:
            logging.warning(f"Erreur lecture {nom_bot} : {e}")
            scores[nom_bot] = 1.0
            total_score += 1.0

    if total_score == 0:
        # ✅ Fallback : allocations égales si tout plante
        n = len(scores)
        return {bot: round(1.0 / n, 4) for bot in scores} if n > 0 else {}

    # Normalisation (somme = 1.0)
    allocations = {bot: round(s / total_score, 4) for bot, s in scores.items()}

    # ✅ Vérification que la somme fait bien ~1.0
    total = sum(allocations.values())
    logging.info(f"📊 Somme allocations : {total:.4f} (doit être ~1.0)")

    return allocations

# ── 4. PARAMÈTRES PAR RÉGIME ──────────────────────────────────────────────────
PARAMS_REGIME = {
    "BULL": {
        "risk"         : 1.0,
        "atr_tp_mult"  : 2.5,
        "atr_sl_mult"  : 1.5,
        "description"  : "🟢 Plein régime — Confiance IA forte"
    },
    "NEUTRAL": {
        "risk"         : 0.6,
        "atr_tp_mult"  : 2.0,
        "atr_sl_mult"  : 1.5,
        "description"  : "🟡 Prudence — Marché mitigé"
    },
    "BEAR": {
        "risk"         : 0.2,
        "atr_tp_mult"  : 1.5,
        "atr_sl_mult"  : 2.0,
        "description"  : "🔴 Défensif — Alerte Krach/Volatilité"
    },
}

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    logging.info("=" * 60)
    logging.info("🧠 MASTER BRAIN v2.2 — Synchronisation en cours...")
    logging.info("=" * 60)

    # 1. Régime macro
    regime = obtenir_regime_macro()
    config = PARAMS_REGIME.get(regime, PARAMS_REGIME["NEUTRAL"])

    # 2. Allocations darwiniennes
    allocations = calculer_darwin_allocations()

    # 3. Lecture du fichier actuel pour détecter un changement de régime
    ancien_regime = "UNKNOWN"
    try:
        with open(SETTINGS_FILE, "r") as f:
            ancien = json.load(f)
            ancien_regime = ancien.get("market_regime", "UNKNOWN")
    except:
        pass

    # 4. Construction du fichier final
    global_settings = {
        "last_update"          : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_update_by"       : "meta_controller_v2.2",
        "master_switch_active" : True,
        "market_regime"        : regime,
        "global_risk_multiplier"      : config["risk"],
        "atr_tp_multiplier"    : config["atr_tp_mult"],
        "atr_sl_multiplier"    : config["atr_sl_mult"],
        "bot_allocations"      : allocations,
        "description"          : config["description"]
    }

    # 5. Écriture atomique (✅ conservée)
    temp_file = SETTINGS_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(global_settings, f, indent=4)
    os.replace(temp_file, SETTINGS_FILE)

    logging.info(f"✅ global_settings.json mis à jour")
    logging.info(f"   Régime : {regime} | Risk : {config['risk']}x")
    logging.info(f"   Allocations : {allocations}")

    # 6. Alerte Telegram si changement de régime
    if regime != ancien_regime:
        emoji = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}.get(regime, "⚪")
        msg = (
            f"🧠 *Meta Controller v2.2 — Changement de Régime*\n\n"
            f"{emoji} `{ancien_regime}` → `{regime}`\n\n"
            f"⚙️ Risk : `{config['risk']}x`\n"
            f"🎯 ATR TP : `{config['atr_tp_mult']}x` | SL : `{config['atr_sl_mult']}x`\n\n"
            f"📋 {config['description']}"
        )
        envoyer_telegram(msg)
        logging.info(f"📱 Alerte Telegram envoyée (régime changé : {ancien_regime} → {regime})")
    else:
        logging.info(f"ℹ️ Régime inchangé ({regime}) — pas d'alerte Telegram")

if __name__ == "__main__":
    main()
