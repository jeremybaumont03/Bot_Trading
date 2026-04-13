"""
meta_controller.py — Le Cerveau Autonome du Hedge Fund
Rôle : Lit les fichiers du Laboratoire (HMM + KMeans) et ajuste
       automatiquement global_settings.json pour protéger tous les bots.
Lance ce script 5 minutes AVANT tes bots de trading.
"""

import json
import os
import requests
from datetime import datetime

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "global_settings.json")
HMM_FILE      = os.path.join(BASE_DIR, "regime_log.json")
KMEANS_FILE   = os.path.join(BASE_DIR, "kmeans_log.json")

TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── RÈGLES DE DÉCISION ────────────────────────────────────────────────────────
# Ces règles traduisent la "météo" du marché en paramètres de risque concrets.
REGLES = {
    # (regime_hmm, regime_kmeans) -> (risk_multiplier, atr_tp_mult, atr_sl_mult, description)
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
REGLE_DEFAUT = (0.5, 2.0, 1.5, "⚪ Données insuffisantes — Prudence par défaut")

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def envoyer_telegram(message):
    if not TOKEN_TELEGRAM or not CHAT_ID_TELEGRAM:
        return
    try:
        url     = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
        payload = {"chat_id": CHAT_ID_TELEGRAM, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Telegram : {e}")

# ── LECTURE DES FICHIERS LABO ─────────────────────────────────────────────────
def lire_regime_hmm():
    """Lit le dernier régime HMM depuis regime_log.json"""
    try:
        with open(HMM_FILE, "r") as f:
            data = json.load(f)
        # Support liste ou objet unique
        if isinstance(data, list):
            derniere_entree = data[-1]
        else:
            derniere_entree = data
        regime = derniere_entree.get("regime", "NEUTRAL").upper()
        confiance = derniere_entree.get("confiance", 0.5)
        # Normalisation des labels
        if "BULL" in regime or "HAUSSIER" in regime or "NORMAL" in regime:
            regime = "BULL"
        elif "BEAR" in regime or "BAISSIER" in regime or "CRISE" in regime:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"
        return regime, float(confiance)
    except FileNotFoundError:
        print(f"⚠️ {HMM_FILE} introuvable — HMM ignoré")
        return "NEUTRAL", 0.0
    except Exception as e:
        print(f"⚠️ Erreur lecture HMM : {e}")
        return "NEUTRAL", 0.0

def lire_regime_kmeans():
    """Lit le dernier régime K-Means depuis kmeans_log.json"""
    try:
        with open(KMEANS_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            derniere_entree = data[-1]
        else:
            derniere_entree = data
        regime_raw = derniere_entree.get("regime", "NEUTRAL").upper()
        # Normalisation des labels K-Means (souvent en français)
        if "BULL" in regime_raw or "HAUSSIER" in regime_raw or "NORMAL" in regime_raw:
            regime = "BULL"
        elif "BEAR" in regime_raw or "BAISSIER" in regime_raw or "CRISE" in regime_raw:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"
        return regime
    except FileNotFoundError:
        print(f"⚠️ {KMEANS_FILE} introuvable — KMeans ignoré")
        return "NEUTRAL"
    except Exception as e:
        print(f"⚠️ Erreur lecture KMeans : {e}")
        return "NEUTRAL"

# ── LECTURE DES SETTINGS ACTUELS ──────────────────────────────────────────────
def charger_settings_actuels():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "master_switch_active": True,
            "risk_multiplier"     : 1.0,
            "atr_tp_multiplier"   : 2.0,
            "atr_sl_multiplier"   : 1.5,
            "description"         : "Master Controller",
            "last_update"         : ""
        }

# ── LOGIQUE PRINCIPALE ────────────────────────────────────────────────────────
def calculer_nouveaux_params(regime_hmm, confiance_hmm, regime_kmeans):
    """
    Détermine les nouveaux paramètres de risque selon la météo.
    Si la confiance HMM est faible (<60%), on pèse moins le HMM.
    """
    cle = (regime_hmm, regime_kmeans)
    risk, atr_tp, atr_sl, description = REGLES.get(cle, REGLE_DEFAUT)

    # Pondération par la confiance HMM
    if confiance_hmm < 0.60:
        # Confiance faible → on réduit l'effet du régime, on reste prudent
        risk = min(risk, 0.7)
        description += f" (confiance HMM faible: {confiance_hmm:.0%})"

    return round(risk, 2), round(atr_tp, 2), round(atr_sl, 2), description

def main():
    aujourd_hui = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n🧠 META-CONTROLLER — {aujourd_hui}")
    print("═" * 60)

    # 1. Lecture des régimes
    regime_hmm,    confiance_hmm = lire_regime_hmm()
    regime_kmeans                = lire_regime_kmeans()

    print(f"  📡 HMM       : {regime_hmm} (confiance: {confiance_hmm:.0%})")
    print(f"  📡 K-Means   : {regime_kmeans}")

    # 2. Calcul des nouveaux paramètres
    nouveau_risk, nouveau_tp, nouveau_sl, description = calculer_nouveaux_params(
        regime_hmm, confiance_hmm, regime_kmeans
    )

    # 3. Lecture des settings actuels
    settings_actuels = charger_settings_actuels()
    ancien_risk      = settings_actuels.get("risk_multiplier", 1.0)

    # 4. Mise à jour du fichier global_settings.json
    nouveaux_settings = {
        "master_switch_active" : settings_actuels.get("master_switch_active", True),
        "risk_multiplier"      : nouveau_risk,
        "atr_tp_multiplier"    : nouveau_tp,
        "atr_sl_multiplier"    : nouveau_sl,
        "regime_hmm"           : regime_hmm,
        "regime_kmeans"        : regime_kmeans,
        "confiance_hmm"        : round(confiance_hmm, 3),
        "description"          : description,
        "last_update"          : aujourd_hui,
        "last_update_by"       : "meta_controller.py"
    }

    with open(SETTINGS_FILE, "w") as f:
        json.dump(nouveaux_settings, f, indent=2)

    print(f"\n  ✅ global_settings.json mis à jour :")
    print(f"     Risk     : {ancien_risk}x → {nouveau_risk}x")
    print(f"     ATR TP   : {nouveau_tp}x")
    print(f"     ATR SL   : {nouveau_sl}x")
    print(f"     Décision : {description}")
    print("═" * 60)

    # 5. Alerte Telegram si changement significatif de risque
    if abs(nouveau_risk - ancien_risk) >= 0.1:
        msg = (
            f"🧠 *Meta-Controller — Mise à jour*\n\n"
            f"📡 HMM : `{regime_hmm}` ({confiance_hmm:.0%})\n"
            f"📡 KMeans : `{regime_kmeans}`\n\n"
            f"⚙️ Risk : `{ancien_risk}x` → `{nouveau_risk}x`\n"
            f"🎯 ATR TP : `{nouveau_tp}x` | SL : `{nouveau_sl}x`\n\n"
            f"📋 {description}"
        )
        envoyer_telegram(msg)
        print("📱 Alerte Telegram envoyée (changement de risque détecté)")
    else:
        print("ℹ️ Pas d'alerte Telegram (changement mineur)")

if __name__ == "__main__":
    main()
