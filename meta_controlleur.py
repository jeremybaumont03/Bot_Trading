"""
META-CONTROLLER — LE CERVEAU CENTRAL DE LA FLOTTE
Rôle : Lire la météo (K-Means), appliquer les "Safety Rails", et ajuster le risque global.
"""

import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Assure-toi que ce nom correspond au fichier généré par ton script K-Means
KMEANS_FILE = os.path.join(BASE_DIR, "kmeans_log.json") 
SETTINGS_FILE = os.path.join(BASE_DIR, "global_settings.json")

def analyser_meteo():
    regime_actuel = "INCONNU"
    
    if os.path.exists(KMEANS_FILE):
        try:
            with open(KMEANS_FILE, "r") as f:
                data = json.load(f)
                if len(data) > 0:
                    regime_actuel = data[-1].get("regime", "INCONNU")
        except Exception as e:
            print(f"⚠️ Erreur lecture météo : {e}")
    else:
        print(f"⚠️ Fichier {KMEANS_FILE} introuvable. On passe en sécurité.")

    print(f"🌍 Météo des marchés détectée : {regime_actuel}")

    # 🛡️ BASE CONFIGURATION : SAFETY RAILS (Garde-fous)
    config = {
        "master_switch_active": True,
        "kill_switch": False,       # 🚨 Bouton d'urgence : Mets à True manuellement pour TOUT couper
        "max_drawdown": 0.10,       # 🚨 Si le bot détecte -10% de perte, il s'arrête (à coder dans la V3)
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # ── LA MATRICE DE DÉCISION DU RISQUE ──
    if "BULL" in regime_actuel.upper() or "NORMAL" in regime_actuel.upper():
        print("✅ Mode BULL/NORMAL : Le marché est sain. Feu vert.")
        config.update({
            "risk_multiplier": 1.0,      # Risque normal (100%)
            "atr_tp_multiplier": 2.0,    # Objectifs de gains larges
            "atr_sl_multiplier": 1.5     # Stop loss normal
        })
    elif "BEAR" in regime_actuel.upper() or "CRASH" in regime_actuel.upper():
        print("🚨 Mode BEAR/CRASH : Le marché s'effondre. Mode Survie actif !")
        config.update({
            "risk_multiplier": 0.2,      # Mises divisées par 5 (20%)
            "atr_tp_multiplier": 1.5,    # On prend ses profits plus vite
            "atr_sl_multiplier": 1.0     # Stop loss très serré
        })
    else:
        print("⚠️ Mode INCONNU : Météo incertaine. Sécurité par défaut.")
        config.update({
            "risk_multiplier": 0.5,      # Risque divisé par 2
            "atr_tp_multiplier": 2.0,
            "atr_sl_multiplier": 1.5
        })

    return config

def mettre_a_jour_cerveau():
    nouveaux_parametres = analyser_meteo()
    
    with open(SETTINGS_FILE, "w") as f:
        json.dump(nouveaux_parametres, f, indent=4)
        
    print(f"\n🧠 Cerveau Central ({SETTINGS_FILE}) mis à jour avec succès !")
    print(json.dumps(nouveaux_parametres, indent=2))

if __name__ == "__main__":
    print("🤖 Démarrage du Meta-Controller...")
    mettre_a_jour_cerveau()