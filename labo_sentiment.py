"""
📰 LABO SENTIMENT V6.1 — HEDGE FUND GRADE
Optimisations : Contextual Regex, Real-time Decay, Dynamic EMA, Robust Confidence & Bounding.
"""
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import os
import time
import math
import re
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_FILE = os.path.join(BASE_DIR, "sentiment_log.json")

# 1. Pondération des sources optimisée
FEEDS = [
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ,NVDA,AAPL,MSFT", "poids": 1.0, "nom": "Yahoo"},
    {"url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", "poids": 1.3, "nom": "CNBC"}
]

def ajustement_regex(titre):
    """Corrige le biais de VADER pour le langage financier."""
    titre = titre.lower()
    score = 0.0
    if re.search(r"inflation (falls|cools|drops|easing)", titre):
        score += 0.5
    if re.search(r"beats? (expectations|estimates|forecasts)", titre):
        score += 0.6
    if re.search(r"miss(es)? (expectations|estimates|forecasts)", titre):
        score -= 0.6
    if re.search(r"rate (hikes?|increases?)", titre):
        score -= 0.4
    if re.search(r"rate (cuts?|decreases?)", titre):
        score += 0.4
    return score

def lire_ancien_score():
    """Récupère l'ancien score et l'ancienne confiance pour le lissage EMA."""
    try:
        if os.path.exists(SENTIMENT_FILE):
            with open(SENTIMENT_FILE, "r") as f:
                data = json.load(f)
                return data.get("sentiment_lisse", 0.0), data.get("confidence_score", 0.5)
    except Exception as e:
        print(f"⚠️ Info : Impossible de lire l'ancien score ({e}). Démarrage à zéro.")
    return 0.0, 0.5

def analyser_sentiment():
    start_time = time.time()
    print("⚡ Chargement VADER (Alpha Lexicon)...")
    analyzer = SentimentIntensityAnalyzer()

    # Mise à jour du lexique avec du jargon de marché
    mots_finance = {
        "crash": -4.0, "plummet": -4.0, "bear": -3.0, "bull": 3.0, "surge": 3.0,
        "layoffs": -3.0, "recession": -4.0, "growth": 2.0, "rally": 3.0
    }
    analyzer.lexicon.update(mots_finance)

    titres_vus = set()
    scores_individuels = []
    somme_des_poids = 0.0
    score_total_pondere = 0.0

    print("📰 Scraping RSS & Analyse en temps réel...")
    for feed_info in FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:30]:
                titre_brut = getattr(entry, "title", "").strip()
                if not titre_brut:
                    continue

                # Nettoyage pro (garde les pourcentages et le symbole dollar)
                titre_clean = re.sub(r'[^a-zA-Z0-9\s%$]', '', titre_brut.lower()).strip()
                if len(titre_clean.split()) < 4 or titre_clean in titres_vus:
                    continue
                titres_vus.add(titre_clean)

                # Calcul du poids par âge réel (Decay 12h)
                published = entry.get("published_parsed")
                if published:
                    age_hours = (time.time() - time.mktime(published)) / 3600
                    recency_weight = math.exp(-age_hours / 12)
                else:
                    recency_weight = 0.5

                poids_final = feed_info["poids"] * max(0.1, recency_weight)

                # Score combiné (VADER + Regex Contextuel)
                vader_score = analyzer.polarity_scores(titre_brut)["compound"]
                regex_score = ajustement_regex(titre_brut)
                final_score_titre = max(-1.0, min(1.0, vader_score + regex_score))

                scores_individuels.append(final_score_titre)
                score_total_pondere += final_score_titre * poids_final
                somme_des_poids += poids_final
        except Exception as e:
            print(f"⚠️ Erreur sur le flux {feed_info['nom']} : {e}")

    news_count = len(titres_vus)
    score_du_jour = (score_total_pondere / somme_des_poids) if somme_des_poids > 0 else 0.0

    # Calcul de la Confiance Robuste (Volume + Faible Dispersion)
    variance = float(np.std(scores_individuels)) if len(scores_individuels) > 1 else 0.5
    confidence_brute = (news_count / 25.0) * (1 - (variance * 0.5))
    confidence = max(0.0, min(1.0, confidence_brute)) # Bornage strict de sécurité

    # EMA Dynamique (plus réactif si la confiance est élevée)
    score_hier, _ = lire_ancien_score()
    alpha_brut = 0.5 + (confidence * 0.3)
    alpha = max(0.5, min(0.8, alpha_brut)) # Bornage strict de sécurité
    score_lisse = (alpha * score_du_jour) + (1 - alpha) * score_hier

    # Détection d'état fine pour le Meta-Controller
    if score_lisse < -0.5:
        etat = "CRASH"
    elif score_lisse < -0.2:
        etat = "PANIQUE"
    elif score_lisse > 0.5:
        etat = "BULLE"
    elif score_lisse > 0.2:
        etat = "EUPHORIE"
    else:
        etat = "NEUTRE"

    duree = round((time.time() - start_time) * 1000)

    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentiment_brut_jour": round(score_du_jour, 3),
        "sentiment_lisse": round(score_lisse, 3),
        "news_count": news_count,
        "confidence_score": round(confidence, 2),
        "etat": etat,
        "audit": {
            "variance": round(variance, 3),
            "alpha_used": round(alpha, 2),
            "runtime_ms": duree
        }
    }

if __name__ == "__main__":
    resultat = analyser_sentiment()
    
    # Écriture Atomique recommandée (Optionnel mais plus sûr)
    temp_file = SENTIMENT_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(resultat, f, indent=4)
    os.replace(temp_file, SENTIMENT_FILE)
    
    print(f"✅ Sentiment Pro V6.1 : {resultat['etat']} (Score: {resultat['sentiment_lisse']}, Confiance: {resultat['confidence_score']})")
    print(f"📊 Audit : {resultat['news_count']} news lues, Variance: {resultat['audit']['variance']}, Exécution: {resultat['audit']['runtime_ms']}ms")
