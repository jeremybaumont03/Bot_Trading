"""
📰 LABO SENTIMENT — VADER + RSS (Hedge Fund Grade)
Intègre : Regex Clean, Pondération, Exponential Decay, EMA Smoothing.
"""
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import os
import time
import math
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_FILE = os.path.join(BASE_DIR, "sentiment_log.json")

FEEDS = [
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ,NVDA,AAPL,MSFT", "poids": 1.0, "nom": "Yahoo"},
    {"url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", "poids": 1.2, "nom": "CNBC"}
]

def lire_ancien_score():
    try:
        if os.path.exists(SENTIMENT_FILE):
            with open(SENTIMENT_FILE, "r") as f:
                return json.load(f).get("sentiment_lisse", 0.0)
    except: pass
    return 0.0

def analyser_sentiment():
    start_time = time.time()
    print("⚡ Chargement VADER (Alpha Lexicon)...")
    analyzer = SentimentIntensityAnalyzer()

    mots_finance = {
        "crash": -4.0, "plummet": -4.0, "bankrupt": -4.0, "bear": -3.0, 
        "inflation": -2.0, "downgrade": -3.0, "layoffs": -3.0, "recession": -4.0,
        "bull": 3.0, "surge": 3.0, "soar": 3.0, "record high": 3.0, 
        "outperform": 2.0, "upgrade": 3.0, "dividend": 2.0, "beat estimates": 3.0
    }
    analyzer.lexicon.update(mots_finance)

    titres_vus = set()
    score_total_pondere = 0.0
    somme_des_poids = 0.0

    print("📰 Scraping RSS...")
    for feed_info in FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for i, entry in enumerate(feed.entries[:25]):
                titre_brut = entry.title
                
                titre_clean = re.sub(r'[^\w\s]', '', titre_brut.lower()).strip()
                if len(titre_clean.split()) < 4: continue 
                
                if titre_clean not in titres_vus:
                    titres_vus.add(titre_clean)
                    score_brut_vader = analyzer.polarity_scores(titre_brut)['compound']
                    
                    recency_weight = math.exp(-i / 10.0)
                    poids_final = feed_info["poids"] * max(0.1, recency_weight)
                    
                    score_total_pondere += (score_brut_vader * poids_final)
                    somme_des_poids += poids_final
        except Exception as e:
            print(f"⚠️ Erreur flux {feed_info['nom']} : {e}")

    if len(titres_vus) < 10 or somme_des_poids == 0:
        score_du_jour = 0.0
    else:
        score_du_jour = score_total_pondere / somme_des_poids

    score_du_jour = max(-1.0, min(1.0, score_du_jour))
    score_hier = lire_ancien_score()
    score_lisse = (0.7 * score_du_jour) + (0.3 * score_hier)

    duree = round((time.time() - start_time) * 1000)
    print(f"🧠 {len(titres_vus)} news analysées en {duree} ms. Lissé final : {score_lisse:+.3f}")

    return score_du_jour, score_lisse

if __name__ == "__main__":
    score_brut, score_lisse = analyser_sentiment()
    
    if score_lisse < -0.25: etat = "PANIQUE"
    elif score_lisse > 0.25: etat = "EUPHORIE"
    else: etat = "NEUTRE"

    with open(SENTIMENT_FILE, "w") as f:
        json.dump({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sentiment_brut_jour": round(score_brut, 3),
            "sentiment_lisse": round(score_lisse, 3),
            "etat": etat
        }, f, indent=4)