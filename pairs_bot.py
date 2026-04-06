"""
🔬 LABO 3 : MASTER PAIRS SCANNER & VISUALIZER
Scanne une liste de tickers, trouve les paires co-intégrées,
et génère automatiquement le graphique de la meilleure opportunité.
"""

import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import itertools
import warnings

# Désactiver les avertissements inutiles
warnings.filterwarnings("ignore")

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


print(f"📥 1. Téléchargement des données pour {len(TICKERS)} actifs...")
data = yf.download(TICKERS, period="2y", progress=False)['Close']
data = data.dropna()

print(f"🧮 2. Génération de toutes les paires possibles...")
all_pairs = list(itertools.combinations(TICKERS, 2))
print(f"   -> {len(all_pairs)} combinaisons à tester.")

print("\n🔬 3. Test de Co-intégration (Recherche des élastiques invisibles)...")
cointegrated_pairs = []

for asset_A, asset_B in all_pairs:
    price_A = data[asset_A]
    price_B = data[asset_B]
    
    # Test d'Engle-Granger
    score, p_value, _ = coint(price_A, price_B)
    
    # Si p_value < 0.05, l'élastique existe (Confiance > 95%)
    if p_value < 0.05:
        X = sm.add_constant(price_B)
        ols_model = sm.OLS(price_A, X).fit()
        hedge_ratio = ols_model.params.iloc[1]
        
        # Calcul du Spread et du Z-Score historique complet
        spread = price_A - (hedge_ratio * price_B)
        rolling_mean = spread.rolling(window=30).mean()
        rolling_std = spread.rolling(window=30).std()
        
        z_score_series = (spread - rolling_mean) / rolling_std
        latest_z_score = z_score_series.iloc[-1]
        
        cointegrated_pairs.append({
            'Pair': f"{asset_A} / {asset_B}",
            'Asset_A': asset_A,
            'Asset_B': asset_B,
            'P-Value': p_value,
            'Z-Score': latest_z_score,
            'Z-Series': z_score_series # On sauvegarde toute l'histoire pour le graphique
        })

print("\n" + "="*50)
print(" 🏆 RÉSULTATS DU SCANNER (Classés par tension)")
print("="*50)

if not cointegrated_pairs:
    print("Aucune paire valide trouvée dans cette liste.")
else:
    # Trier par l'anomalie la plus forte (valeur absolue du Z-Score)
    cointegrated_pairs.sort(key=lambda x: abs(x['Z-Score']), reverse=True)
    
    for item in cointegrated_pairs:
        z = item['Z-Score']
        if z > 2.0:
            signal = "🔴 SHORT THE SPREAD (Tendu vers le haut)"
        elif z < -2.0:
            signal = "🟢 LONG THE SPREAD (Tendu vers le bas)"
        else:
            signal = "🟡 HOLD (Zone normale)"
            
        print(f"🔹 {item['Pair']}")
        print(f"   Confiance : {1 - item['P-Value']:.1%}")
        print(f"   Z-Score   : {z:.2f} -> {signal}\n")

    # --- 4. TRACER AUTOMATIQUEMENT LA MEILLEURE PAIRE ---
    best_pair = cointegrated_pairs[0]
    best_z_series = best_pair['Z-Series']
    asset_A = best_pair['Asset_A']
    asset_B = best_pair['Asset_B']
    
    print("="*50)
    print(f"📊 Ouverture du graphique pour l'anomalie #1 : {asset_A} vs {asset_B}")
    print("="*50)
    
    plt.figure(figsize=(12, 6))
    plt.plot(best_z_series.index, best_z_series.values, label=f"Z-Score ({asset_A}/{asset_B})", color='blue')
    
    # Lignes de limites d'intervention
    plt.axhline(0, color='black', linestyle='--', label='Moyenne (0)')
    plt.axhline(2.0, color='red', linestyle='--', label='Action : SHORT THE SPREAD (+2.0)')
    plt.axhline(-2.0, color='green', linestyle='--', label='Action : LONG THE SPREAD (-2.0)')
    
    plt.title(f"Arbitrage Statistique : Tension de l'écart entre {asset_A} et {asset_B}")
    plt.xlabel("Date")
    plt.ylabel("Z-Score (Tension)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Afficher le graphique (bloque le script tant que la fenêtre n'est pas fermée)
    plt.show()