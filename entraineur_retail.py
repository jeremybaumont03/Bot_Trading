"""
BOT DE PAPER TRADING — entraineur_v14.py (PROP DESK V14.2 - OBSERVABILITY & CACHE)
✅ Moteur ML Rapide (RandomForest n_jobs=-1).
✅ Exponential Weighting & Empirical EV.
✅ Observability Layer (Empreinte à l'achat, Autopsie à la vente, Dashboard).
✅ FIX : Model Registry (Sauvegarde des modèles .joblib pour éviter le Timeout GitHub).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
import shutil
import requests
import joblib
import time
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# ── CONFIGURATION TELEGRAM ────────────────────────────────────────────────────
TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── CONFIGURATION DES ACTIFS ──────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    "AMD", "INTC", "TSM", "QCOM", "JPM", "V", "BAC", "GS",
    "WMT", "JNJ", "PG", "HD", "DIS", "BTC-USD", "ETH-USD",
    "SPY", "QQQ", "IWM", "TLT", "GLD", "XLK", "XLF",
]

CAPITAL_DEPART   = 1000.0
FRAIS            = 0.001
SLIPPAGE         = 0.0005
MAX_POSITIONS    = 3
ATR_PERIOD       = 14

# ── PARAMÈTRES V14.2 ──────────────────────────────────────────────────────────
BASE_SEUIL_ENSEMBLE = 0.55    
MIN_TRADES_30D      = 5       
MAX_DD_LIMIT        = -0.15
MAX_CVAR_95         = -0.04   
COOLDOWN_JOURS      = 3
KELLY_FRACTION      = 0.5
MAX_ALLOC_PAR_TRADE = 0.25
MIN_ALLOC_PAR_TRADE = 0.02
CORR_MAX            = 0.75
VOL_TARGET          = 0.15
BASE_ATR_TP_MULT    = 2.0
BASE_ATR_SL_MULT    = 1.5
MODEL_EXPIRY_DAYS   = 7 # Les modèles expirent après 7 jours

# ── CONFIGURATION DES CHEMINS ─────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
FICHIER         = os.path.join(BASE_DIR, "portfolio_v14.json")
DOSSIER_BACKUP  = os.path.join(BASE_DIR, "backups")
DOSSIER_MODELES = os.path.join(BASE_DIR, "modeles_ia_v14")

# ── TRANSFORMER WINSORIZATION CUSTOM ──────────────────────────────────────────
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        self.lower_bounds_ = np.nanpercentile(X, self.limits[0] * 100, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, 100 - (self.limits[1] * 100), axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

# ── GESTION DE LA DONNÉE ──────────────────────────────────────────────────────
DF_CACHE = {}

def charger_donnees():
    print("⚡ Téléchargement des données (YFinance Batch)...")
    try:
        data = yf.download(TICKERS, period="6y", interval="1d", progress=False, group_by="ticker")
        for ticker in TICKERS:
            try:
                df_ticker = data[ticker].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
                if len(df_ticker) > 300 and (df_ticker['Close'].isna().sum() / len(df_ticker) <= 0.20):
                    DF_CACHE[ticker] = df_ticker
            except: pass
    except Exception as e:
        print(f"❌ Erreur globale YFinance : {e}")

    if "SPY" not in DF_CACHE:
        print("❌ ERREUR CRITIQUE : SPY introuvable. Moteur Macro désactivé.")
        sys.exit(1)

def get_prix(ticker):
    return float(DF_CACHE[ticker]['Close'].iloc[-1]) if ticker in DF_CACHE else 0.0

def calculer_nav(portfolio):
    nav = portfolio['capital_cash']
    for t, pos in portfolio['positions'].items():
        px = get_prix(t)
        nav += pos['quantite'] * px if px > 0 else pos['mise']
    return nav

# ── UNSUPERVISED REGIME DETECTION ─────────────────────────────────────────────
def detecter_regime_macro():
    spy = DF_CACHE['SPY']
    ma200 = spy['Close'].rolling(200).mean().iloc[-1]
    prix_actuel = spy['Close'].iloc[-1]
    vol_20j = spy['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    
    if prix_actuel < ma200 and vol_20j > 0.18: return "BEAR"
    elif prix_actuel > ma200 and vol_20j < 0.15: return "BULL"
    else: return "CHOP"

# ── RISK ENGINE ───────────────────────────────────────────────────────────────
def calculer_kelly(proba_gain, ratio_gain_perte):
    p, b = max(0.01, min(0.99, proba_gain)), max(0.1, ratio_gain_perte)
    return max(0.0, (p - (1 - p) / b) * KELLY_FRACTION)

def calculer_cvar_portefeuille(portfolio_positions, new_ticker, new_weight_cible, nav_actuelle):
    tickers_valides, weights_bruts = [], []
    for t, pos in portfolio_positions.items():
        px = get_prix(t)
        if px > 0 and t in DF_CACHE:
            tickers_valides.append(t)
            weights_bruts.append((pos['quantite'] * px) / nav_actuelle)
            
    if new_ticker in DF_CACHE:
        tickers_valides.append(new_ticker)
        weights_bruts.append(new_weight_cible)
    else: return 0.0 
        
    if not tickers_valides: return 0.0

    sum_w = sum(weights_bruts)
    weights = [w / sum_w for w in weights_bruts] if sum_w > 0 else weights_bruts
    df_returns = pd.DataFrame()
    
    for t in tickers_valides:
        df_returns[t] = np.log(DF_CACHE[t]['Close'] / DF_CACHE[t]['Close'].shift(1)).tail(120)
            
    df_returns = df_returns.dropna()
    if len(df_returns) < 20: return 0.0 
    
    port_returns = df_returns.dot(weights)
    sorted_returns = np.sort(port_returns)
    index = int(0.05 * len(sorted_returns))
    return np.mean(sorted_returns[:max(1, index)])

def verifier_correlation(ticker_candidat, positions_ouvertes):
    if not positions_ouvertes: return True, "OK"
    df_cand = DF_CACHE.get(ticker_candidat)
    if df_cand is None or len(df_cand) < 120: return True, "OK"
    
    rend_cand = np.log(df_cand['Close'] / df_cand['Close'].shift(1)).tail(120).dropna() 
    for t_pos in positions_ouvertes:
        if DF_CACHE.get(t_pos) is None: continue
        rend_pos = np.log(DF_CACHE[t_pos]['Close'] / DF_CACHE[t_pos]['Close'].shift(1)).tail(120).dropna()
        common = rend_cand.index.intersection(rend_pos.index)
        if len(common) >= 30 and abs(rend_cand[common].corr(rend_pos[common])) > CORR_MAX:
            return False, f"Corrélé à {t_pos}"
    return True, "OK"

# ── PIPELINE ML V14.2 (Avec Model Registry Joblib) ────────────────────────────
def creer_features_v14(df, is_spy=False):
    df = df.copy()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    df['Mom_10j'] = df['Close'] / df['Close'].shift(10)
    df['Mom_60j'] = df['Close'] / df['Close'].shift(60)
    df['Drawdown'] = df['Close'] / df['Close'].cummax() - 1
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['ATR'] = (pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)).rolling(14).mean()
    df['ATR_ratio'] = df['ATR'] / (df['Close'] + 1e-10)
    df['MACD'] = (df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()) / (df['Close'] + 1e-10)
    df['Vol_regime'] = df['Volatility'].rolling(5).mean() / (df['Volatility'].rolling(60).mean() + 1e-10)
    
    if 'SPY' in DF_CACHE:
        spy_df = DF_CACHE['SPY']
        spy_trend = ((spy_df['Close'].rolling(200).mean() / spy_df['Close'].rolling(200).mean().shift(20)) - 1).shift(1)
        spy_vix = (spy_df['Close'].pct_change().rolling(20).std() * np.sqrt(252)).shift(1)
        spy_close_t1 = spy_df['Close'].shift(1)
        
        df['SPY_Trend'] = spy_trend.reindex(df.index)
        df['SPY_VIX_Proxy'] = spy_vix.reindex(df.index)
        df['Rel_SPY'] = df['Close'] / spy_close_t1.reindex(df.index) if not is_spy else 1.0
    else:
        df['SPY_Trend'], df['SPY_VIX_Proxy'], df['Rel_SPY'] = 0.0, 0.15, 1.0

    df['Target_5j']  = (df['Close'].shift(-5)  / df['Close'] - 1) - (FRAIS + SLIPPAGE)
    df['Target_10j'] = (df['Close'].shift(-10) / df['Close'] - 1) - (FRAIS + SLIPPAGE)
    df['Target_20j'] = (df['Close'].shift(-20) / df['Close'] - 1) - (FRAIS + SLIPPAGE)
    df['Target_Raw'] = ((df['Target_5j'] > 0.01) | (df['Target_10j'] > 0.015) | (df['Target_20j'] > 0.02)).astype(int)
    
    return df

def analyser_opportunite(ticker, seuil_dynamique, est_en_position):
    if ticker not in DF_CACHE: return None
    
    df = creer_features_v14(DF_CACHE[ticker], is_spy=(ticker == "SPY"))
    features = ['MA50', 'MA200', 'Volatility', 'Mom_10j', 'Mom_60j', 'Drawdown', 'RSI', 'ATR_ratio', 'MACD', 'Vol_regime', 'Rel_SPY', 'SPY_Trend', 'SPY_VIX_Proxy']
    df = df.dropna(subset=['Rel_SPY'])
    
    os.makedirs(DOSSIER_MODELES, exist_ok=True)
    fichier_modele = os.path.join(DOSSIER_MODELES, f"v14_{ticker}.joblib")
    
    cache_ml = None
    
    # 🚨 V14.2 FIX : Model Registry Loading
    if os.path.exists(fichier_modele):
        age_jours = (time.time() - os.path.getmtime(fichier_modele)) / 86400
        if age_jours < MODEL_EXPIRY_DAYS:
            try:
                cache_ml = joblib.load(fichier_modele)
            except: pass

    if not cache_ml:
        # Entraînement complet (Exécuté 1 fois par semaine par ticker)
        df_train = df.iloc[:-20].dropna(subset=features + ['Target_Raw', 'Target_10j']) 
        if len(df_train) < 100 or len(np.unique(df_train['Target_Raw'])) < 2: return None
        
        X_train, y_train = df_train[features].values, df_train['Target_Raw'].values
        weights = np.exp(np.linspace(-2, 0, len(y_train)))
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
        pipeline = Pipeline([('winsorizer', Winsorizer()), ('scaler', RobustScaler()), ('classifier', rf)])
        pipeline.fit(X_train, y_train, classifier__sample_weight=weights) 
        
        win_returns = df_train['Target_10j'][df_train['Target_Raw'] == 1]
        loss_returns = df_train['Target_10j'][df_train['Target_Raw'] == 0]
        
        e_win = np.clip(win_returns.mean(), 0.005, 0.05) if len(win_returns) > 5 else 0.015
        e_loss = np.clip(abs(loss_returns.mean()), 0.005, 0.05) if len(loss_returns) > 5 else 0.015

        # Sauvegarde dans le registre
        cache_ml = {"pipeline": pipeline, "e_win": float(e_win), "e_loss": float(e_loss)}
        joblib.dump(cache_ml, fichier_modele)

    pipeline = cache_ml["pipeline"]
    e_win = cache_ml["e_win"]
    e_loss = cache_ml["e_loss"]

    derniers_jours = df.iloc[-3:][features].copy().dropna()
    if len(derniers_jours) < 2: return None
    
    proba_lisse = np.mean(pipeline.predict_proba(derniers_jours.values)[:, 1])
    
    seuil_effectif = seuil_dynamique - 0.02 if est_en_position else seuil_dynamique
    if proba_lisse < seuil_effectif: return None

    last_row = df.iloc[-1]
    prix, atr, vol_ann = get_prix(ticker), max(0, float(last_row['ATR'])), float(last_row['Volatility']) * np.sqrt(252)
    if atr <= 0 or vol_ann <= 0: return None

    expected_value = (proba_lisse * e_win) - ((1 - proba_lisse) * e_loss)
    
    if expected_value <= 0: return None 

    ratio_gp = e_win / e_loss
    alloc_kelly = calculer_kelly(proba_lisse, ratio_gp)
    alloc = min(MAX_ALLOC_PAR_TRADE, max(MIN_ALLOC_PAR_TRADE, alloc_kelly * (VOL_TARGET / (vol_ann + 1e-6))))

    return {'ticker': ticker, 'proba': proba_lisse, 'ev': expected_value, 'alloc': alloc, 'atr': atr, 'prix': prix}

def gerer_telegram(msg): 
    if TOKEN_TELEGRAM and CHAT_ID_TELEGRAM:
        try: requests.post(f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage", data={"chat_id": CHAT_ID_TELEGRAM, "text": msg, "parse_mode": "Markdown"}, timeout=10)
        except: pass

def faire_backup():
    if os.path.exists(FICHIER):
        os.makedirs(DOSSIER_BACKUP, exist_ok=True)
        shutil.copy2(FICHIER, os.path.join(DOSSIER_BACKUP, f"portfolio_v14_{datetime.now().strftime('%Y-%m-%d')}.json"))

# ── EXÉCUTION PRINCIPALE ──────────────────────────────────────────────────────
def executer_trades():
    if not os.path.exists(FICHIER):
        with open(FICHIER, "w") as f: json.dump({"capital_depart": CAPITAL_DEPART, "capital_cash": CAPITAL_DEPART, "positions": {}, "historique": [], "valeur_historique": []}, f)
    with open(FICHIER, "r") as f: portfolio = json.load(f)

    aujourd_hui = datetime.now().strftime("%Y-%m-%d")
    nav_actuelle = calculer_nav(portfolio)

    hist_valeurs = [h['valeur'] for h in portfolio.get('valeur_historique', [])]
    rendements = pd.Series(hist_valeurs).pct_change().dropna() if len(hist_valeurs) > 5 else pd.Series()
    current_dd = (( (1+rendements).cumprod() - (1+rendements).cumprod().cummax() ) / (1+rendements).cumprod().cummax()).min() if not rendements.empty else 0.0
    
    regime = detecter_regime_macro()
    seuil_dynamique = BASE_SEUIL_ENSEMBLE
    atr_tp, atr_sl = BASE_ATR_TP_MULT, BASE_ATR_SL_MULT
    
    if regime == "BEAR":
        seuil_dynamique += 0.03  
        atr_tp *= 0.8            
    elif regime == "BULL":
        seuil_dynamique -= 0.02  
        atr_tp *= 1.2            

    trades_30d = [t for t in portfolio.get('historique', []) if (datetime.strptime(aujourd_hui, "%Y-%m-%d") - datetime.strptime(t.get('date_sortie', t['date_entree']), "%Y-%m-%d")).days <= 30]
    if len(trades_30d) < MIN_TRADES_30D: seuil_dynamique = max(0.50, seuil_dynamique - 0.02)

    cb_actif = False
    if current_dd <= MAX_DD_LIMIT:
        portfolio['circuit_breaker_date'] = aujourd_hui
        cb_actif = True
    elif portfolio.get('circuit_breaker_date'):
        if (datetime.strptime(aujourd_hui, "%Y-%m-%d") - datetime.strptime(portfolio['circuit_breaker_date'], "%Y-%m-%d")).days < COOLDOWN_JOURS: cb_actif = True
        else: portfolio['circuit_breaker_date'] = None

    risk_adj = max(0.5, 1.0 - (current_dd / MAX_DD_LIMIT)) if current_dd < 0 else 1.0

    print(f"\n📅 {aujourd_hui} — V14.2 PROP DESK CACHED")
    print(f"   Régime: {regime} | NAV: {nav_actuelle:.2f}€ | DD: {current_dd:.1%} | Seuil: {seuil_dynamique:.2f}")
    if cb_actif: print("   🚨 CIRCUIT BREAKER ACTIF — ACHATS BLOQUÉS")
    print("─" * 90)

    for ticker in list(portfolio['positions'].keys()):
        pos = portfolio['positions'][ticker]
        prix = get_prix(ticker)
        if prix <= 0: continue
        
        rendement = (prix - pos['prix_achat']) / pos['prix_achat']
        tp, sl = pos.get('tp_cible'), pos.get('sl_cible')
        vendre, raison, emj = False, "", "🔴"
        
        if tp and prix >= tp: vendre, raison, emj = True, "TAKE PROFIT", "✅"
        elif sl and prix <= sl: vendre, raison, emj = True, "STOP LOSS", "🛑"
        else:
            opp = analyser_opportunite(ticker, seuil_dynamique, est_en_position=True)
            if opp is None: vendre, raison, emj = True, "SIGNAL DECAY", "🤖"

        if vendre:
            val_nette = (pos['quantite'] * prix) * (1 - (FRAIS + SLIPPAGE))
            portfolio['capital_cash'] += val_nette
            pnl = val_nette - pos['mise']
            jours_detention = (datetime.strptime(aujourd_hui, "%Y-%m-%d") - datetime.strptime(pos['date_achat'], "%Y-%m-%d")).days
            
            trade_log = {
                "date_entree": pos['date_achat'], "date_sortie": aujourd_hui, "ticker": ticker,
                "action": "VENTE", "raison": raison, "pnl": round(pnl, 2), "pnl_pct": round(rendement, 4),
                "jours_detention": jours_detention, "regime_entree": pos.get('regime_macro', 'UNKNOWN'),
                "proba_entree": pos.get('proba', 0), "ev_entree": pos.get('ev', 0)
            }
            portfolio['historique'].append(trade_log)
            del portfolio['positions'][ticker]
            print(f"{ticker:<8} {'—':<6} {emj} VENDU ({raison:<10}) PnL:{pnl:+.0f}€")
        else:
            print(f"{ticker:<8} {'—':<6} 🔵 EN POSITION       PnL:{rendement*100:+.1f}%")

    if not cb_actif and len(portfolio['positions']) < MAX_POSITIONS:
        candidats = []
        for ticker in [t for t in TICKERS if t not in portfolio['positions']]:
            opp = analyser_opportunite(ticker, seuil_dynamique, est_en_position=False)
            if opp: candidats.append(opp)
        
        candidats = sorted(candidats, key=lambda x: x['ev'], reverse=True)
        
        for cand in candidats:
            if len(portfolio['positions']) >= MAX_POSITIONS: break
            
            ticker, alloc, prix, atr = cand['ticker'], cand['alloc'], cand['prix'], cand['atr']
            if not verifier_correlation(ticker, list(portfolio['positions'].keys())): continue
            if calculer_cvar_portefeuille(portfolio['positions'], ticker, alloc, nav_actuelle) < MAX_CVAR_95: continue
                
            mise_nette = (nav_actuelle * alloc * risk_adj) * (1 - FRAIS - SLIPPAGE)
            if portfolio['capital_cash'] >= (mise_nette/(1-FRAIS-SLIPPAGE)) and mise_nette > 5:
                portfolio['capital_cash'] -= (mise_nette / (1 - FRAIS - SLIPPAGE))
                
                portfolio['positions'][ticker] = {
                    "quantite": mise_nette / prix, "prix_achat": prix, "date_achat": aujourd_hui,
                    "mise": mise_nette / (1 - FRAIS - SLIPPAGE), "tp_cible": prix + (atr * atr_tp), 
                    "sl_cible": prix - (atr * atr_sl), "proba": cand['proba'], "ev": cand['ev'],
                    "atr_local": atr, "regime_macro": regime, "poids_portefeuille": alloc
                }
                print(f"{ticker:<8} {cand['proba']:<6.0%} 🟢 ACHETÉ (EV:{cand['ev']:.4f})  TP:{prix+(atr*atr_tp):.2f}")

    portfolio['valeur_historique'].append({"date": aujourd_hui, "valeur": round(calculer_nav(portfolio), 2)})
    with open(FICHIER, "w") as f: json.dump(portfolio, f, indent=2)
    return portfolio

# ── RÉSUMÉ & ANALYTIQUE ───────────────────────────────────────────────────────
def afficher_resume_analytique(portfolio):
    aujourd_hui   = datetime.now().strftime("%Y-%m-%d")
    valeur_totale = calculer_nav(portfolio)
    perf_totale   = (valeur_totale - portfolio['capital_depart']) / portfolio['capital_depart'] * 100

    trades_fermes = [t for t in portfolio['historique'] if t.get('action') == 'VENTE']
    nb_trades     = len(trades_fermes)
    
    wins = [t for t in trades_fermes if t.get('pnl', 0) > 0]
    losses = [t for t in trades_fermes if t.get('pnl', 0) <= 0]
    
    win_rate = (len(wins) / nb_trades * 100) if nb_trades > 0 else 0
    gross_profit = sum(t.get('pnl', 0) for t in wins)
    gross_loss = abs(sum(t.get('pnl', 0) for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0)
    
    avg_win = np.mean([t.get('pnl_pct', 0) for t in wins]) * 100 if wins else 0
    avg_loss = np.mean([t.get('pnl_pct', 0) for t in losses]) * 100 if losses else 0
    
    hist_valeurs = [h['valeur'] for h in portfolio.get('valeur_historique', [])]
    rendements = pd.Series(hist_valeurs).pct_change().dropna() if len(hist_valeurs) > 5 else pd.Series()
    sharpe = rendements.mean() / rendements.std() * np.sqrt(252) if len(rendements) > 0 and rendements.std() > 0 else 0.0
    max_dd = (( (1+rendements).cumprod() - (1+rendements).cumprod().cummax() ) / (1+rendements).cumprod().cummax()).min() if not rendements.empty else 0.0

    print("\n" + "═" * 75)
    print(f"📊 DASHBOARD ANALYTIQUE V14.2 — {aujourd_hui}")
    print("═" * 75)
    print(f"  NAV actuelle         : {valeur_totale:.2f} € ({perf_totale:+.2f}%)")
    print(f"  Positions ouvertes   : {len(portfolio['positions'])} / {MAX_POSITIONS}")
    print(f"  Trades fermés        : {nb_trades} (Win rate: {win_rate:.0f}%)")
    print(f"  Profit Factor        : {profit_factor:.2f}")
    print(f"  Sharpe Ratio         : {sharpe:.2f} | Max Drawdown : {max_dd:.1%}")
    print("═" * 75)
    
    msg_tg = f"📊 *V14.2 CACHED*\n💰 NAV: {valeur_totale:.2f}€\n📈 WinRate: {win_rate:.0f}%\n📉 MaxDD: {max_dd:.1%}"
    gerer_telegram(msg_tg)

if __name__ == "__main__":
    faire_backup()
    charger_donnees()
    p = executer_trades()
    afficher_resume_analytique(p)
