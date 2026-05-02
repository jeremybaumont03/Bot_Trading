"""
🎯 MOTEUR ALPHA & PORTFOLIO (PROP DESK V15) — VERSION SENTIMENT
Le cerveau d'exécution final du Mini Hedge Fund avec intégration VADER.
  ✅ Ajustement Dynamique : Le seuil d'achat s'adapte au Sentiment Macro (FOMO vs Paranoïa).
  ✅ Vrai Cross-Sectional : Les features sont rankées quotidiennement sur tout l'univers.
  ✅ Clustering K-Means : Regroupe les candidats par comportement.
  ✅ Sécurité Institutionnelle : Refuse de trader si les ordres du Boss ont plus de 2 heures.
  ✅ Sauvegarde Atomique : Protège le portfolio contre la corruption de fichiers.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
import shutil
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    "AMD", "INTC", "TSM", "QCOM", "JPM", "V", "BAC", "GS",
    "WMT", "JNJ", "PG", "HD", "DIS", "BTC-USD", "ETH-USD",
    "SPY", "QQQ", "IWM", "TLT", "GLD", "XLK", "XLF",
]

CAPITAL_DEPART  = 1000.0
MAX_POSITIONS   = 3
MIN_PROBA_BASE  = 0.52  # Seuil de base, sera ajusté par les sentiments
FRAIS           = 0.001
SLIPPAGE        = 0.0005

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE  = os.path.join(BASE_DIR, "global_settings.json")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio_v15_fund.json")
BACKUP_DIR     = os.path.join(BASE_DIR, "backups")

# ── CACHE GLOBAL (1 seul download pour tout le script) ────────────────────────
DF_WIDE  = None   
DF_PANEL = None   

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

# ── BACKUP ────────────────────────────────────────────────────────────────────
def faire_backup():
    if not os.path.exists(PORTFOLIO_FILE):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dest = os.path.join(BACKUP_DIR, f"portfolio_v15_{datetime.now().strftime('%Y-%m-%d')}.json")
    if not os.path.exists(dest):
        shutil.copy2(PORTFOLIO_FILE, dest)
        print(f"💾 Backup : {os.path.basename(dest)}")

# ── PORTFOLIO ─────────────────────────────────────────────────────────────────
def charger_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    p = {
        "capital_depart"   : CAPITAL_DEPART,
        "capital_cash"     : CAPITAL_DEPART,
        "positions"        : {},
        "historique"       : [],
        "valeur_historique": []
    }
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=4)
    return p

def sauvegarder_portfolio(portfolio):
    # Atomic write to prevent portfolio corruption
    temp_file = PORTFOLIO_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(portfolio, f, indent=4, default=str)
    os.replace(temp_file, PORTFOLIO_FILE)

def get_prix(ticker):
    global DF_WIDE
    if DF_WIDE is not None and ticker in DF_WIDE.columns:
        return float(DF_WIDE[ticker].dropna().iloc[-1])
    return 0.0

def calculer_nav(portfolio):
    nav = portfolio["capital_cash"]
    for ticker, pos in portfolio["positions"].items():
        prix = get_prix(ticker)
        nav += pos["quantite"] * prix if prix > 0 else pos["mise"]
    return round(nav, 2)

# ── 1. LECTURE DU CERVEAU MACRO & SENTIMENTS ──────────────────────────────────
def lire_ordres_macro():
    try:
        if not os.path.exists(SETTINGS_FILE):
            raise FileNotFoundError("global_settings.json introuvable.")

        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)

        # 🚨 SÉCURITÉ INSTITUTIONNELLE : Check fraîcheur (Timeout 2h)
        last_update_str = settings.get("last_update", "2000-01-01 00:00:00")
        last_update = datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
        age_secondes = (datetime.now() - last_update).total_seconds()
        
        if age_secondes > 7200:
            print(f"🛑 [CRITICAL] Ordres du Boss trop vieux ({age_secondes}s). Sécurité activée.")
            sys.exit(1)

        if not settings.get("master_switch_active", True):
            print("🛑 Master Switch OFF — arrêt.")
            sys.exit(0)

        regime    = settings.get("market_regime", "NEUTRAL")
        risk_mult = settings.get("global_risk_multiplier", 1.0)
        allow_buying = regime in ["BULL", "NEUTRAL"]
        
        # 🤖 LECTURE DU SENTIMENT
        sentiment_str = settings.get("sentiment_impact", "1.00x")
        try:
            sentiment_val = float(sentiment_str.replace("x", ""))
        except:
            sentiment_val = 1.0

        print(f"🧠 Ordres Macro : Régime={regime} | Risk={risk_mult}x | Sentiment={sentiment_val}x | Achats={'✅' if allow_buying else '🛑'}")

        settings["allow_buying"] = allow_buying
        settings["global_risk_multiplier"] = risk_mult
        settings["sentiment_val"] = sentiment_val # Passé au ML
        return settings

    except Exception as e:
        print(f"❌ Erreur lecture Macro : {e}")
        sys.exit(1)

# ── 2. DOWNLOAD UNIQUE BATCH ──────────────────────────────────────────────────
def charger_donnees():
    global DF_WIDE
    print("⚡ Téléchargement de l'univers (1 seul batch)...")
    try:
        raw = yf.download(TICKERS, period="3y", interval="1d", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            DF_WIDE = raw["Close"]
        else:
            DF_WIDE = raw[["Close"]]

        DF_WIDE = DF_WIDE.dropna(thresh=int(len(TICKERS) * 0.8))
        print(f"✅ {len(DF_WIDE.columns)} tickers | {len(DF_WIDE)} jours chargés")

    except Exception as e:
        print(f"❌ Erreur download : {e}")
        sys.exit(1)

# ── 3. CONSTRUCTION DU PANEL CROSS-SECTIONAL ──────────────────────────────────
def construire_panel_global():
    global DF_WIDE, DF_PANEL
    print("🔧 Construction du panel cross-sectionnel...")

    df = DF_WIDE.stack().reset_index()
    df.columns = ["Date", "Ticker", "Close"]
    df = df.set_index(["Date", "Ticker"]).sort_index()

    df["Ret_1d"]   = df.groupby(level="Ticker")["Close"].pct_change()
    df["Ret_10d"]  = df.groupby(level="Ticker")["Close"].pct_change(10)
    df["Ret_20d"]  = df.groupby(level="Ticker")["Close"].pct_change(20)
    df["Vol_20d"]  = df.groupby(level="Ticker")["Ret_1d"].transform(lambda x: x.rolling(20, min_periods=10).std())
    df["Drawdown"] = df.groupby(level="Ticker")["Close"].transform(lambda x: x / x.cummax() - 1)

    def rsi_vectorise(series):
        delta = series.diff()
        gain  = delta.where(delta > 0, 0).rolling(14, min_periods=7).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
        rs    = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    df["RSI"] = df.groupby(level="Ticker")["Close"].transform(rsi_vectorise)

    cross_features = ["Ret_10d", "Ret_20d", "Vol_20d", "Drawdown", "RSI"]
    for feat in cross_features:
        df[f"Rank_{feat}"] = df.groupby(level="Date")[feat].rank(pct=True)

    df["Future_Ret_10d"] = df.groupby(level="Ticker")["Close"].shift(-10) / df["Close"] - 1
    df["Target"] = (df["Future_Ret_10d"] > 0.015).astype(int)

    DF_PANEL = df.dropna()
    print(f"✅ Panel : {len(DF_PANEL)} lignes | {DF_PANEL.index.get_level_values('Ticker').nunique()} tickers")
    return DF_PANEL

# ── 4. MACHINE LEARNING ALPHA (AVEC SENTIMENT) ────────────────────────────────
def entrainer_et_predire_alpha(macro_settings):
    global DF_PANEL
    print("🤖 Entraînement du modèle Alpha Global...")

    features = ["Rank_Ret_10d", "Rank_Ret_20d", "Rank_Vol_20d", "Rank_Drawdown", "Rank_RSI"]

    jours_uniques      = DF_PANEL.index.get_level_values("Date").unique().sort_values()
    date_aujourd_hui   = jours_uniques[-1]
    date_limite_train  = jours_uniques[-15] 

    train_data   = DF_PANEL[DF_PANEL.index.get_level_values("Date") <= date_limite_train]
    predict_data = DF_PANEL[DF_PANEL.index.get_level_values("Date") == date_aujourd_hui]

    if len(train_data) < 100 or len(predict_data) == 0:
        print("⚠️ Données insuffisantes pour le ML")
        return pd.DataFrame()

    X_train = train_data[features].values
    y_train = train_data["Target"].values

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=30,
        max_features="sqrt", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    X_pred = predict_data[features].values
    probs  = model.predict_proba(X_pred)[:, 1]

    resultats = predict_data.reset_index()[["Ticker", "Close", "Vol_20d"]].copy()
    resultats["Proba_Alpha"] = probs

    # 🤖 AJUSTEMENT DYNAMIQUE DU SEUIL SELON LE SENTIMENT
    sentiment_val = macro_settings.get("sentiment_val", 1.0)
    
    # Si le sentiment est bon (ex: 1.15), le seuil baisse (plus d'opportunités, FOMO)
    # Si le sentiment est mauvais (ex: 0.85), le seuil monte (plus exigeant, Paranoïa)
    seuil_dynamique = MIN_PROBA_BASE - ((sentiment_val - 1.0) * 0.15)
    seuil_dynamique = max(0.50, min(0.65, seuil_dynamique)) # Limites de sécurité dures

    candidats = (
        resultats[resultats["Proba_Alpha"] > seuil_dynamique]
        .sort_values("Proba_Alpha", ascending=False)
        .reset_index(drop=True)
    )

    print(f"🌟 {len(candidats)} candidats trouvés (Seuil dynamique {seuil_dynamique:.1%} basé sur Sentiment {sentiment_val}x)")
    return candidats

# ── 5. K-MEANS CLUSTERING (Diversification) ───────────────────────────────────
def filtrer_par_clustering(candidats, max_places):
    global DF_WIDE

    if len(candidats) <= max_places:
        return candidats

    print(f"🧩 K-Means clustering pour diversifier {len(candidats)} candidats...")

    tickers_candidats = candidats["Ticker"].tolist()
    tickers_dispo = [t for t in tickers_candidats if t in DF_WIDE.columns]
    
    if len(tickers_dispo) < 2:
        return candidats.head(max_places)

    returns = DF_WIDE[tickers_dispo].pct_change().tail(60).dropna().T

    if returns.shape[0] < 2 or returns.shape[1] < 10:
        return candidats.head(max_places)

    scaler    = StandardScaler()
    X_cluster = scaler.fit_transform(returns)

    n_clusters = min(len(tickers_dispo), max_places)
    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    labels = kmeans.fit_predict(X_cluster)

    candidats_clustered = candidats[candidats["Ticker"].isin(tickers_dispo)].copy()
    ticker_to_cluster   = dict(zip(tickers_dispo, labels))
    candidats_clustered["Cluster"] = candidats_clustered["Ticker"].map(ticker_to_cluster)

    selection = (
        candidats_clustered
        .loc[candidats_clustered.groupby("Cluster")["Proba_Alpha"].idxmax()]
        .sort_values("Proba_Alpha", ascending=False)
        .head(max_places)
    )

    for _, row in selection.iterrows():
        print(f"   ↳ {row['Ticker']:<6} (Cluster {int(row['Cluster'])}) | Alpha: {row['Proba_Alpha']:.1%}")

    return selection.reset_index(drop=True)

# ── 6. EXÉCUTION DU PORTFOLIO ─────────────────────────────────────────────────
def executer_marche(selection, macro_settings, portfolio):
    aujourd_hui  = datetime.now().strftime("%Y-%m-%d")
    allow_buying = macro_settings.get("allow_buying", False)
    risk_mult    = macro_settings.get("global_risk_multiplier", 0.6)

    trades_du_jour = []

    print(f"\n💼 EXÉCUTION DU PORTFOLIO — {aujourd_hui}")
    print("─" * 60)

    # ── 1. Gestion des sorties ────────────────────────────────────────────────
    for ticker in list(portfolio["positions"].keys()):
        pos         = portfolio["positions"][ticker]
        prix_actuel = get_prix(ticker) 

        if prix_actuel <= 0:
            continue

        jours_detention = (datetime.now() - datetime.strptime(pos["date_achat"], "%Y-%m-%d")).days
        rendement = (prix_actuel - pos["prix_achat"]) / pos["prix_achat"]

        vendre, raison, emoji = False, "", "🔴"

        if not allow_buying:
            vendre, raison, emoji = True, "MACRO BEAR MODE", "🚨"
        elif prix_actuel <= pos.get("sl_cible", 0):
            vendre, raison, emoji = True, "STOP LOSS", "🛑"
        elif prix_actuel >= pos.get("tp_cible", float("inf")):
            vendre, raison, emoji = True, "TAKE PROFIT", "✅"
        elif jours_detention >= 10:
            vendre, raison, emoji = True, "TIME STOP (10j)", "⏱️"

        if vendre:
            valeur_vente = pos["quantite"] * prix_actuel
            frais_vente  = valeur_vente * (FRAIS + SLIPPAGE)
            valeur_nette = valeur_vente - frais_vente
            pnl          = valeur_nette - pos["mise"]

            portfolio["capital_cash"] += valeur_nette

            trade = {
                "date"       : aujourd_hui,
                "ticker"     : ticker,
                "action"     : "VENTE",
                "raison"     : raison,
                "prix"       : prix_actuel,
                "pnl"        : round(pnl, 2),
                "pnl_pct"    : round(rendement * 100, 2),
                "duree_jours": jours_detention
            }
            portfolio["historique"].append(trade)
            trades_du_jour.append(trade)
            del portfolio["positions"][ticker]

            print(f"{emoji} VENTE {ticker:<6} | {raison:<18} | PnL: {pnl:+.1f}€ ({rendement:+.1%})")
        else:
            print(f"🔵 HOLD  {ticker:<6} | PnL actuel: {rendement:+.1%} | Jours: {jours_detention}/10")

    # ── 2. Gestion des entrées ────────────────────────────────────────────────
    places_dispo = MAX_POSITIONS - len(portfolio["positions"])

    if not allow_buying:
        print("🛑 Achats bloqués par le Moteur Macro (Régime BEAR).")

    elif places_dispo > 0 and len(selection) > 0:
        for _, row in selection.head(places_dispo).iterrows():
            ticker = row["Ticker"]
            prix   = row["Close"]

            if ticker in portfolio["positions"]:
                continue

            vol_ann = float(row["Vol_20d"]) * np.sqrt(252) if row["Vol_20d"] > 0 else 0.20

            # Taille de position : Risk Parity × Macro Risk Multiplier
            poids_cible = min(0.30, 0.15 / (vol_ann + 1e-6))
            mise_brute  = portfolio["capital_cash"] * poids_cible * risk_mult
            frais_achat = mise_brute * (FRAIS + SLIPPAGE)
            mise_nette  = mise_brute - frais_achat

            if portfolio["capital_cash"] >= mise_brute and mise_nette > 5:
                quantite = mise_nette / prix

                vol_jour = vol_ann / np.sqrt(252)
                sl_cible = round(prix * (1 - vol_jour * 3), 4)   
                tp_cible = round(prix * (1 + vol_jour * 5), 4)   

                portfolio["capital_cash"] -= mise_brute
                portfolio["positions"][ticker] = {
                    "quantite"  : round(quantite, 6),
                    "prix_achat": prix,
                    "date_achat": aujourd_hui,
                    "mise"      : round(mise_brute, 2),
                    "sl_cible"  : sl_cible,
                    "tp_cible"  : tp_cible,
                    "proba"     : round(float(row["Proba_Alpha"]), 4),
                    "cluster"   : int(row.get("Cluster", -1))
                }

                trade = {
                    "date"    : aujourd_hui,
                    "ticker"  : ticker,
                    "action"  : "ACHAT",
                    "prix"    : prix,
                    "quantite": round(quantite, 6),
                    "mise"    : round(mise_brute, 2),
                    "frais"   : round(frais_achat, 2),
                    "sl_cible": sl_cible,
                    "tp_cible": tp_cible
                }
                portfolio["historique"].append(trade)
                trades_du_jour.append(trade)

                print(f"🟢 ACHAT {ticker:<6} | Alpha: {row['Proba_Alpha']:.1%} | Mise: {mise_brute:.1f}€ | SL: {sl_cible:.2f} | TP: {tp_cible:.2f}")
            else:
                print(f"⚠️ Cash insuffisant pour {ticker} (besoin: {mise_brute:.1f}€)")
    else:
        print("😴 Aucun candidat ou plus de place disponible.")

    return portfolio, trades_du_jour

# ── 7. RÉSUMÉ ─────────────────────────────────────────────────────────────────
def afficher_resume(portfolio):
    aujourd_hui   = datetime.now().strftime("%Y-%m-%d")
    nav           = calculer_nav(portfolio)
    perf          = (nav - portfolio["capital_depart"]) / portfolio["capital_depart"] * 100

    portfolio["valeur_historique"].append({"date": aujourd_hui, "valeur": nav})

    trades_fermes = [t for t in portfolio["historique"] if t.get("action") == "VENTE"]
    nb_trades     = len(trades_fermes)
    wins          = [t for t in trades_fermes if t.get("pnl", 0) > 0]
    win_rate      = len(wins) / nb_trades * 100 if nb_trades > 0 else 0

    print("\n" + "═" * 60)
    print(f"💼 RÉSUMÉ V15 Prop Desk — {aujourd_hui}")
    print("═" * 60)
    print(f"  NAV actuelle   : {nav:.2f} € ({perf:+.2f}%)")
    print(f"  Cash           : {portfolio['capital_cash']:.2f} €")
    print(f"  Positions      : {len(portfolio['positions'])} / {MAX_POSITIONS}")
    print(f"  Trades fermés  : {nb_trades} (Win rate: {win_rate:.0f}%)")
    print("═" * 60)

    return portfolio

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🦅 MOTEUR ALPHA & PORTFOLIO V15 — PROP DESK (SENTIMENT)")
    print(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 60)

    faire_backup()
    
    # 1. Lit les ordres et le Sentiment
    macro_settings = lire_ordres_macro()
    
    charger_donnees()
    construire_panel_global()
    
    # 2. Transmet le Sentiment au modèle Alpha
    candidats = entrainer_et_predire_alpha(macro_settings)
    
    places_dispo = MAX_POSITIONS - len(charger_portfolio().get("positions", {}))
    selection    = filtrer_par_clustering(candidats, MAX_POSITIONS)

    portfolio              = charger_portfolio()
    portfolio, trades      = executer_marche(selection, macro_settings, portfolio)
    portfolio              = afficher_resume(portfolio)
    
    # 💾 SAUVEGARDE ATOMIQUE DU PORTFOLIO
    sauvegarder_portfolio(portfolio)

    nav_fin = calculer_nav(portfolio)
    if trades:
        lignes = []
        for t in trades:
            if t["action"] == "ACHAT":
                lignes.append(f"🟢 ACHAT {t['ticker']} @ {t['prix']:.2f} | {t['mise']:.0f}€")
            elif t["action"] == "VENTE":
                emoji = {"TAKE PROFIT": "✅", "STOP LOSS": "🛑", "TIME STOP (10j)": "⏱️"}.get(t.get("raison", ""), "🔴")
                lignes.append(f"{emoji} VENTE {t['ticker']} — PnL: {t.get('pnl', 0):+.0f}€ ({t.get('raison', '')})")
        envoyer_telegram(f"🦅 *V15 Prop Desk*\n\n" + "\n".join(lignes) + f"\n\n💰 NAV: {nav_fin:.2f}€")
    else:
        envoyer_telegram(f"😴 *V15 Scan Terminé*\nAucun mouvement.\n💰 NAV: {nav_fin:.2f}€")
