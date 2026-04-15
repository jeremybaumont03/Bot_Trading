"""
BOT DE PAPER TRADING — V2 AGRESSIF (Edition "Canard" ATR HYBRIDE)
Améliorations :
  - Univers étendu : 31 Tickers (Grande liquidité)
  - Ultra Agressif : Seuil IA baissé à 45%
  - ATR Dynamique : TP et SL calculés en fonction de la volatilité
  - Contrôle Central : Connecté au Master Brain v2.0 (Macro + Darwin)
  - Shadow Logging : Enregistre les probabilités IA même en cash
  - Sortie Dynamique : L'IA peut couper ses pertes sans restriction
  - FIX ANTI-OVERFITTING : Entraînement Live Quant.
  - FIX PRIX 0.0 : Sécurité contre les bugs Yahoo Finance.
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

# ── CONFIGURATION TELEGRAM ────────────────────────────────────────────────────
TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    "AMD", "INTC", "TSM", "QCOM",
    "JPM", "V", "BAC", "GS",
    "WMT", "JNJ", "PG", "HD", "DIS",
    "BTC-USD", "ETH-USD",
    "SPY", "QQQ", "IWM", "TLT", "GLD", "XLK", "XLF",
]

CAPITAL_DEPART   = 1000.0
FRAIS            = 0.001       # 0.1% par trade
SLIPPAGE         = 0.0005      # 0.05% slippage
VOL_TARGET       = 0.15        # cible volatilité annualisée
MAX_POSITIONS    = 3

# ✅ PARAMÈTRES AGRESSIFS & ATR HYBRIDE
SEUIL_IA_FIXE    = 0.45        # l'IA n'a besoin d'être sûre qu'à 45%
ML_TARGET_HAUSSE = 0.015       # cible d'entrainement ML : +1.5% en 10 jours
ATR_PERIOD       = 14

DEFAULT_ATR_TP_MULT = 2.0    # Fallback TP
DEFAULT_ATR_SL_MULT = 1.5    # Fallback SL

# ── CONFIGURATION DES CHEMINS ─────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
FICHIER        = os.path.join(BASE_DIR, "portfolio_aggressive.json")
DOSSIER_BACKUP = os.path.join(BASE_DIR, "backups")
SETTINGS_FILE  = os.path.join(BASE_DIR, "global_settings.json")

# ── LECTURE DU CERVEAU CENTRAL (MAJ DARWIN) ───────────────────────────────────
def charger_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)

        if not settings.get("master_switch_active", True):
            print("🛑 MASTER SWITCH DÉSACTIVÉ — Bot en mode veille")
            return None

        # 🧠 Lecture du Risque Macro
        risk = settings.get("global_risk_multiplier", 1.0)
        
        # 🧬 Lecture de la sélection naturelle (Darwin)
        nom_fichier_bot = os.path.basename(FICHIER).replace(".json", "")
        alloc_darwin = settings.get("bot_allocations", {}).get(nom_fichier_bot, 1.0)

        print(f"🧠 Master Brain lu : Risk={risk}x | Budget Darwin={alloc_darwin*100:.1f}%")
        return {"risk": risk, "alloc_darwin": alloc_darwin}

    except Exception as e:
        print(f"⚠️ Erreur lecture Cerveau Central : {e} — Mode survie activé")
        return {"risk": 1.0, "alloc_darwin": 1.0}

# ── FONCTION TELEGRAM ─────────────────────────────────────────────────────────
def envoyer_alerte_telegram(message):
    if not TOKEN_TELEGRAM or not CHAT_ID_TELEGRAM:
        print("ℹ️ Telegram ignoré en mode local (pas de tokens).")
        return
        
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    payload = {"chat_id": CHAT_ID_TELEGRAM, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Erreur Telegram : {e}")

# ── LA FONCTION DE SAUVEGARDE ─────────────────────────────────────────────────
def faire_backup():
    if not os.path.exists(FICHIER):
        return  
    try:
        os.makedirs(DOSSIER_BACKUP, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        nom_backup = f"portfolio_aggressive_{date_str}.json"
        dest = os.path.join(DOSSIER_BACKUP, nom_backup)

        if not os.path.exists(dest):
            shutil.copy2(FICHIER, dest)
            print(f"💾 Backup sauvegardé avec succès : {nom_backup}")
    except Exception as e:
        print(f"⚠️ Erreur lors du backup : {e}")

# ── PORTFOLIO ─────────────────────────────────────────────────────────────────
def charger_portfolio():
    if os.path.exists(FICHIER):
        with open(FICHIER, "r") as f:
            return json.load(f)
    portfolio = {
        "capital_depart"   : CAPITAL_DEPART,
        "capital_cash"     : CAPITAL_DEPART,
        "positions"        : {},
        "historique"       : [],
        "valeur_historique": [],
        "logs_journaliers" : [] 
    }
    sauvegarder_portfolio(portfolio)
    print(f"✅ Nouveau portfolio créé avec {CAPITAL_DEPART}€ virtuels")
    return portfolio

def sauvegarder_portfolio(portfolio):
    with open(FICHIER, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)

# ── ATR ───────────────────────────────────────────────────────────────────────
def calculer_atr(df, period=14):
    high_low    = df['High'] - df['Low']
    high_close  = (df['High'] - df['Close'].shift()).abs()
    low_close   = (df['Low']  - df['Close'].shift()).abs()
    true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

# ── SIGNAL ────────────────────────────────────────────────────────────────────
def calculer_signal(ticker):
    try:
        df = yf.download(ticker, period="6y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 250:
            return False, 0.0, 0.0, 0.0, 0.0

        df['MA50']       = df['Close'].rolling(50).mean()
        df['MA200']      = df['Close'].rolling(200).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()
        df['Mom_20j']    = df['Close'] / df['Close'].shift(20)
        df['Drawdown']   = df['Close'] / df['Close'].cummax() - 1
        df['Target']     = (df['Close'].shift(-10) / df['Close'] - 1 > ML_TARGET_HAUSSE).astype(int)
        df['ATR']        = calculer_atr(df, ATR_PERIOD)
        df = df.dropna()

        features = ['MA50', 'MA200', 'Volatility', 'Mom_20j', 'Drawdown']
        
        # ✅ FIX PRO : Entraînement sur TOUT l'historique jusqu'à hier ([:-1])
        X_train = df[features].iloc[:-1]
        y_train = df['Target'].iloc[:-1]

        # ✅ FIX ANTI-OVERFITTING : On bride l'arbre pour forcer la généralisation
        model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=30, random_state=42)
        model.fit(X_train, y_train)

        last  = df.iloc[-1]
        proba = model.predict_proba(df[features].iloc[[-1]])[0][1]

        # ✅ IA PURE : Seulement la probabilité compte (plus de filtres)
        signal   = proba > SEUIL_IA_FIXE
        vol_ann  = float(last['Volatility']) * np.sqrt(252)
        alloc    = min(0.20, VOL_TARGET / vol_ann) if (signal and vol_ann > 0) else 0.0
        prix     = float(last['Close'])
        atr      = float(last['ATR'])

        return signal, round(proba, 4), round(alloc, 4), round(prix, 4), round(atr, 4)

    except Exception as e:
        return False, 0.0, 0.0, 0.0, 0.0

# ── VALEUR DU PORTFOLIO ───────────────────────────────────────────────────────
def calculer_valeur_totale(portfolio):
    valeur = portfolio['capital_cash']
    for ticker, pos in portfolio['positions'].items():
        try:
            prix = float(yf.download(ticker, period="2d", interval="1d", progress=False)['Close'].iloc[-1])
            valeur += prix * pos.get('quantite', pos['mise'] / pos['prix_achat'])
        except:
            valeur += pos['mise']
    return round(valeur, 2)

# ── MÉTRIQUES DE PERFORMANCE ──────────────────────────────────────────────────
def calculer_metriques(portfolio):
    historique = portfolio.get('valeur_historique', [])
    if len(historique) < 5:
        return None, None

    valeurs  = [h['valeur'] for h in historique]
    series   = pd.Series(valeurs)
    rendements = series.pct_change().dropna()

    if rendements.std() > 0:
        sharpe = rendements.mean() / rendements.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    cumul      = (1 + rendements).cumprod()
    max_dd     = ((cumul - cumul.cummax()) / cumul.cummax()).min() if not cumul.empty else 0.0

    return round(sharpe, 3), round(max_dd, 4)

# ── TRADES ────────────────────────────────────────────────────────────────────
def executer_trades(portfolio, settings):
    aujourd_hui    = datetime.now().strftime("%Y-%m-%d")
    trades_du_jour = []

    risk_mult    = settings["risk"]
    alloc_darwin = settings["alloc_darwin"]
    atr_tp_mult  = DEFAULT_ATR_TP_MULT
    atr_sl_mult  = DEFAULT_ATR_SL_MULT

    if 'logs_journaliers' not in portfolio:
        portfolio['logs_journaliers'] = []

    print(f"\n📅 Analyse du {aujourd_hui} — RF Agressif (V2 ATR Hybride)")
    print(f"   Positions ouvertes : {len(portfolio['positions'])} / {MAX_POSITIONS} | Tickers scannés : {len(TICKERS)}")
    print("─" * 90)
    print(f"{'ACTIF':<10} {'IA%':<7} {'ATR':<8} {'SIGNAL':<12} {'ACTION':<20} {'DÉTAIL'}")
    print("─" * 90)

    for ticker in TICKERS:
        signal, proba, allocation, prix, atr = calculer_signal(ticker)
        
        # ✅ THE 0.0 PRICE BUG FIX
        if prix <= 0.0:
            print(f"⚠️ Yahoo Finance API Bug for {ticker} (Price is 0.0). Skipping.")
            continue
            
        position_ouverte = ticker in portfolio['positions']
        action_str       = "⚪ CASH"
        detail           = ""

        portfolio['logs_journaliers'].append({
            "date"          : aujourd_hui,
            "ticker"        : ticker,
            "proba_ia"      : proba,
            "signal_valide" : bool(signal)
        })
        portfolio['logs_journaliers'] = portfolio['logs_journaliers'][-1000:]

        # ── VÉRIFICATION TP / SL DYNAMIQUE ──────────────────────
        if position_ouverte:
            pos        = portfolio['positions'][ticker]
            prix_achat = pos['prix_achat']
            tp_cible   = pos.get('tp_cible')
            sl_cible   = pos.get('sl_cible')
            rendement  = (prix - prix_achat) / prix_achat

            vendre = False
            raison = ""

            if tp_cible and prix >= tp_cible:
                vendre, raison = True, "TAKE PROFIT"
                emoji = "✅"
            elif sl_cible and prix <= sl_cible:
                vendre, raison = True, "STOP LOSS"
                emoji = "🛑"
            elif not signal: # L'IA n'y croit plus
                vendre, raison = True, "SIGNAL IA"
                emoji = "🤖"

            if vendre:
                quantite     = pos.get('quantite', pos['mise'] / prix_achat)
                valeur_vente = quantite * prix
                frais_vente  = valeur_vente * (FRAIS + SLIPPAGE)
                valeur_nette = valeur_vente - frais_vente
                pnl_net      = valeur_nette - pos['mise']
                duree_jours  = (datetime.now() - datetime.strptime(pos['date_achat'], "%Y-%m-%d")).days

                portfolio['capital_cash'] += valeur_nette

                trade = {
                    "date"          : aujourd_hui,
                    "ticker"        : ticker,
                    "action"        : "VENTE",
                    "raison"        : raison,
                    "prix"          : prix,
                    "quantite"      : round(quantite, 6),
                    "valeur"        : round(valeur_nette, 2),
                    "pnl"           : round(pnl_net, 2),
                    "pnl_pct"       : round(rendement * 100, 2),
                    "frais"         : round(frais_vente, 2),
                    "duree_jours"   : duree_jours,
                    "atr_lors_achat": pos.get('atr_lors_achat', 0)
                }
                portfolio['historique'].append(trade)
                trades_du_jour.append(trade)
                del portfolio['positions'][ticker]
                action_str       = f"{emoji} VENDU ({raison})"
                detail           = f"PnL: {pnl_net:+.0f}€ ({rendement*100:+.1f}%)"
                position_ouverte = False

        # ── ACHAT HYBRIDE ──────────────────────────────────────────────────
        if signal and not position_ouverte:
            if len(portfolio['positions']) >= MAX_POSITIONS:
                action_str = "🚫 MAX ATTEINT"
                detail     = f"({MAX_POSITIONS} positions max)"
            elif atr == 0.0:
                action_str = "⚠️ ATR INDISPONIBLE"
            else:
                # 🧬 LA MAGIE DARWIN OPERE ICI !
                mise_brute  = portfolio['capital_cash'] * allocation * risk_mult * alloc_darwin
                frais_achat = mise_brute * (FRAIS + SLIPPAGE)
                mise_nette  = mise_brute - frais_achat

                if portfolio['capital_cash'] >= mise_brute and mise_nette > 5:
                    quantite = mise_nette / prix
                    tp_cible = round(prix + (atr * atr_tp_mult), 4)
                    sl_cible = round(prix - (atr * atr_sl_mult), 4)

                    portfolio['capital_cash'] -= mise_brute
                    portfolio['positions'][ticker] = {
                        "quantite"      : round(quantite, 6),
                        "prix_achat"    : prix,
                        "date_achat"    : aujourd_hui,
                        "mise"          : round(mise_brute, 2),
                        "tp_cible"      : tp_cible,
                        "sl_cible"      : sl_cible,
                        "atr_lors_achat": atr,
                        "atr_tp_mult"   : atr_tp_mult,
                        "atr_sl_mult"   : atr_sl_mult
                    }
                    trade = {
                        "date"    : aujourd_hui,
                        "ticker"  : ticker,
                        "action"  : "ACHAT",
                        "prix"    : prix,
                        "quantite": round(quantite, 6),
                        "mise"    : round(mise_brute, 2),
                        "frais"   : round(frais_achat, 2),
                        "tp_cible": tp_cible,
                        "sl_cible": sl_cible,
                        "atr"     : atr
                    }
                    portfolio['historique'].append(trade)
                    trades_du_jour.append(trade)
                    action_str = "🟢 ACHETÉ"
                    detail     = f"{mise_brute:.0f}€ @ {prix:.2f} | TP:{tp_cible:.2f} | SL:{sl_cible:.2f}"
                else:
                    action_str = "⚠️ BUDGET INSUFFISANT"

        # ── EN POSITION ────────────────────────────────────────────────────
        elif position_ouverte and ticker in portfolio['positions']:
            pos       = portfolio['positions'][ticker]
            rendement = (prix - pos['prix_achat']) / pos['prix_achat']
            tp_cible  = pos.get('tp_cible', 0)
            sl_cible  = pos.get('sl_cible', 0)
            action_str = "🔵 EN POSITION"
            detail     = f"PnL:{rendement*100:+.1f}% | TP:{tp_cible:.2f} | SL:{sl_cible:.2f}"

        signal_txt = "🟢 ACHAT" if signal else "⚪ CASH"
        atr_txt    = f"{atr:.2f}" if atr > 0 else "N/A"
        print(f"{ticker:<10} {proba:<7.0%} {atr_txt:<8} {signal_txt:<12} {action_str:<20} {detail}")

    return portfolio, trades_du_jour

# ── RÉSUMÉ ────────────────────────────────────────────────────────────────────
def afficher_resume(portfolio):
    aujourd_hui   = datetime.now().strftime("%Y-%m-%d")
    valeur_totale = calculer_valeur_totale(portfolio)
    perf_totale   = (valeur_totale - portfolio['capital_depart']) / portfolio['capital_depart'] * 100

    portfolio['valeur_historique'].append({"date": aujourd_hui, "valeur": valeur_totale})

    trades_fermes   = [t for t in portfolio['historique'] if t['action'] == 'VENTE']
    nb_trades       = len(trades_fermes)
    wins            = [t for t in portfolio['historique'] if t.get('pnl', 0) > 0]
    stop_losses     = [t for t in trades_fermes if t.get('raison') == 'STOP LOSS']
    take_profits    = [t for t in trades_fermes if t.get('raison') == 'TAKE PROFIT']
    win_rate        = len(wins) / nb_trades * 100 if nb_trades > 0 else 0

    sharpe, max_dd  = calculer_metriques(portfolio)

    print("\n" + "═" * 75)
    print(f"💼 RÉSUMÉ RF Agressif (V2 ATR Hybride) — {aujourd_hui}")
    print("═" * 75)
    print(f"  Capital de départ   : {portfolio['capital_depart']:.2f} €")
    print(f"  Valeur actuelle     : {valeur_totale:.2f} €")
    print(f"  Performance totale  : {perf_totale:+.2f}%")
    print(f"  Cash disponible     : {portfolio['capital_cash']:.2f} €")
    print(f"  Positions ouvertes  : {len(portfolio['positions'])} / {MAX_POSITIONS}")
    print(f"  Trades fermés       : {nb_trades} (TP: {len(take_profits)} | SL: {len(stop_losses)})")
    print(f"  Win rate            : {win_rate:.0f}%")

    if sharpe is not None:
        interpretation = "✅ Bon" if sharpe > 1 else ("⚠️ Moyen" if sharpe > 0 else "❌ Négatif")
        print(f"  Sharpe Ratio        : {sharpe:.2f} {interpretation}")
        print(f"  Max Drawdown        : {max_dd:.1%}")

    if portfolio['positions']:
        print("\n  📂 Positions ouvertes :")
        for ticker, pos in portfolio['positions'].items():
            try:
                prix_actuel = float(yf.download(ticker, period="2d", interval="1d", progress=False)['Close'].iloc[-1])
                rendement   = (prix_actuel - pos['prix_achat']) / pos['prix_achat'] * 100
                tp_cible    = pos.get('tp_cible', 0)
                sl_cible    = pos.get('sl_cible', 0)
                print(f"    {ticker:<8} @ {pos['prix_achat']:.2f} → {rendement:+.1f}% | TP:{tp_cible:.2f} | SL:{sl_cible:.2f}")
            except:
                print(f"    {ticker:<8} @ {pos['prix_achat']:.2f}")

    print("═" * 75)
    return portfolio

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 BOT PAPER TRADING V2 - AGRESSIF (ATR HYBRIDE)")
    print(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"   Seuil IA fixe : {SEUIL_IA_FIXE:.0%} | Max pos : {MAX_POSITIONS}\n")

    faire_backup()
    
    settings = charger_settings()
    if settings is None:
        print("🛑 Master Switch OFF — arrêt du bot.")
        sys.exit(0)

    portfolio          = charger_portfolio()
    portfolio, trades  = executer_trades(portfolio, settings)
    portfolio          = afficher_resume(portfolio)
    sauvegarder_portfolio(portfolio)
    
    val_fin = calculer_valeur_totale(portfolio)

    if trades:
        lignes = []
        for t in trades:
            if t['action'] == 'ACHAT':
                lignes.append(f"🟢 ACHAT {t['ticker']} @ {t['prix']:.2f} | TP:{t.get('tp_cible',0):.2f} | SL:{t.get('sl_cible',0):.2f} | ATR:{t.get('atr',0):.2f}")
            elif t['action'] == 'VENTE':
                emoji = {"TAKE PROFIT": "✅", "STOP LOSS": "🛑", "SIGNAL IA": "🤖"}.get(t.get('raison', ''), "🔴")
                lignes.append(f"{emoji} VENTE {t['ticker']} — PnL : {t.get('pnl', 0):+.0f}€ ({t.get('raison', '')})")
        
        msg = "\n".join(lignes)
        envoyer_alerte_telegram(f"🚀 *RF Agressif ATR — Mouvements*\n\n{msg}\n\n💰 Portfolio : {val_fin:.2f}€")
    else:
        envoyer_alerte_telegram(f"😴 *RF Agressif ATR — Scan terminé*\nAucun mouvement.\n💰 Portfolio : {val_fin:.2f}€")
