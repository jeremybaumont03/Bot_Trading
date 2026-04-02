"""
BOT CANARY — entraineur_mr_canary.py
But : générer de l'activité et vérifier que toute l'infrastructure fonctionne.
Règles volontairement plus souples que le bot MR Elite.
RSI < 45 au lieu de 30 — trade presque tous les jours.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import shutil
import requests
from datetime import datetime

# ── CONFIGURATION TELEGRAM ────────────────────────────────────────────────────
TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
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

CAPITAL_DEPART = 1000.0
FRAIS          = 0.001
SLIPPAGE       = 0.0005

# ── PARAMÈTRES CANARY (volontairement souples) ────────────────────────────────
RSI_PERIOD       = 7
RSI_SEUIL_ACHAT  = 45        # Souples — achète les légers dips
TAKE_PROFIT_BASE = 0.02      # +2% — sortie rapide
TAKE_PROFIT_MAX  = 0.05      # +5% max
STOP_LOSS        = -0.08     # -8% — plus de marge
MAX_DUREE        = 7         # Time stop 7 jours
MAX_POSITIONS    = 3
MISE_PAR_TRADE   = 0.15      # 15% du capital

FICHIER        = "portfolio_mr_canary.json"
DOSSIER_BACKUP = "backups"

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
FICHIER        = os.path.join(BASE_DIR, "portfolio_mr_canary.json")
DOSSIER_BACKUP = os.path.join(BASE_DIR, "backups")

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def envoyer_alerte_telegram(message):
    if not TOKEN_TELEGRAM or not CHAT_ID_TELEGRAM:
        print("ℹ️ Telegram non configuré — alerte ignorée")
        return
    try:
        url     = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
        payload = {"chat_id": CHAT_ID_TELEGRAM, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=10)
        print("📱 Alerte Telegram envoyée")
    except Exception as e:
        print(f"⚠️ Erreur Telegram : {e}")

# ── BACKUP ────────────────────────────────────────────────────────────────────
def faire_backup():
    if not os.path.exists(FICHIER):
        return
    try:
        os.makedirs(DOSSIER_BACKUP, exist_ok=True)
        date_str   = datetime.now().strftime("%Y-%m-%d")
        nom_backup = f"portfolio_mr_canary_{date_str}.json"
        dest       = os.path.join(DOSSIER_BACKUP, nom_backup)
        if not os.path.exists(dest):
            shutil.copy2(FICHIER, dest)
            print(f"💾 Backup sauvegardé : {nom_backup}")
        else:
            print(f"ℹ️ Backup déjà à jour ({date_str}).")
    except Exception as e:
        print(f"⚠️ Erreur backup : {e}")

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
    print(f"✅ Nouveau portfolio Canary créé avec {CAPITAL_DEPART}€ virtuels")
    return portfolio

def sauvegarder_portfolio(portfolio):
    with open(FICHIER, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)

# ── RSI ───────────────────────────────────────────────────────────────────────
def calculer_rsi(series, period=14):
    delta = series.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

# ── SIGNAL ────────────────────────────────────────────────────────────────────
def calculer_signal(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 20:
            return False, 0.0, 0.0, 0.0, 1.0

        df['RSI']  = calculer_rsi(df['Close'], RSI_PERIOD)
        df['MA20'] = df['Close'].rolling(20).mean()
        df = df.dropna()

        last        = df.iloc[-1]
        rsi         = float(last['RSI'])
        prix        = float(last['Close'])
        ma20        = float(last['MA20'])
        vol         = float(df['Close'].pct_change().tail(20).std())
        distance_ma = prix / ma20

        # Canary : règles souples — RSI < 45 suffit
        signal = rsi < RSI_SEUIL_ACHAT

        return signal, round(rsi, 2), round(prix, 4), round(vol, 4), round(distance_ma, 4)

    except Exception as e:
        print(f"  ⚠️  Erreur {ticker} : {e}")
        return False, 0.0, 0.0, 0.0, 1.0

# ── VALEUR DU PORTFOLIO ───────────────────────────────────────────────────────
def calculer_valeur_totale(portfolio):
    valeur = portfolio['capital_cash']
    for ticker, pos in portfolio['positions'].items():
        try:
            prix = float(yf.download(ticker, period="2d", interval="1d", progress=False)['Close'].iloc[-1])
            valeur += pos['mise'] * (1 + (prix - pos['prix_achat']) / pos['prix_achat'])
        except:
            valeur += pos['mise']
    return round(valeur, 2)

# ── MÉTRIQUES ─────────────────────────────────────────────────────────────────
def calculer_metriques(portfolio):
    historique = portfolio.get('valeur_historique', [])
    if len(historique) < 5:
        return None, None
    valeurs    = [h['valeur'] for h in historique]
    rendements = pd.Series(valeurs).pct_change().dropna()
    
    if rendements.std() > 0:
        sharpe = rendements.mean() / rendements.std() * np.sqrt(252)
    else:
        sharpe = 0.0
        
    cumul      = (1 + rendements).cumprod()
    max_dd     = ((cumul - cumul.cummax()) / cumul.cummax()).min()
    return round(sharpe, 3), round(max_dd, 4)

# ── TRADES ────────────────────────────────────────────────────────────────────
def executer_trades(portfolio):
    aujourd_hui    = datetime.now().strftime("%Y-%m-%d")
    trades_du_jour = []

    if 'logs_journaliers' not in portfolio:
        portfolio['logs_journaliers'] = []

    print(f"\n📅 Analyse du {aujourd_hui} — Canary Bot (RSI < {RSI_SEUIL_ACHAT})")
    print(f"   Positions : {len(portfolio['positions'])} / {MAX_POSITIONS} | Tickers : {len(TICKERS)}")
    print("─" * 80)
    print(f"{'ACTIF':<10} {'RSI':<8} {'SIGNAL':<10} {'ACTION':<22} {'DÉTAIL'}")
    print("─" * 80)

    for ticker in TICKERS:
        signal, rsi, prix, vol, distance_ma = calculer_signal(ticker)
        position_ouverte = ticker in portfolio['positions']
        action_str       = "⚪ CASH"
        detail           = ""

        # ── SHADOW LOGGING — enregistre même en cash ──────────────────────
        portfolio['logs_journaliers'].append({
            "date"          : aujourd_hui,
            "ticker"        : ticker,
            "rsi"           : rsi,
            "distance_ma20" : distance_ma,
            "signal_valide" : bool(signal)
        })
        # Garde seulement les 1000 derniers logs
        portfolio['logs_journaliers'] = portfolio['logs_journaliers'][-1000:]

        # ── TP / SL / TIME STOP ───────────────────────────────────────────
        if position_ouverte:
            pos         = portfolio['positions'][ticker]
            rendement   = (prix - pos['prix_achat']) / pos['prix_achat']
            tp_cible    = pos.get('tp_cible', TAKE_PROFIT_BASE)
            duree_jours = (datetime.now() - datetime.strptime(pos['date_achat'], "%Y-%m-%d")).days

            if rendement >= tp_cible or rendement <= STOP_LOSS or duree_jours >= MAX_DUREE:
                pnl         = pos['mise'] * rendement
                frais_vente = pos['mise'] * (1 + rendement) * (FRAIS + SLIPPAGE)
                pnl_net     = pnl - frais_vente
                portfolio['capital_cash'] += pos['mise'] + pnl_net

                if rendement >= tp_cible:
                    raison, emoji = "TAKE PROFIT", "✅"
                elif rendement <= STOP_LOSS:
                    raison, emoji = "STOP LOSS", "🛑"
                else:
                    raison, emoji = "TIME STOP", "⏱️"

                trade = {
                    "date": aujourd_hui, "ticker": ticker, "action": "VENTE",
                    "raison": raison, "prix": prix, "quantite": pos.get('quantite', 0),
                    "valeur": round(pos['mise'] + pnl_net, 2), "pnl": round(pnl_net, 2),
                    "pnl_pct": round(rendement * 100, 2), "frais": round(frais_vente, 2),
                    "duree_jours": duree_jours
                }
                portfolio['historique'].append(trade)
                trades_du_jour.append(trade)
                del portfolio['positions'][ticker]
                action_str       = f"{emoji} VENDU ({raison})"
                detail           = f"PnL: {pnl_net:+.0f}€ ({rendement*100:+.1f}%)"
                position_ouverte = False

        # ── ACHAT ─────────────────────────────────────────────────────────
        if signal and not position_ouverte:
            if len(portfolio['positions']) >= MAX_POSITIONS:
                action_str = "🚫 MAX ATTEINT"
            else:
                mise        = portfolio['capital_cash'] * MISE_PAR_TRADE
                if vol > 0.04:
                    mise   *= 0.7
                frais_achat = mise * (FRAIS + SLIPPAGE)
                mise_nette  = mise - frais_achat

                if portfolio['capital_cash'] >= mise and mise_nette > 5:
                    quantite     = mise_nette / prix
                    tp_dynamique = min(max(TAKE_PROFIT_BASE, vol * 2), TAKE_PROFIT_MAX)
                    portfolio['capital_cash'] -= mise
                    portfolio['positions'][ticker] = {
                        "quantite": round(quantite, 6), "prix_achat": prix,
                        "date_achat": aujourd_hui, "mise": round(mise, 2),
                        "tp_cible": round(tp_dynamique, 3)
                    }
                    trade = {
                        "date": aujourd_hui, "ticker": ticker, "action": "ACHAT",
                        "prix": prix, "quantite": round(quantite, 6),
                        "mise": round(mise, 2), "frais": round(frais_achat, 2)
                    }
                    portfolio['historique'].append(trade)
                    trades_du_jour.append(trade)
                    action_str = "🟢 ACHETÉ"
                    detail     = f"{mise:.0f}€ @ {prix:.2f} | RSI {rsi:.1f} | TP: +{tp_dynamique*100:.1f}%"

        elif position_ouverte and ticker in portfolio['positions']:
            pos         = portfolio['positions'][ticker]
            rendement   = (prix - pos['prix_achat']) / pos['prix_achat']
            tp_cible    = pos.get('tp_cible', TAKE_PROFIT_BASE)
            duree_jours = (datetime.now() - datetime.strptime(pos['date_achat'], "%Y-%m-%d")).days
            action_str  = "🔵 EN POSITION"
            detail      = f"PnL: {rendement*100:+.1f}% | TP: +{tp_cible*100:.1f}% | Jour {duree_jours}/{MAX_DUREE}"

        signal_txt = f"🟢 {rsi:.1f}" if signal else f"⚪ {rsi:.1f}"
        print(f"{ticker:<10} {rsi:<8.1f} {signal_txt:<10} {action_str:<22} {detail}")

    return portfolio, trades_du_jour

# ── RÉSUMÉ ────────────────────────────────────────────────────────────────────
def afficher_resume(portfolio):
    aujourd_hui   = datetime.now().strftime("%Y-%m-%d")
    valeur_totale = calculer_valeur_totale(portfolio)
    perf_totale   = (valeur_totale - portfolio['capital_depart']) / portfolio['capital_depart'] * 100

    portfolio['valeur_historique'].append({"date": aujourd_hui, "valeur": valeur_totale})

    trades_fermes = [t for t in portfolio['historique'] if t['action'] == 'VENTE']
    nb_trades     = len(trades_fermes)
    wins          = [t for t in trades_fermes if t.get('pnl', 0) > 0]
    win_rate      = len(wins) / nb_trades * 100 if nb_trades > 0 else 0
    sharpe, max_dd = calculer_metriques(portfolio)

    print("\n" + "═" * 75)
    print(f"💼 RÉSUMÉ Canary Bot — {aujourd_hui}")
    print("═" * 75)
    print(f"  Capital de départ   : {portfolio['capital_depart']:.2f} €")
    print(f"  Valeur actuelle     : {valeur_totale:.2f} €")
    print(f"  Performance totale  : {perf_totale:+.2f}%")
    print(f"  Cash disponible     : {portfolio['capital_cash']:.2f} €")
    print(f"  Positions ouvertes  : {len(portfolio['positions'])} / {MAX_POSITIONS}")
    print(f"  Trades fermés       : {nb_trades} | Win rate : {win_rate:.0f}%")
    print(f"  Logs journaliers    : {len(portfolio.get('logs_journaliers', []))} entrées")
    if sharpe is not None:
        print(f"  Sharpe Ratio        : {sharpe:.2f}")
        print(f"  Max Drawdown        : {max_dd:.1%}")
    print("═" * 75)
    return portfolio

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🐦 BOT CANARY — RSI < 45 | 31 tickers")
    print(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")

    faire_backup()
    portfolio         = charger_portfolio()
    portfolio, trades = executer_trades(portfolio)
    portfolio         = afficher_resume(portfolio)
    sauvegarder_portfolio(portfolio)

    val_fin = calculer_valeur_totale(portfolio)

    if trades:
        lignes = []
        for t in trades:
            if t['action'] == 'ACHAT':
                lignes.append(f"🟢 ACHAT {t['ticker']} @ {t['prix']:.2f} — {t['mise']:.0f}€")
            elif t['action'] == 'VENTE':
                if   t.get('raison') == "TAKE PROFIT": emoji = "✅"
                elif t.get('raison') == "STOP LOSS":   emoji = "🛑"
                else:                                  emoji = "⏱️"
                lignes.append(f"{emoji} VENTE {t['ticker']} — PnL : {t.get('pnl', 0):+.0f}€")
        msg = "\n".join(lignes)
        envoyer_alerte_telegram(f"🐦 *Canary Bot — Mouvements*\n\n{msg}\n\n💰 Portfolio : {val_fin:.2f}€")
    else:
        envoyer_alerte_telegram(f"🐦 *Canary Bot — Scan terminé*\nAucun signal (RSI < {RSI_SEUIL_ACHAT})\n💰 Portfolio : {val_fin:.2f}€")

    print(f"\n✅ Sauvegardé dans '{FICHIER}'\n")