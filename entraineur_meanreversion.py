"""
BOT DE PAPER TRADING — V2 MEAN REVERSION (V4 ATR HYBRIDE)
Améliorations vs V3 Élite :
  - ATR Dynamique : TP et SL calculés en fonction de la volatilité réelle de chaque actif
  - Contrôle Central : Multiplicateurs ATR lus depuis global_settings.json
  - Time Stop (10j) conservé
  - Tous les filtres V3 conservés (RSI, MA20, filtre SPY)
  - Correction sys.exit() appliquée
  - FIX PRIX 0.0 : Sécurité contre les bugs Yahoo Finance
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

# ── CONFIGURATION TELEGRAM ────────────────────────────────────────────────────
TOKEN_TELEGRAM   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID_TELEGRAM = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TICKERS        = ["NVDA", "AAPL", "BTC-USD", "GLD", "TSLA", "MSFT", "SPY", "TLT"]
CAPITAL_DEPART = 1000.0
FRAIS          = 0.001       # 0.1% par trade
SLIPPAGE       = 0.0005      # 0.05% slippage

# ── PARAMÈTRES MEAN REVERSION (ATR HYBRIDE) ───────────────────────────────────
RSI_PERIOD       = 7
RSI_SEUIL_ACHAT  = 30
ATR_PERIOD       = 14        # ✅ NOUVEAU : Période ATR standard
MAX_DUREE        = 10        # Time Stop : sort après 10 jours
MAX_POSITIONS    = 3
MISE_PAR_TRADE   = 0.20

# ── FALLBACKS (si global_settings.json absent) ────────────────────────────────
DEFAULT_ATR_TP_MULT = 2.0    # TP = prix_achat + ATR * 2.0
DEFAULT_ATR_SL_MULT = 1.5    # SL = prix_achat - ATR * 1.5

# ── CONFIGURATION DES CHEMINS ─────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
FICHIER        = os.path.join(BASE_DIR, "portfolio_mr.json")
DOSSIER_BACKUP = os.path.join(BASE_DIR, "backups")
SETTINGS_FILE  = os.path.join(BASE_DIR, "global_settings.json")

# ── LECTURE DU CERVEAU CENTRAL ────────────────────────────────────────────────
def charger_settings():
    """
    Lit global_settings.json pour récupérer les multiplicateurs ATR.
    Si le fichier est absent, utilise les valeurs par défaut (le bot ne plante pas).
    """
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)

        # Vérification du master switch
        if not settings.get("master_switch_active", True):
            print("🛑 MASTER SWITCH DÉSACTIVÉ — Bot en mode veille (aucun achat)")
            return None  # Signal d'arrêt

        atr_tp = settings.get("atr_tp_multiplier", DEFAULT_ATR_TP_MULT)
        atr_sl = settings.get("atr_sl_multiplier", DEFAULT_ATR_SL_MULT)
        risk   = settings.get("risk_multiplier", 1.0)
        print(f"🧠 Cerveau Central chargé : ATR_TP={atr_tp}x | ATR_SL={atr_sl}x | Risk={risk}x")
        return {"atr_tp": atr_tp, "atr_sl": atr_sl, "risk": risk}

    except FileNotFoundError:
        print(f"⚠️ global_settings.json introuvable — valeurs par défaut utilisées")
        return {"atr_tp": DEFAULT_ATR_TP_MULT, "atr_sl": DEFAULT_ATR_SL_MULT, "risk": 1.0}
    except Exception as e:
        print(f"⚠️ Erreur lecture settings : {e} — valeurs par défaut utilisées")
        return {"atr_tp": DEFAULT_ATR_TP_MULT, "atr_sl": DEFAULT_ATR_SL_MULT, "risk": 1.0}

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
        nom_backup = f"portfolio_mr_{date_str}.json"
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
        "valeur_historique": []
    }
    sauvegarder_portfolio(portfolio)
    print(f"✅ Nouveau portfolio Mean Reversion créé avec {CAPITAL_DEPART}€ virtuels")
    return portfolio

def sauvegarder_portfolio(portfolio):
    with open(FICHIER, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)

# ── RSI ───────────────────────────────────────────────────────────────────────
def calculer_rsi(series, period=14):
    """RSI < 30 = survente (rebond probable)"""
    delta = series.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

# ── ATR ───────────────────────────────────────────────────────────────────────
def calculer_atr(df, period=14):
    """
    Average True Range — mesure la respiration réelle de l'actif.
    Un ATR élevé = actif très volatil (Tesla, BTC).
    Un ATR faible = actif calme (GLD, TLT).
    Le TP et SL seront automatiquement ajustés en conséquence.
    """
    high_low    = df['High'] - df['Low']
    high_close  = (df['High'] - df['Close'].shift()).abs()
    low_close   = (df['Low']  - df['Close'].shift()).abs()
    true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

# ── SIGNAL MEAN REVERSION (V4 ATR HYBRIDE) ────────────────────────────────────
def calculer_signal_mr(ticker):
    """
    Retourne (signal_achat, rsi_actuel, prix_actuel, atr_actuel)
    Logique identique à V3 Élite + calcul ATR ajouté.
    """
    try:
        # Filtre marché global (SPY)
        try:
            spy = yf.download("SPY", period="6mo", interval="1d", progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy_ma50    = spy['Close'].rolling(50).mean().iloc[-1]
            marche_sain = float(spy['Close'].iloc[-1]) > float(spy_ma50)
        except:
            marche_sain = True

        # Données de l'actif
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 20:
            return False, 0.0, 0.0, 0.0

        df['RSI']  = calculer_rsi(df['Close'], RSI_PERIOD)
        df['MA20'] = df['Close'].rolling(20).mean()
        df['ATR']  = calculer_atr(df, ATR_PERIOD)  # ✅ NOUVEAU
        df = df.dropna()

        last = df.iloc[-1]
        prev = df.iloc[-2]
        rsi  = float(last['RSI'])
        prix = float(last['Close'])
        ma20 = float(last['MA20'])
        atr  = float(last['ATR'])   # ✅ NOUVEAU

        prix_remonte    = float(last['Close']) > float(prev['Close'])
        distance_ma     = prix / ma20
        cassure_validee = distance_ma < 0.97

        signal = (rsi < RSI_SEUIL_ACHAT) and (prix_remonte or rsi < 25) and marche_sain and cassure_validee

        return signal, round(rsi, 2), round(prix, 4), round(atr, 4)

    except Exception as e:
        print(f"  ⚠️  Erreur {ticker} : {e}")
        return False, 0.0, 0.0, 0.0

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
    valeurs    = [h['valeur'] for h in historique]
    rendements = pd.Series(valeurs).pct_change().dropna()
    sharpe = rendements.mean() / rendements.std() * np.sqrt(252) if rendements.std() > 0 else 0.0
    cumul  = (1 + rendements).cumprod()
    max_dd = ((cumul - cumul.cummax()) / cumul.cummax()).min()
    return round(sharpe, 3), round(max_dd, 4)

# ── TRADES ────────────────────────────────────────────────────────────────────
def executer_trades(portfolio, settings):
    aujourd_hui    = datetime.now().strftime("%Y-%m-%d")
    trades_du_jour = []

    # Extraction des multiplicateurs depuis le Cerveau Central
    atr_tp_mult = settings["atr_tp"]
    atr_sl_mult = settings["atr_sl"]
    risk_mult   = settings["risk"]

    print(f"\n📅 Analyse du {aujourd_hui} — Mean Reversion ATR Hybride (V4)")
    print(f"   Positions ouvertes : {len(portfolio['positions'])} / {MAX_POSITIONS}")
    print(f"   Multiplicateurs    : TP={atr_tp_mult}x ATR | SL={atr_sl_mult}x ATR | Risk={risk_mult}x")
    print("─" * 90)
    print(f"{'ACTIF':<10} {'RSI':<8} {'ATR':<10} {'SIGNAL':<12} {'ACTION':<22} {'DÉTAIL'}")
    print("─" * 90)

    for ticker in TICKERS:
        signal, rsi, prix, atr = calculer_signal_mr(ticker)
        
        # ✅ THE 0.0 PRICE BUG FIX (Sécurité anti-crash)
        if prix <= 0.0:
            print(f"⚠️ Yahoo Finance API Bug for {ticker} (Price is 0.0). Skipping.")
            continue
            
        position_ouverte = ticker in portfolio['positions']
        action_str       = "⚪ CASH"
        detail           = ""

        # ── VÉRIFICATION TP / SL / TIME STOP ─────────────────────────────────
        if position_ouverte:
            pos         = portfolio['positions'][ticker]
            prix_achat  = pos['prix_achat']
            tp_cible    = pos.get('tp_cible')    # Prix absolu ✅
            sl_cible    = pos.get('sl_cible')    # Prix absolu ✅
            duree_jours = (datetime.now() - datetime.strptime(pos['date_achat'], "%Y-%m-%d")).days
            rendement   = (prix - prix_achat) / prix_achat

            vendre = False
            raison = ""

            if tp_cible and prix >= tp_cible:
                vendre, raison = True, "TAKE PROFIT"
                emoji = "✅"
            elif sl_cible and prix <= sl_cible:
                vendre, raison = True, "STOP LOSS"
                emoji = "🛑"
            elif duree_jours >= MAX_DUREE:
                vendre, raison = True, "TIME STOP"
                emoji = "⏱️"

            if vendre:
                quantite     = pos.get('quantite', pos['mise'] / prix_achat)
                valeur_vente = quantite * prix
                frais_vente  = valeur_vente * (FRAIS + SLIPPAGE)
                valeur_nette = valeur_vente - frais_vente
                pnl_net      = valeur_nette - pos['mise']

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

        # ── ACHAT (RSI < 30 + filtres V3) ─────────────────────────────────────
        if signal and not position_ouverte:
            if len(portfolio['positions']) >= MAX_POSITIONS:
                action_str = "🚫 MAX ATTEINT"
                detail     = f"({MAX_POSITIONS} positions max)"
            elif atr == 0.0:
                action_str = "⚠️ ATR INDISPONIBLE"
            else:
                mise_brute  = portfolio['capital_cash'] * MISE_PAR_TRADE * risk_mult
                frais_achat = mise_brute * (FRAIS + SLIPPAGE)
                mise_nette  = mise_brute - frais_achat

                if portfolio['capital_cash'] >= mise_brute and mise_nette > 5:
                    quantite = mise_nette / prix

                    # ✅ ATR HYBRIDE : TP et SL en prix absolus
                    tp_cible = round(prix + (atr * atr_tp_mult), 4)
                    sl_cible = round(prix - (atr * atr_sl_mult), 4)

                    portfolio['capital_cash'] -= mise_brute
                    portfolio['positions'][ticker] = {
                        "quantite"      : round(quantite, 6),
                        "prix_achat"    : prix,
                        "date_achat"    : aujourd_hui,
                        "mise"          : round(mise_brute, 2),
                        "tp_cible"      : tp_cible,   # Prix absolu
                        "sl_cible"      : sl_cible,   # Prix absolu
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
                    detail     = f"{mise_brute:.0f}€ @ {prix:.2f} | TP:{tp_cible:.2f} | SL:{sl_cible:.2f} | ATR:{atr:.2f}"
                else:
                    action_str = "⚠️ CASH INSUFFISANT"

        # ── EN POSITION ────────────────────────────────────────────────────────
        elif position_ouverte and ticker in portfolio['positions']:
            pos         = portfolio['positions'][ticker]
            rendement   = (prix - pos['prix_achat']) / pos['prix_achat']
            tp_cible    = pos.get('tp_cible', 0)
            sl_cible    = pos.get('sl_cible', 0)
            duree_jours = (datetime.now() - datetime.strptime(pos['date_achat'], "%Y-%m-%d")).days
            action_str  = "🔵 EN POSITION"
            detail      = f"PnL:{rendement*100:+.1f}% | TP:{tp_cible:.2f} | SL:{sl_cible:.2f} | Jours:{duree_jours}/{MAX_DUREE}"

        signal_txt = f"🟢 RSI {rsi:.1f}" if signal else f"⚪ RSI {rsi:.1f}"
        atr_txt    = f"{atr:.2f}" if atr > 0 else "N/A"
        print(f"{ticker:<10} {rsi:<8.1f} {atr_txt:<10} {signal_txt:<12} {action_str:<22} {detail}")

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
    stop_losses   = [t for t in trades_fermes if t.get('raison') == 'STOP LOSS']
    take_profits  = [t for t in trades_fermes if t.get('raison') == 'TAKE PROFIT']
    win_rate      = len(wins) / nb_trades * 100 if nb_trades > 0 else 0

    sharpe, max_dd = calculer_metriques(portfolio)

    print("\n" + "═" * 75)
    print(f"💼 RÉSUMÉ Mean Reversion ATR Hybride (V4) — {aujourd_hui}")
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
                atr_achat   = pos.get('atr_lors_achat', 0)
                print(f"    {ticker:<8} @ {pos['prix_achat']:.2f} → {rendement:+.1f}% | TP:{tp_cible:.2f} | SL:{sl_cible:.2f} | ATR:{atr_achat:.2f}")
            except:
                print(f"    {ticker:<8} @ {pos['prix_achat']:.2f}")

    print("═" * 75)
    return portfolio

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 BOT MEAN REVERSION V4 (ATR HYBRIDE)")
    print(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"   RSI {RSI_PERIOD}j < {RSI_SEUIL_ACHAT} | ATR {ATR_PERIOD}j | Time Stop {MAX_DUREE}j\n")

    faire_backup()

    settings = charger_settings()
    if settings is None:
        print("🛑 Master Switch OFF — arrêt du bot.")
        sys.exit(0) 

    portfolio         = charger_portfolio()
    portfolio, trades = executer_trades(portfolio, settings)
    val_fin           = calculer_valeur_totale(portfolio)
    portfolio         = afficher_resume(portfolio)
    sauvegarder_portfolio(portfolio)

    if trades:
        lignes = []
        for t in trades:
            if t['action'] == 'ACHAT':
                lignes.append(f"🟢 ACHAT {t['ticker']} @ {t['prix']:.2f} | TP:{t.get('tp_cible',0):.2f} | SL:{t.get('sl_cible',0):.2f} | ATR:{t.get('atr',0):.2f}")
            elif t['action'] == 'VENTE':
                emoji = {"TAKE PROFIT": "✅", "STOP LOSS": "🛑", "TIME STOP": "⏱️"}.get(t.get('raison', ''), "🔴")
                lignes.append(f"{emoji} VENTE {t['ticker']} — PnL : {t.get('pnl', 0):+.0f}€ ({t.get('raison', '')})")
        msg = "\n".join(lignes)
        envoyer_alerte_telegram(f"🎯 *Mean Reversion V4 ATR — Mouvements*\n\n{msg}\n\n💰 Portfolio : {val_fin:.2f}€")
    else:
        envoyer_alerte_telegram(f"😴 *Mean Reversion V4 — Scan terminé*\nAucun signal optimal.\n💰 Portfolio : {val_fin:.2f}€")
