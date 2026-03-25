"""
BOT DE PAPER TRADING — V2
Améliorations vs V1 :
  - Seuil IA fixe (plus de percentile instable)
  - Stop loss à -8%
  - Maximum 3 positions simultanées
  - Sauvegarde automatique (backup quotidien)
  - Log de performance : Sharpe + Drawdown
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import shutil
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TICKERS = ["NVDA", "AAPL", "BTC-USD", "GLD", "TSLA", "MSFT", "SPY", "TLT"]
CAPITAL_DEPART   = 1000.0
FRAIS            = 0.001       # 0.1% par trade
SLIPPAGE         = 0.0005      # 0.05% slippage
VOL_TARGET       = 0.15        # cible volatilité annualisée

# ✅ SEUIL FIXE — plus de percentile qui change chaque jour
SEUIL_IA_FIXE    = 0.65        # l'IA doit être sûre à 65% minimum

# ✅ STOP LOSS — vendre automatiquement si perte > 8%
STOP_LOSS        = -0.05

# ✅ MAX POSITIONS — jamais plus de 3 actifs en même temps
MAX_POSITIONS    = 3

TARGET_HAUSSE    = 0.03        # cible : +3% en 10 jours
FICHIER          = "portfolio_conservative.json"
DOSSIER_BACKUP   = "backups"

# ── CONFIGURATION DES CHEMINS (Indispensable pour la fonction) ──
# Cela récupère le dossier où se trouve ton fichier .py actuel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FICHIER  = os.path.join(BASE_DIR, "portfolio_conservative.json")
DOSSIER_BACKUP = os.path.join(BASE_DIR, "backups")

# ── LA FONCTION DE SAUVEGARDE ──
def faire_backup():
    """
    Copie portfolio.json dans un dossier backups/ avec la date du jour.
    Cela permet de ne jamais perdre ton historique de trading.
    """
    # 1. On vérifie si le fichier portfolio existe déjà
    if not os.path.exists(FICHIER):
        return  # Si pas de fichier, on ne peut pas faire de copie

    try:
        # 2. On crée le dossier 'backups' s'il n'existe pas encore
        os.makedirs(DOSSIER_BACKUP, exist_ok=True)

        # 3. On prépare le nom du fichier avec la date (ex: portfolio_2026-03-24.json)
        date_str = datetime.now().strftime("%Y-%m-%d")
        nom_backup = f"portfolio_{date_str}.json"
        dest = os.path.join(DOSSIER_BACKUP, nom_backup)

        # 4. On ne fait la copie que si elle n'existe pas déjà pour aujourd'hui
        if not os.path.exists(dest):
            shutil.copy2(FICHIER, dest)
            print(f"💾 Backup sauvegardé avec succès : {nom_backup}")
        else:
            print(f"ℹ️ Backup déjà à jour pour aujourd'hui ({date_str}).")

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
        "valeur_historique": []
    }
    sauvegarder_portfolio(portfolio)
    print(f"✅ Nouveau portfolio créé avec {CAPITAL_DEPART}€ virtuels")
    return portfolio

def sauvegarder_portfolio(portfolio):
    with open(FICHIER, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)

# ── SIGNAL ────────────────────────────────────────────────────────────────────
def calculer_signal(ticker):
    """
    Retourne (signal_bool, proba, allocation, prix_actuel)
    Utilise un SEUIL FIXE — stable d'un jour à l'autre.
    """
    try:
        df = yf.download(ticker, period="6y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 250:
            return False, 0.0, 0.0, 0.0

        df['MA50']       = df['Close'].rolling(50).mean()
        df['MA200']      = df['Close'].rolling(200).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()
        df['Mom_20j']    = df['Close'] / df['Close'].shift(20)
        df['Drawdown']   = df['Close'] / df['Close'].cummax() - 1
        df['Target']     = (df['Close'].shift(-10) / df['Close'] - 1 > TARGET_HAUSSE).astype(int)
        df = df.dropna()

        features = ['MA50', 'MA200', 'Volatility', 'Mom_20j', 'Drawdown']
        split    = int(len(df) * 0.85)

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(df[features].iloc[:split], df['Target'].iloc[:split])

        last  = df.iloc[-1]
        proba = model.predict_proba(df[features].iloc[[-1]])[0][1]

        trend_ok = float(last['Close'])   > float(last['MA200'])
        mom_ok   = float(last['Mom_20j']) > 1.00
        # ✅ seuil fixe — cohérent d'un jour à l'autre
        ia_ok    = proba > SEUIL_IA_FIXE

        signal   = trend_ok and mom_ok and ia_ok
        vol_ann  = float(last['Volatility']) * np.sqrt(252)
        alloc    = min(0.20, VOL_TARGET / vol_ann) if (signal and vol_ann > 0) else 0.0
        prix     = float(last['Close'])

        return signal, round(proba, 4), round(alloc, 4), round(prix, 4)

    except Exception as e:
        print(f"  ⚠️  Erreur {ticker} : {e}")
        return False, 0.0, 0.0, 0.0

# ── VALEUR DU PORTFOLIO ───────────────────────────────────────────────────────
def calculer_valeur_totale(portfolio):
    valeur = portfolio['capital_cash']
    for ticker, pos in portfolio['positions'].items():
        try:
            prix = float(yf.download(ticker, period="2d", interval="1d", progress=False)['Close'].iloc[-1])
            valeur += prix * pos['quantite']
        except:
            valeur += pos['mise']
    return round(valeur, 2)

# ── MÉTRIQUES DE PERFORMANCE ──────────────────────────────────────────────────
def calculer_metriques(portfolio):
    """
    Calcule Sharpe Ratio et Max Drawdown depuis l'historique de valeur.
    Ces deux métriques te disent si le bot est réellement bon ou juste chanceux.
    """
    historique = portfolio.get('valeur_historique', [])
    if len(historique) < 5:
        return None, None

    valeurs  = [h['valeur'] for h in historique]
    series   = pd.Series(valeurs)
    rendements = series.pct_change().dropna()

    # Sharpe : rendement moyen / volatilité (annualisé sur 252 jours)
    if rendements.std() > 0:
        sharpe = rendements.mean() / rendements.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown : pire perte depuis un sommet
    cumul      = (1 + rendements).cumprod()
    max_dd     = ((cumul - cumul.cummax()) / cumul.cummax()).min()

    return round(sharpe, 3), round(max_dd, 4)

# ── TRADES ────────────────────────────────────────────────────────────────────
def executer_trades(portfolio):
    aujourd_hui   = datetime.now().strftime("%Y-%m-%d")
    trades_du_jour = []

    print(f"\n📅 Analyse du {aujourd_hui}")
    print(f"   Positions ouvertes : {len(portfolio['positions'])} / {MAX_POSITIONS}")
    print("─" * 75)
    print(f"{'ACTIF':<10} {'SIGNAL':<10} {'IA%':<7} {'ACTION':<18} {'DÉTAIL'}")
    print("─" * 75)

    for ticker in TICKERS:
        signal, proba, allocation, prix = calculer_signal(ticker)
        position_ouverte = ticker in portfolio['positions']
        action_str       = "⚪ CASH"
        detail           = ""

        # ── VÉRIFICATION STOP LOSS (priorité absolue) ──────────────────────
        if position_ouverte:
            pos     = portfolio['positions'][ticker]
            pnl_pct = (prix - pos['prix_achat']) / pos['prix_achat']

            # ✅ Stop loss : vendre si perte > STOP_LOSS (ex : -8%)
            if pnl_pct < STOP_LOSS:
                signal = False   # forcer la vente

        # ── CAS 1 : ACHAT ──────────────────────────────────────────────────
        if signal and not position_ouverte:

            # ✅ Limite du nombre de positions
            if len(portfolio['positions']) >= MAX_POSITIONS:
                action_str = "🚫 MAX ATTEINT"
                detail     = f"({MAX_POSITIONS} positions max)"

            else:
                mise        = portfolio['capital_cash'] * allocation
                frais_achat = mise * (FRAIS + SLIPPAGE)
                mise_nette  = mise - frais_achat

                if mise_nette > 5:
                    quantite = mise_nette / prix
                    portfolio['capital_cash'] -= mise
                    portfolio['positions'][ticker] = {
                        "quantite"  : round(quantite, 6),
                        "prix_achat": prix,
                        "date_achat": aujourd_hui,
                        "mise"      : round(mise, 2)
                    }
                    trade = {
                        "date"    : aujourd_hui,
                        "ticker"  : ticker,
                        "action"  : "ACHAT",
                        "prix"    : prix,
                        "quantite": round(quantite, 6),
                        "mise"    : round(mise, 2),
                        "frais"   : round(frais_achat, 2)
                    }
                    portfolio['historique'].append(trade)
                    trades_du_jour.append(trade)
                    action_str = "🟢 ACHETÉ"
                    detail     = f"{mise:.0f}€ @ {prix:.2f}"

        # ── CAS 2 : VENTE (signal rouge OU stop loss) ──────────────────────
        elif not signal and position_ouverte:
            pos          = portfolio['positions'][ticker]
            valeur_vente = pos['quantite'] * prix
            frais_vente  = valeur_vente * (FRAIS + SLIPPAGE)
            valeur_nette = valeur_vente - frais_vente
            pnl          = valeur_nette - pos['mise']
            pnl_pct_reel = (pnl / pos['mise']) * 100
            duree        = (datetime.now() - datetime.strptime(pos['date_achat'], "%Y-%m-%d")).days

            # Savoir si c'est un stop loss ou un signal normal
            raison = "STOP LOSS" if pnl_pct_reel < STOP_LOSS * 100 else "SIGNAL"

            portfolio['capital_cash'] += valeur_nette
            del portfolio['positions'][ticker]

            trade = {
                "date"       : aujourd_hui,
                "ticker"     : ticker,
                "action"     : "VENTE",
                "raison"     : raison,
                "prix"       : prix,
                "quantite"   : pos['quantite'],
                "valeur"     : round(valeur_nette, 2),
                "pnl"        : round(pnl, 2),
                "pnl_pct"    : round(pnl_pct_reel, 2),
                "frais"      : round(frais_vente, 2),
                "duree_jours": duree
            }
            portfolio['historique'].append(trade)
            trades_du_jour.append(trade)

            emoji      = "🛑" if raison == "STOP LOSS" else "🔴"
            action_str = f"{emoji} VENDU ({raison})"
            detail     = f"PnL: {pnl:+.0f}€ ({pnl_pct_reel:+.1f}%)"

        # ── CAS 3 : EN POSITION ────────────────────────────────────────────
        elif signal and position_ouverte:
            pos        = portfolio['positions'][ticker]
            pnl_latent = (prix - pos['prix_achat']) / pos['prix_achat'] * 100
            action_str = "🔵 EN POSITION"
            detail     = f"PnL latent: {pnl_latent:+.1f}% | SL: {STOP_LOSS*100:.0f}%"

        signal_txt = "🟢 ACHAT" if signal else "🔴 CASH"
        print(f"{ticker:<10} {signal_txt:<10} {proba:<7.0%} {action_str:<18} {detail}")

    return portfolio, trades_du_jour

# ── RÉSUMÉ ────────────────────────────────────────────────────────────────────
def afficher_resume(portfolio):
    aujourd_hui   = datetime.now().strftime("%Y-%m-%d")
    valeur_totale = calculer_valeur_totale(portfolio)
    perf_totale   = (valeur_totale - portfolio['capital_depart']) / portfolio['capital_depart'] * 100

    portfolio['valeur_historique'].append({
        "date"  : aujourd_hui,
        "valeur": valeur_totale
    })

    # Stats trades fermés
    trades_fermes   = [t for t in portfolio['historique'] if t['action'] == 'VENTE']
    nb_trades       = len(trades_fermes)
    wins            = [t for t in portfolio['historique'] if t.get('pnl', 0) > 0]
    stop_losses     = [t for t in trades_fermes if t.get('raison') == 'STOP LOSS']
    win_rate        = len(wins) / nb_trades * 100 if nb_trades > 0 else 0

    # Métriques avancées
    sharpe, max_dd  = calculer_metriques(portfolio)

    print("\n" + "═" * 75)
    print(f"💼 RÉSUMÉ — {aujourd_hui}")
    print("═" * 75)
    print(f"  Capital de départ   : {portfolio['capital_depart']:.2f} €")
    print(f"  Valeur actuelle     : {valeur_totale:.2f} €")
    print(f"  Performance totale  : {perf_totale:+.2f}%")
    print(f"  Cash disponible     : {portfolio['capital_cash']:.2f} €")
    print(f"  Positions ouvertes  : {len(portfolio['positions'])} / {MAX_POSITIONS}")
    print(f"  Trades fermés       : {nb_trades} (dont {len(stop_losses)} stop loss)")
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
                pnl         = (prix_actuel - pos['prix_achat']) / pos['prix_achat'] * 100
                sl_distance = pnl - (STOP_LOSS * 100)
                print(f"    {ticker:<8} @ {pos['prix_achat']:.2f} → {pnl:+.1f}% | Stop dans {sl_distance:.1f}%")
            except:
                print(f"    {ticker:<8} @ {pos['prix_achat']:.2f}")

    print("═" * 75)
    return portfolio

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 BOT PAPER TRADING V2")
    print(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"   Seuil IA fixe : {SEUIL_IA_FIXE:.0%} | Stop loss : {STOP_LOSS:.0%} | Max positions : {MAX_POSITIONS}\n")

    faire_backup()
    portfolio            = charger_portfolio()
    portfolio, trades    = executer_trades(portfolio)
    portfolio            = afficher_resume(portfolio)
    sauvegarder_portfolio(portfolio)

    print(f"\n✅ Sauvegardé dans '{FICHIER}'")
    print("   Lance ce script chaque soir\n")
