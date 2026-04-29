import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant Dashboard V9.2", layout="wide", page_icon="📈")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    "AMD", "INTC", "TSM", "QCOM",
    "JPM", "V", "BAC", "GS",
    "WMT", "JNJ", "PG", "HD", "DIS",
    "BTC-USD", "ETH-USD",
    "SPY", "QQQ", "IWM", "TLT", "GLD", "XLK", "XLF",
]
CAPITAL_TOTAL     = 1000
FRAIS_TRANSACTION = 0.001
SLIPPAGE          = 0.0005
VOL_TARGET        = 0.15

# ── GITHUB — lecture des portfolios réels ─────────────────────────────────────
GITHUB_RAW = "https://raw.githubusercontent.com/jeremybaumont03/Bot_Trading/main"

BOTS = {
    "V15 Prop Desk"  : "portfolio_v15_fund.json",
    "Random Forest"  : "portfolio_v14_safe.json", 
    "Logistic Reg."  : "portfolio_lr.json",
    "Gradient Boost" : "portfolio_gb.json",
    "RF Conserv."    : "portfolio_conservative.json",
    "RF Agressif"    : "portfolio_aggressive.json",
    "Mean Reversion" : "portfolio_mr.json",
    "MR Canary"      : "portfolio_mr_canary.json",
}
PARAMS = {
    "V15 Prop Desk"  : {"Modèle": "Cross-Sectional V15", "Seuil": "Top Ranking", "Stop": "ATR Volatility", "Target": "ATR Volatility"},
    "Random Forest"  : {"Modèle": "Random Forest",   "Seuil": "0.55", "Stop": "ATR x1.5", "Target": "ATR x2.0"},
    "Logistic Reg."  : {"Modèle": "Logistic Reg.",   "Seuil": "0.55", "Stop": "ATR x1.5", "Target": "ATR x2.0"},
    "Gradient Boost" : {"Modèle": "Gradient Boost",  "Seuil": "0.55", "Stop": "ATR x1.5", "Target": "ATR x2.0"},
    "RF Conserv."    : {"Modèle": "Random Forest",   "Seuil": "0.65", "Stop": "ATR x1.5", "Target": "ATR x2.0"},
    "RF Agressif"    : {"Modèle": "Random Forest",   "Seuil": "0.45", "Stop": "ATR x1.5", "Target": "ATR x2.0"},
    "Mean Reversion" : {"Modèle": "RSI (<30) + MA20","Seuil": "< 30", "Stop": "ATR x1.5", "Target": "ATR x2.0"},
    "MR Canary"      : {"Modèle": "RSI (<45) Loose", "Seuil": "< 45", "Stop": "ATR x1.5", "Target": "ATR x2.0"},
}

COULEURS = {
    "V15 Prop Desk"  : "#FFFFFF", 
    "Random Forest"  : "#00FF88",
    "Logistic Reg."  : "#4488FF",
    "Gradient Boost" : "#FF8844",
    "RF Conserv."    : "#AAAAAA",
    "RF Agressif"    : "#FF4444",
    "Mean Reversion" : "#B266FF",
    "MR Canary"      : "#FFD700",
}

# ── MOTEUR PAR ACTIF (Simulation IA Locale) ───────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_asset_data(ticker: str):
    df = yf.download(ticker, period="8y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['MA50']       = df['Close'].rolling(50).mean()
    df['MA200']      = df['Close'].rolling(200).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    df['Mom_20j']    = df['Close'] / df['Close'].shift(20)
    df['Drawdown']   = df['Close'] / df['Close'].cummax() - 1
    df['Target']     = (df['Close'].shift(-10) / df['Close'] - 1 > 0.01).astype(int)
    df = df.dropna()

    features = ['MA50', 'MA200', 'Volatility', 'Mom_20j', 'Drawdown']
    WINDOW_TRAIN, WINDOW_TEST = 504, 126
    all_results = []

    for start in range(WINDOW_TRAIN, len(df) - WINDOW_TEST, WINDOW_TEST):
        train = df.iloc[start - WINDOW_TRAIN : start]
        test  = df.iloc[start : start + WINDOW_TEST].copy()
        if len(train) < 200 or len(test) < 10: continue
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(train[features], train['Target'])
        seuil = np.percentile(model.predict_proba(train[features])[:, 1], 70)
        test['Proba'] = model.predict_proba(test[features])[:, 1]
        test['Signal'] = ((test['Proba'] > seuil) & (test['Close'] > test['MA200']) & (test['Mom_20j'] > 1.02)).astype(int)
        all_results.append(test)

    if not all_results: return df, pd.DataFrame(), None, features, 0.55, 1.0, 1.0, 0.0, 0.0, 0, 0.0

    test_wf = pd.concat(all_results).sort_index()
    test_wf = test_wf[~test_wf.index.duplicated(keep='first')]

    daily_ret  = test_wf['Close'].pct_change().fillna(0)
    position   = test_wf['Signal'].shift(1).fillna(0)
    changes    = position.diff().abs().fillna(0)
    cout_total = changes * (FRAIS_TRANSACTION + SLIPPAGE)
    strat_net  = daily_ret * position - cout_total

    test_wf['Strat_Returns'] = (1 + strat_net).cumprod()
    test_wf['BH_Returns']    = (1 + daily_ret).cumprod()

    final_strat = test_wf['Strat_Returns'].iloc[-1]
    final_bh    = test_wf['BH_Returns'].iloc[-1]
    sharpe      = strat_net.mean() / strat_net.std() * np.sqrt(252) if strat_net.std() > 0 else 0
    cumul       = test_wf['Strat_Returns']
    max_dd      = ((cumul - cumul.cummax()) / cumul.cummax()).min()
    win_rate    = (strat_net[strat_net != 0] > 0).mean() if len(strat_net[strat_net != 0]) > 0 else 0.0

    model_live = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    model_live.fit(df[features].iloc[:-1], df['Target'].iloc[:-1])
    return df, test_wf, model_live, features, 0.55, final_strat, final_bh, sharpe, max_dd, int(changes.sum()), win_rate

# ── FETCH GITHUB (OPTIMISÉ POUR TOUT AFFICHER) ────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def charger_portfolio_github(fichier):
    try:
        r = requests.get(f"{GITHUB_RAW}/{fichier}", timeout=10)
        return json.loads(r.text) if r.status_code == 200 else None
    except: return None

@st.cache_data(ttl=300, show_spinner=False)
def charger_settings_github():
    try:
        r = requests.get(f"{GITHUB_RAW}/global_settings.json", timeout=10)
        return json.loads(r.text) if r.status_code == 200 else None
    except: return None

@st.cache_data(ttl=300, show_spinner=False)
def charger_tous_portfolios():
    data = {}
    for nom, fichier in BOTS.items():
        p = charger_portfolio_github(fichier)
        
        # ✅ OPTIMISATION : Faux portfolio si le bot n'a pas encore tradé
        if p is None:
            p = {
                "capital_depart": CAPITAL_TOTAL,
                "capital_cash": CAPITAL_TOTAL,
                "positions": {},
                "historique": [],
                "valeur_historique": []
            }
            statut = "⏳ En attente"
        else:
            statut = "✅ Actif"

        seen = {}
        for h in p.get('valeur_historique', []):
            seen[h['date']] = h['valeur']
        valeurs = list(seen.values())
        capital = p.get('capital_depart', CAPITAL_TOTAL)

        if len(valeurs) >= 2:
            rend   = pd.Series(valeurs).pct_change().dropna()
            sharpe = rend.mean() / rend.std() * np.sqrt(252) if rend.std() > 0 else 0.0
            cumul  = (1 + rend).cumprod()
            max_dd = ((cumul - cumul.cummax()) / cumul.cummax()).min()
            valeur_actuelle = valeurs[-1]
            perf   = (valeur_actuelle - capital) / capital * 100
        else:
            sharpe = None
            max_dd = None
            valeur_actuelle = p.get('capital_cash', capital)
            perf   = 0.0

        # ✅ CORRECTION 1 APPLIQUÉE : Seuls les trades fermés comptent pour le win rate
        trades_fermes = [t for t in p.get('historique', []) if t.get('action') == 'VENTE']
        wins          = [t for t in trades_fermes if t.get('pnl', 0) > 0]
        win_rate      = len(wins) / len(trades_fermes) * 100 if trades_fermes else 0.0

        data[nom] = {
            'portfolio'      : p,
            'valeur_actuelle': valeur_actuelle,
            'perf_totale'    : perf,
            'sharpe'         : round(sharpe, 3) if sharpe is not None else None,
            'max_dd'         : round(max_dd, 4) if max_dd is not None else None,
            'nb_trades'      : len(trades_fermes),
            'win_rate'       : win_rate,
            'nb_jours'       : len(valeurs),
            'seen'           : seen,
            'statut'         : statut,
            **PARAMS[nom],
        }
    return data

@st.cache_data(ttl=3600, show_spinner=False)
def get_portfolio(capital: float, cible_vol: float):
    results, prices = [], {}
    progress = st.progress(0, text="Chargement et entraînement des IA (Cette étape peut prendre 10-20 sec)…")
    for i, ticker in enumerate(TICKERS):
        progress.progress((i + 1) / len(TICKERS), text=f"Analyse IA et Backtest : {ticker} ({i+1}/{len(TICKERS)})")
        try:
            df, test_wf, model, features, seuil_live, final_strat, final_bh, sharpe, max_dd, nb_trades, win_rate = get_asset_data(ticker)
            last       = df.iloc[-1]
            proba_live = model.predict_proba(df[features].iloc[[-1]])[0][1]
            signal     = proba_live > seuil_live
            vol_ann    = float(last['Volatility']) * np.sqrt(252)
            alloc      = min(0.20, cible_vol / vol_ann) if (signal and vol_ann > 0) else 0.0
            
            prices[ticker] = df['Close'].rename(ticker)
            results.append({
                'Ticker': ticker, 'Signal': "🟢 ACHAT" if signal else "🔴 CASH",
                'Confiance IA': proba_live, 'Volatilité': vol_ann, 'Allocation': alloc,
                'Mise (€)': capital * alloc, 'Strat (x)': final_strat, 'B&H (x)': final_bh,
                'Sharpe': sharpe, 'Max DD': max_dd, 'Trades': nb_trades, 'Win Rate': win_rate,
                '_trend': float(last['Close']) > float(last['MA200']), '_mom': float(last['Mom_20j']) > 1.02, '_ia': signal, '_test_wf': test_wf, '_df': df,
            })
        except: pass
    progress.empty()
    corr_matrix = pd.concat(prices.values(), axis=1).dropna().pct_change().dropna().corr() if len(prices) >= 2 else pd.DataFrame()
    return results, corr_matrix, sum(r['Allocation'] for r in results)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Quant Dashboard V9.2 (8 Bots — Hedge Fund Level)")
st.caption("Affiche TOUS les bots, même ceux en attente de trades.")

with st.sidebar:
    st.header("⚙️ Paramètres")
    capital    = st.number_input("Capital par stratégie (€)", 100, 1_000_000, CAPITAL_TOTAL, 100)
    VOL_TARGET = st.slider("Cible de volatilité (%)", 5, 40, int(VOL_TARGET * 100), 1) / 100
    st.divider()

    settings = charger_settings_github()
    if settings:
        # ✅ CORRECTION 2 APPLIQUÉE : Lecture de allow_buying avec fallback sur master_switch_active
        switch = settings.get("allow_buying", settings.get("master_switch_active", True))
        risk   = settings.get("global_risk_multiplier", 1.0)
        atr_tp = settings.get("atr_tp_multiplier", 2.0)
        atr_sl = settings.get("atr_sl_multiplier", 1.5)
        
        st.subheader("🧠 Cerveau Central")
        st.metric("Statut", "🟢 ACHATS AUTORISÉS" if switch else "🛑 ACHATS BLOQUÉS")
        c1, c2 = st.columns(2)
        c1.metric("Risk", f"{risk}x"); c2.metric("ATR TP", f"{atr_tp}x"); c1.metric("ATR SL", f"{atr_sl}x")
        if risk < 1.0: st.warning(f"⚠️ Mode défensif ({risk}x)")
        elif risk > 1.0: st.error(f"🔥 Mode agressif ({risk}x)")
    else: st.warning("⚠️ global_settings.json non trouvé sur GitHub")
    st.divider()
    st.caption(f"Frais : {FRAIS_TRANSACTION:.1%} · Slippage : {SLIPPAGE:.2%}")

results, corr_matrix, total_alloc = get_portfolio(capital, VOL_TARGET)

# ── ONGLET MES VRAIS TRADES (Dashboard GitHub) ────────────────────────────────
st.subheader("🤖 Mes vrais trades — Comparaison des 8 bots")
st.caption("Vue à 360° : Tous les bots sont affichés, même s'ils n'ont pas encore tradé.")

if st.button("🔄 Rafraîchir depuis GitHub"):
    st.cache_data.clear()
    st.rerun()

data_bots = charger_tous_portfolios()

bots_sharpe     = {k: v for k, v in data_bots.items() if v['sharpe'] is not None}
bots_dd         = {k: v for k, v in data_bots.items() if v['max_dd'] is not None}
meilleur_sharpe = max(bots_sharpe, key=lambda k: bots_sharpe[k]['sharpe']) if bots_sharpe else "—"
meilleure_perf  = max(data_bots,   key=lambda k: data_bots[k]['perf_totale']) if data_bots else "—"
moins_risque    = max(bots_dd,     key=lambda k: bots_dd[k]['max_dd'])        if bots_dd   else "—"

b1, b2, b3, b4 = st.columns(4)
b1.metric("Meilleur Sharpe", meilleur_sharpe, f"{data_bots[meilleur_sharpe]['sharpe']:.2f}" if meilleur_sharpe != "—" else "")
b2.metric("Meilleure perf",  meilleure_perf, f"{data_bots[meilleure_perf]['perf_totale']:+.2f}%" if meilleure_perf != "—" else "")
b3.metric("Moins risqué",    moins_risque, f"Max DD {data_bots[moins_risque]['max_dd']:.1%}" if moins_risque != "—" else "")
b4.metric("Total Bots", f"{len(data_bots)} Bots Enregistrés")

rows = []
for nom, d in data_bots.items():
    rows.append({
        'Statut'     : d['statut'],
        'Bot'        : f"🏆 {nom}" if nom == meilleur_sharpe else nom,
        'Modèle'     : d['Modèle'],
        'Capital (€)': f"{d['valeur_actuelle']:.2f}",
        'Perf.'      : f"{d['perf_totale']:+.2f}%",
        'Sharpe'     : f"{d['sharpe']:.2f}" if d['sharpe'] is not None else "—",
        'Max DD'     : f"{d['max_dd']:.1%}"  if d['max_dd']  is not None else "—",
        'Trades'     : d['nb_trades'],
        'Win Rate'   : f"{d['win_rate']:.0f}%",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

fig3 = go.Figure()
for nom, d in data_bots.items():
    if len(d['seen']) < 2: continue
    fig3.add_trace(go.Scatter(
        x=list(d['seen'].keys()), y=list(d['seen'].values()),
        name=nom, line=dict(color=COULEURS.get(nom, "#FFFFFF"), width=2.5), mode='lines+markers'
    ))
fig3.add_hline(y=1000, line_dash="dash", line_color="white", opacity=0.3, annotation_text="Capital initial 1000€")
fig3.update_layout(template='plotly_dark', height=400, yaxis_title="Capital (€)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig3, use_container_width=True)

# ── POSITIONS OUVERTES ───────────────────────────────────────────────────
st.write("**Positions ouvertes en ce moment :**")
pos_trouvees   = False
current_prices = {r['Ticker']: float(r['_df']['Close'].iloc[-1]) for r in results}

for nom, d in data_bots.items():
    positions = d['portfolio'].get('positions', {})
    if not positions: continue
    pos_trouvees = True
    st.write(f"*{nom}* :")
    lignes_pos = []

    for t, p in positions.items():
        prix_achat  = p.get('prix_achat', 0)
        mise        = p.get('mise', 0)
        quantite    = p.get('quantite', mise / prix_achat if prix_achat > 0 else 0)
        prix_actuel = current_prices.get(t, prix_achat)
        pnl_pct     = (prix_actuel - prix_achat) / prix_achat if prix_achat > 0 else 0
        pnl_eur     = quantite * (prix_actuel - prix_achat)
        date_achat  = p.get('date_achat', '')
        try: duree_txt = f"{(datetime.now() - datetime.strptime(date_achat, '%Y-%m-%d')).days}j"
        except: duree_txt = "—"

        sl_cible, tp_cible, atr_val = p.get('sl_cible'), p.get('tp_cible'), p.get('atr_lors_achat', 0)

        if sl_cible is not None and isinstance(sl_cible, float) and sl_cible > 1.0:
            sl_txt      = f"{sl_cible:.2f}€ ({(sl_cible/prix_achat - 1)*100:+.1f}%)"
            tp_txt      = f"{tp_cible:.2f}€ ({(tp_cible/prix_achat - 1)*100:+.1f}%)" if tp_cible else "—"
            atr_txt     = f"{atr_val:.2f}"
            tp_distance = (tp_cible - prix_actuel) / prix_actuel * 100 if tp_cible else 0
            tp_dist_txt = f"+{tp_distance:.1f}%" if tp_distance > 0 else "⚠️ DÉPASSÉ"
        else:
            tp_ratio    = tp_cible if (tp_cible and isinstance(tp_cible, float) and tp_cible < 1) else 0.04
            sl_ratio    = -0.08
            sl_txt      = f"{prix_achat * (1 + sl_ratio):.2f}€ ({sl_ratio*100:+.1f}%)"
            tp_txt      = f"{prix_achat * (1 + tp_ratio):.2f}€ ({tp_ratio*100:+.1f}%)"
            atr_txt     = "—"
            tp_distance = tp_ratio * 100 - pnl_pct * 100
            tp_dist_txt = f"+{tp_distance:.1f}%" if tp_distance > 0 else "⚠️ DÉPASSÉ"

        lignes_pos.append({
            '📈 Ticker'     : t,
            '📅 Acheté le'  : date_achat,
            '⏱️ Durée'      : duree_txt,
            '🛒 Prix achat' : f"{prix_achat:.2f}€",
            '💲 Prix actuel': f"{prix_actuel:.2f}€",
            '💰 Mise'       : f"{mise:.2f}€",
            '💵 PnL (€)'    : f"{pnl_eur:+.2f}€",
            '📊 PnL (%)'    : f"{pnl_pct*100:+.1f}%",
            '🎯 TP cible'   : tp_txt,
            '🛑 SL cible'   : sl_txt,
            '→ TP dans'     : tp_dist_txt,
        })
    st.dataframe(pd.DataFrame(lignes_pos), hide_index=True, use_container_width=True)
    st.divider()

if not pos_trouvees: st.info("Aucune position ouverte en ce moment — tous les bots sont en cash")

# ── PIE CHART ─────────────────────────────────────────────────────────────
st.subheader("🌍 Exposition Globale du Hedge Fund")
total_cash    = sum(d['portfolio'].get('capital_cash', 0) for d in data_bots.values())
total_investi = sum(sum(p['mise'] for p in d['portfolio'].get('positions', {}).values()) for d in data_bots.values())
if total_cash > 0 or total_investi > 0:
    fig_pie = px.pie(values=[total_cash, total_investi], names=["Cash Sécurisé", "Capital Investi (Risqué)"], color_discrete_sequence=["#00FF88", "#FF4444"], hole=0.4)
    fig_pie.update_layout(template='plotly_dark', height=350, margin=dict(t=30, b=30))
    st.plotly_chart(fig_pie, use_container_width=True)

# ── HISTORIQUE ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📜 Historique des derniers trades clôturés")
tous_les_trades = []
for nom, d in data_bots.items():
    for trade in d['portfolio'].get('historique', []):
        if trade.get('action') == 'VENTE':
            tous_les_trades.append({**trade, 'Bot': nom})

if tous_les_trades:
    tous_les_trades.sort(key=lambda x: x.get('date', ''), reverse=True)
    df_trades = pd.DataFrame([{
        '🤖 Bot'        : t['Bot'],
        '📅 Date'       : t.get('date', 'N/A'),
        '📈 Actif'      : t.get('ticker', 'N/A'),
        '💰 Prix Vente' : f"{t.get('prix', 0):.2f}€",
        '💵 PnL Net'    : f"{t.get('pnl', 0):+.2f}€",
        '📊 PnL %'      : f"{t.get('pnl_pct', 0):+.1f}%",
        '🎯 Raison'     : t.get('raison', 'N/A'),
    } for t in tous_les_trades[:20]])
    st.dataframe(df_trades, use_container_width=True, hide_index=True)
else:
    st.info("Aucun trade n'a encore été clôturé par les bots.")

# ── COURBE TOTALE ─────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Performance Cumulée du Fonds Global")

all_dates = set()
for d in data_bots.values(): all_dates.update(d['seen'].keys())

if all_dates:
    df_total = pd.DataFrame(index=sorted(list(all_dates)))
    for nom, d in data_bots.items(): df_total[nom] = pd.Series(d['seen'])
    df_total          = df_total.ffill().fillna(1000)
    df_total['TOTAL'] = df_total.sum(axis=1)
    capital_initial   = 1000 * len(data_bots)

    fig_total = go.Figure()
    fig_total.add_trace(go.Scatter(x=df_total.index, y=df_total['TOTAL'], name=f"Valeur totale ({len(data_bots)} bots)", line=dict(color='#00FF88', width=4), fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'))
    fig_total.add_hline(y=capital_initial, line_dash="dash", line_color="white", opacity=0.5, annotation_text=f"Capital Initial ({capital_initial:.0f}€)")
    fig_total.update_layout(template='plotly_dark', height=450, yaxis_title="Capital Total (€)", hovermode="x unified")
    st.plotly_chart(fig_total, use_container_width=True)

    valeur_totale_actuelle = df_total['TOTAL'].iloc[-1]
    perf_globale           = (valeur_totale_actuelle - capital_initial) / capital_initial * 100
    f1, f2, f3             = st.columns(3)
    f1.metric("Capital initial", f"{capital_initial:.0f} €")
    f2.metric("Valeur actuelle", f"{valeur_totale_actuelle:.2f} €", f"{perf_globale:+.2f}%")
    f3.metric("Gain / Perte",    f"{valeur_totale_actuelle - capital_initial:+.2f} €")

# ── RECHERCHE MANUELLE ───────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Analyse détaillée manuelle (Simulateur)")
selected = st.selectbox("Choisir un actif pour voir les graphiques techniques :", TICKERS)
r = next((x for x in results if x['Ticker'] == selected), None)

if r:
    df, test_wf = r['_df'], r['_test_wf']
    tab1, tab2, tab3 = st.tabs(["📈 Prix & MA", "📊 Walk-Forward Backtest", "📉 Mean Reversion (RSI)"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Prix"))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='#FF4B4B', width=2), name="MA200"))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='#FFA500', width=1.5, dash='dot'), name="MA50"))
        fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not test_wf.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=test_wf.index, y=test_wf['Strat_Returns'], line=dict(color='#00FF88', width=2.5), name="Stratégie IA"))
            fig2.add_trace(go.Scatter(x=test_wf.index, y=test_wf['BH_Returns'], line=dict(color='#888888', width=1.5, dash='dot'), name="Buy & Hold"))
            fig2.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig2, use_container_width=True)
        else: st.info("Pas assez de données pour le backtest.")

    with tab3:
        df_mr = df.copy()
        delta = df_mr['Close'].diff()
        gain, loss = (delta.where(delta > 0, 0)).rolling(7).mean(), (-delta.where(delta < 0, 0)).rolling(7).mean()
        df_mr['RSI_7'] = 100 - (100 / (1 + gain / loss))
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_mr.index, y=df_mr['RSI_7'], line=dict(color='#B266FF', width=2)))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00FF88", annotation_text="Achat (<30)")
        fig_rsi.update_layout(template='plotly_dark', height=350, title=f"RSI 7 : {float(df_mr['RSI_7'].iloc[-1]):.1f}")
        st.plotly_chart(fig_rsi, use_container_width=True)
