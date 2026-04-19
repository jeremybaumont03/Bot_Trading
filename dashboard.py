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
st.set_page_config(page_title="Quant Dashboard V9.1", layout="wide", page_icon="📈")

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

# ✅ INTÉGRATION DE LA V15 PROP DESK
BOTS = {
    "V15 Prop Desk"  : "portfolio_v15_fund.json",
    "Random Forest"  : "portfolio.json",
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['MA50']       = df['Close'].rolling(50).mean()
    df['MA200']      = df['Close'].rolling(200).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    df['Mom_20j']    = df['Close'] / df['Close'].shift(20)
    df['Drawdown']   = df['Close'] / df['Close'].cummax() - 1
    df['Target']     = (df['Close'].shift(-10) / df['Close'] - 1 > 0.01).astype(int)
    df = df.dropna()

    features     = ['MA50', 'MA200', 'Volatility', 'Mom_20j', 'Drawdown']
    WINDOW_TRAIN = 504
    WINDOW_TEST  = 126
    all_results  = []

    for start in range(WINDOW_TRAIN, len(df) - WINDOW_TEST, WINDOW_TEST):
        train = df.iloc[start - WINDOW_TRAIN : start]
        test  = df.iloc[start : start + WINDOW_TEST].copy()
        if len(train) < 200 or len(test) < 10:
            continue
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(train[features], train['Target'])
        seuil         = np.percentile(model.predict_proba(train[features])[:, 1], 70)
        test['Proba'] = model.predict_proba(test[features])[:, 1]
        test['Signal'] = (
            (test['Proba']   > seuil) &
            (test['Close']   > test['MA200']) &
            (test['Mom_20j'] > 1.02)
        ).astype(int)
        all_results.append(test)

    if not all_results:
        return df, pd.DataFrame(), None, features, 0.55, 1.0, 1.0, 0.0, 0.0, 0, 0.0

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
    nb_trades   = int(changes.sum())
    win_rate    = (strat_net[strat_net != 0] > 0).mean() if len(strat_net[strat_net != 0]) > 0 else 0.0

    model_live = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    model_live.fit(df[features].iloc[:-1], df['Target'].iloc[:-1])
    seuil_live = 0.55

    return df, test_wf, model_live, features, seuil_live, final_strat, final_bh, sharpe, max_dd, nb_trades, win_rate

# ── FETCH GITHUB ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def charger_portfolio_github(fichier):
    try:
        r = requests.get(f"{GITHUB_RAW}/{fichier}", timeout=10)
        return json.loads(r.text) if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def charger_settings_github():
    try:
        r = requests.get(f"{GITHUB_RAW}/global_settings.json", timeout=10)
        return json.loads(r.text) if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def charger_tous_portfolios():
    data = {}
    for nom, fichier in BOTS.items():
        p = charger_portfolio_github(fichier)
        if p is None:
            continue
        seen = {}
        for h in p.get('valeur_historique', []):
            seen[h['date']] = h['valeur']
        valeurs = list(seen.values())
        capital = p['capital_depart']

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

        trades_fermes = [t for t in p.get('historique', []) if t['action'] == 'VENTE']
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
            **PARAMS[nom],
        }
    return data

# ── PORTFOLIO SIMULATION LOCAL ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_portfolio(capital: float, cible_vol: float):
    results = []
    prices  = {}
    progress = st.progress(0, text="Chargement et entraînement des IA (Cette étape peut prendre 10-20 sec)…")
    for i, ticker in enumerate(TICKERS):
        progress.progress((i + 1) / len(TICKERS), text=f"Analyse IA et Backtest : {ticker} ({i+1}/{len(TICKERS)})")
        try:
            out = get_asset_data(ticker)
            df, test_wf, model, features, seuil_live, final_strat, final_bh, sharpe, max_dd, nb_trades, win_rate = out
            last       = df.iloc[-1]
            proba_live = model.predict_proba(df[features].iloc[[-1]])[0][1]
            trend_ok   = float(last['Close'])   > float(last['MA200'])
            mom_ok     = float(last['Mom_20j']) > 1.02
            ia_ok      = proba_live > seuil_live
            signal     = ia_ok
            vol_ann    = float(last['Volatility']) * np.sqrt(252)
            
            alloc      = min(0.20, cible_vol / vol_ann) if (signal and vol_ann > 0) else 0.0
            
            prices[ticker] = df['Close'].rename(ticker)
            results.append({
                'Ticker': ticker, 'Signal': "🟢 ACHAT" if signal else "🔴 CASH",
                'Confiance IA': proba_live, 'Volatilité': vol_ann, 'Allocation': alloc,
                'Mise (€)': capital * alloc, 'Strat (x)': final_strat, 'B&H (x)': final_bh,
                'Sharpe': sharpe, 'Max DD': max_dd, 'Trades': nb_trades, 'Win Rate': win_rate,
                '_trend': trend_ok, '_mom': mom_ok, '_ia': ia_ok, '_test_wf': test_wf, '_df': df,
            })
        except Exception:
            pass
    progress.empty()
    if len(prices) >= 2:
        price_df    = pd.concat(prices.values(), axis=1).dropna()
        corr_matrix = price_df.pct_change().dropna().corr()
    else:
        corr_matrix = pd.DataFrame()
    total_alloc = sum(r['Allocation'] for r in results)
    return results, corr_matrix, total_alloc

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Quant Dashboard V9.1 (8 Bots — Hedge Fund Level)")
st.caption("Walk-Forward · ATR Dynamique · Volatility Targeting · Corrélations · Slippage réaliste")

with st.sidebar:
    st.header("⚙️ Paramètres")
    capital    = st.number_input("Capital par stratégie (€)", 100, 1_000_000, CAPITAL_TOTAL, 100)
    vol_target = st.slider("Cible de volatilité (%)", 5, 40, int(VOL_TARGET * 100), 1)
    VOL_TARGET = vol_target / 100
    st.divider()

    # ── Cerveau Central ───────────────────────────────────────────────────────
    settings = charger_settings_github()
    if settings:
        switch = settings.get("master_switch_active", True)
        risk   = settings.get("global_risk_multiplier", 1.0)
        atr_tp = settings.get("atr_tp_multiplier", 2.0)
        atr_sl = settings.get("atr_sl_multiplier", 1.5)
        st.subheader("🧠 Cerveau Central")
        st.metric("Statut", "🟢 ACTIF" if switch else "🔴 ARRÊTÉ")
        c1, c2 = st.columns(2)
        c1.metric("Risk",   f"{risk}x")
        c2.metric("ATR TP", f"{atr_tp}x")
        c1.metric("ATR SL", f"{atr_sl}x")
        if risk < 1.0:
            st.warning(f"⚠️ Mode défensif ({risk}x)")
        elif risk > 1.0:
            st.error(f"🔥 Mode agressif ({risk}x)")
        st.divider()
    else:
        st.warning("⚠️ global_settings.json non trouvé sur GitHub")
        st.divider()

    st.caption(f"Frais : {FRAIS_TRANSACTION:.1%} · Slippage : {SLIPPAGE:.2%}")
    st.caption("Walk-forward : fenêtre 2 ans / pas 6 mois")

results, corr_matrix, total_alloc = get_portfolio(capital, VOL_TARGET)

# ── VUE PORTFOLIO GLOBALE (Simulé) ────────────────────────────────────────────
st.subheader("💼 Vue Portefeuille Modèle (Simulation Locale)")
p1, p2, p3, p4 = st.columns(4)
actifs_actifs  = [r for r in results if r['Allocation'] > 0]
total_investi  = sum(r['Mise (€)'] for r in results)
p1.metric("Capital par stratégie", f"{capital:.0f} €")
p2.metric("Capital investi",       f"{total_investi:.0f} €", f"{total_investi/capital:.0%} investi")
p3.metric("Cash gardé",            f"{capital - total_investi:.0f} €", f"{1 - total_investi/capital:.0%}")
p4.metric("Actifs en signal",      f"{len(actifs_actifs)} / {len(TICKERS)}")

if capital > 0 and total_investi / capital > 0.80:
    st.warning("⚠️ **Concentration élevée** : plus de 80% du capital est investi.")

st.divider()
st.subheader("📋 Signaux & Allocations (Simulation ML Locale)")
display_df = pd.DataFrame([{
    'Actif'        : r['Ticker'],
    'Signal'       : r['Signal'],
    'Confiance IA' : f"{r['Confiance IA']:.0%}",
    'Volatilité'   : f"{r['Volatilité']:.0%}",
    'Allocation'   : f"{r['Allocation']:.1%}",
    'Mise (€)'     : f"{r['Mise (€)']:.0f} €",
    'Strat (x)'    : f"{r['Strat (x)']:.2f}x",
    'B&H (x)'      : f"{r['B&H (x)']:.2f}x",
    'Sharpe'       : f"{r['Sharpe']:.2f}",
    'Max DD'       : f"{r['Max DD']:.0%}",
    'Win Rate'     : f"{r['Win Rate']:.0%}",
} for r in results])
st.dataframe(display_df, use_container_width=True, hide_index=True)

with st.expander("💡 Comprendre le Volatility Targeting"):
    st.markdown(f"""
**Formule :** `Allocation = VOL_TARGET / Volatilité_annualisée` (plafonnée à 20%)
Avec une cible de **{vol_target}%** :
- Un actif à 20% de vol annuelle → allocation = {vol_target}/20 = **{vol_target/20:.0%}**
- Un actif à 60% de vol annuelle (ex: BTC) → allocation = {vol_target}/60 = **{vol_target/60:.0%}**
- Un actif à 10% de vol annuelle (ex: GLD) → allocation = {vol_target}/10 = **{min(20, vol_target/10):.0%}** (plafonné à 20%)
    """)

# ── CORRÉLATIONS ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔗 Matrice de Corrélation")
if not corr_matrix.empty:
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns.tolist(), y=corr_matrix.index.tolist(),
        colorscale='RdYlGn', zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2), texttemplate="%{text}", showscale=True
    ))
    fig_corr.update_layout(template='plotly_dark', height=400,
                           title="Corrélations des rendements journaliers (8 ans)")
    st.plotly_chart(fig_corr, use_container_width=True)

    alertes = []
    for a, b in combinations(corr_matrix.columns, 2):
        c = corr_matrix.loc[a, b]
        if c > 0.75:
            alertes.append(f"**{a} / {b}** : corrélation {c:.2f}")

    if alertes:
        with st.expander(f"⚠️ {len(alertes)} Paires d'actifs fortement corrélés (> 0.75)", expanded=False):
            st.markdown("Si tu achètes ces actifs en même temps, tu multiplies ton risque :")
            for alerte in alertes:
                st.markdown(f"- {alerte}")
    else:
        st.success("✅ Aucune corrélation excessive détectée (< 0.75)")

# ── DÉTAIL PAR ACTIF ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Analyse détaillée par actif")
selected = st.selectbox("Choisir un actif", TICKERS)

r = next((x for x in results if x['Ticker'] == selected), None)

if r is None:
    st.error(f"⚠️ Données indisponibles pour **{selected}** aujourd'hui.")
else:
    df      = r['_df']
    test_wf = r['_test_wf']

    if r['Allocation'] > 0:
        st.success(f"🟢 **SIGNAL VERT** · Allocation : {r['Allocation']:.1%} → **{r['Mise (€)']:.0f} €**")
    else:
        raisons = []
        if not r['_ia']: raisons.append("🧠 IA (Proba < 55%)")
        st.error("🔴 **SIGNAL ROUGE — CASH**")
        st.info("  ·  ".join(raisons) if raisons else "Conditions non réunies")

    tab1, tab2, tab3 = st.tabs(["📈 Prix & MA", "📊 Walk-Forward Backtest", "📉 Mean Reversion (RSI)"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name="Prix"
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'],
            line=dict(color='#FF4B4B', width=2), name="MA200"))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'],
            line=dict(color='#FFA500', width=1.5, dash='dot'), name="MA50"))
        fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not test_wf.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=test_wf.index, y=test_wf['Strat_Returns'],
                line=dict(color='#00FF88', width=2.5), name="Stratégie IA"))
            fig2.add_trace(go.Scatter(x=test_wf.index, y=test_wf['BH_Returns'],
                line=dict(color='#888888', width=1.5, dash='dot'), name="Buy & Hold"))
            fig2.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.3)
            fig2.update_layout(template='plotly_dark', height=420, yaxis_title="Multiplicateur de capital")
            st.plotly_chart(fig2, use_container_width=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Stratégie", f"{r['Strat (x)']:.2f}x")
            m2.metric("Sharpe",    f"{r['Sharpe']:.2f}")
            m3.metric("Max DD",    f"{r['Max DD']:.0%}")
            m4.metric("Win Rate",  f"{r['Win Rate']:.0%}")
        else:
            st.info("Pas assez de données pour le walk-forward sur cet actif.")

    with tab3:
        df_mr = df.copy()
        delta = df_mr['Close'].diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs    = gain / loss
        df_mr['RSI_7'] = 100 - (100 / (1 + rs))
        last_rsi = float(df_mr['RSI_7'].iloc[-1])

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_mr.index, y=df_mr['RSI_7'], name="RSI (7j)", line=dict(color='#B266FF', width=2)))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00FF88", annotation_text="Zone Achat (<30)")
        fig_rsi.add_hline(y=45, line_dash="dash", line_color="#FFD700", annotation_text="Zone Achat Canary (<45)")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#FF4444", annotation_text="Zone Vente (>70)")
        fig_rsi.update_layout(template='plotly_dark', height=350, yaxis_title="Valeur RSI", title=f"RSI 7 Actuel : {last_rsi:.1f}")
        st.plotly_chart(fig_rsi, use_container_width=True)

# ── ONGLET MES VRAIS TRADES (Dashboard GitHub) ────────────────────────────────
st.divider()
st.subheader("🤖 Mes vrais trades — Comparaison des 8 bots")
st.caption("Données lues en temps réel depuis ton dépôt GitHub")

if st.button("🔄 Rafraîchir depuis GitHub"):
    st.cache_data.clear()
    st.rerun()

data_bots = charger_tous_portfolios()

if not data_bots:
    st.warning("Aucun portfolio trouvé sur GitHub. Tes bots n'ont pas encore tourné.")
else:
    st.success(f"✅ {len(data_bots)} bots chargés depuis GitHub")

    bots_sharpe     = {k: v for k, v in data_bots.items() if v['sharpe'] is not None}
    bots_dd         = {k: v for k, v in data_bots.items() if v['max_dd'] is not None}
    meilleur_sharpe = max(bots_sharpe, key=lambda k: bots_sharpe[k]['sharpe']) if bots_sharpe else "—"
    meilleure_perf  = max(data_bots,   key=lambda k: data_bots[k]['perf_totale']) if data_bots else "—"
    moins_risque    = max(bots_dd,     key=lambda k: bots_dd[k]['max_dd'])        if bots_dd   else "—"

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Meilleur Sharpe", meilleur_sharpe, f"{data_bots[meilleur_sharpe]['sharpe']:.2f}" if meilleur_sharpe != "—" else "")
    b2.metric("Meilleure perf",  meilleure_perf, f"{data_bots[meilleure_perf]['perf_totale']:+.2f}%" if meilleure_perf != "—" else "")
    b3.metric("Moins risqué",    moins_risque, f"Max DD {data_bots[moins_risque]['max_dd']:.1%}" if moins_risque != "—" else "")
    jours_donnees = max((v['nb_jours'] for v in data_bots.values()), default=0)
    b4.metric("Jours de données", f"{jours_donnees} jours")

    rows = []
    for nom, d in data_bots.items():
        rows.append({
            'Bot'        : f"🏆 {nom}" if nom == meilleur_sharpe else nom,
            'Modèle'     : d['Modèle'],
            'Seuil'      : d['Seuil'],
            'Capital (€)': f"{d['valeur_actuelle']:.2f}",
            'Perf.'      : f"{d['perf_totale']:+.2f}%",
            'Sharpe'     : f"{d['sharpe']:.2f}" if d['sharpe'] is not None else "—",
            'Max DD'     : f"{d['max_dd']:.1%}"  if d['max_dd']  is not None else "—",
            'Trades'     : d['nb_trades'],
            'Win Rate'   : f"{d['win_rate']:.0f}%",
            'Jours'      : d['nb_jours'],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    fig3 = go.Figure()
    for nom, d in data_bots.items():
        if len(d['seen']) < 2: continue
        fig3.add_trace(go.Scatter(
            x=list(d['seen'].keys()), y=list(d['seen'].values()),
            name=nom, line=dict(color=COULEURS.get(nom, "#FFFFFF"), width=2.5),
            mode='lines+markers', marker=dict(size=5),
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
            try:
                duree     = (datetime.now() - datetime.strptime(date_achat, "%Y-%m-%d")).days
                duree_txt = f"{duree}j"
            except:
                duree_txt = "—"

            sl_cible = p.get('sl_cible')
            tp_cible = p.get('tp_cible')
            atr_val  = p.get('atr_lors_achat', 0)

            if sl_cible is not None and isinstance(sl_cible, float) and sl_cible > 1.0:
                sl_txt      = f"{sl_cible:.2f}€ ({(sl_cible/prix_achat - 1)*100:+.1f}%)"
                tp_txt      = f"{tp_cible:.2f}€ ({(tp_cible/prix_achat - 1)*100:+.1f}%)" if tp_cible else "—"
                atr_txt     = f"{atr_val:.2f}"
                sl_distance = (prix_actuel - sl_cible) / prix_actuel * 100
                tp_distance = (tp_cible - prix_actuel) / prix_actuel * 100 if tp_cible else 0
                sl_dist_txt = f"{sl_distance:.1f}%"
                tp_dist_txt = f"+{tp_distance:.1f}%" if tp_distance > 0 else "⚠️ DÉPASSÉ"
            else:
                tp_ratio    = tp_cible if (tp_cible and isinstance(tp_cible, float) and tp_cible < 1) else 0.04
                sl_ratio    = -0.08
                sl_abs      = prix_achat * (1 + sl_ratio)
                tp_abs      = prix_achat * (1 + tp_ratio)
                sl_txt      = f"{sl_abs:.2f}€ ({sl_ratio*100:+.1f}%)"
                tp_txt      = f"{tp_abs:.2f}€ ({tp_ratio*100:+.1f}%)"
                atr_txt     = "—"
                sl_distance = pnl_pct * 100 - sl_ratio * 100
                tp_distance = tp_ratio * 100 - pnl_pct * 100
                sl_dist_txt = f"{sl_distance:.1f}%"
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
                '→ SL dans'     : sl_dist_txt,
            })

        st.dataframe(pd.DataFrame(lignes_pos), hide_index=True, use_container_width=True)

        for t, p in positions.items():
            sl_cible = p.get('sl_cible')
            tp_cible = p.get('tp_cible')
            if not (sl_cible and tp_cible and isinstance(sl_cible, float) and sl_cible > 1.0): continue
            prix_actuel  = current_prices.get(t, p.get('prix_achat', 0))
            plage_totale = tp_cible - sl_cible
            if plage_totale <= 0: continue
            progression = max(0.0, min(1.0, (prix_actuel - sl_cible) / plage_totale))
            col_a, col_b = st.columns([1, 5])
            col_a.write(f"**{t}**")
            col_b.progress(progression, text=f"SL {sl_cible:.2f}€ ←→ {prix_actuel:.2f}€ ←→ TP {tp_cible:.2f}€")
        st.divider()

    if not pos_trouvees:
        st.info("Aucune position ouverte en ce moment — tous les bots sont en cash")

    # ── PIE CHART ─────────────────────────────────────────────────────────────
    st.subheader("🌍 Exposition Globale du Hedge Fund")
    total_cash    = sum(d['portfolio'].get('capital_cash', 0) for d in data_bots.values())
    total_investi = sum(sum(p['mise'] for p in d['portfolio'].get('positions', {}).values()) for d in data_bots.values())
    if total_cash > 0 or total_investi > 0:
        fig_pie = px.pie(
            values=[total_cash, total_investi],
            names=["Cash Sécurisé", "Capital Investi (Risqué)"],
            color_discrete_sequence=["#00FF88", "#FF4444"],
            hole=0.4
        )
        fig_pie.update_layout(template='plotly_dark', height=350, margin=dict(t=30, b=30))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── HISTORIQUE ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📜 Historique des derniers trades clôturés")
    tous_les_trades = []
    for nom, d in data_bots.items():
        for trade in d['portfolio'].get('historique', []):
            if trade['action'] == 'VENTE':
                trade_copy        = trade.copy()
                trade_copy['Bot'] = nom
                tous_les_trades.append(trade_copy)

    if tous_les_trades:
        tous_les_trades.sort(key=lambda x: x['date'], reverse=True)
        df_trades = pd.DataFrame([{
            '🤖 Bot'        : t['Bot'],
            '📅 Date'       : t['date'],
            '📈 Actif'      : t['ticker'],
            '💰 Prix Vente' : f"{t['prix']:.2f}€",
            '💵 PnL Net'    : f"{t.get('pnl', 0):+.2f}€",
            '📊 PnL %'      : f"{t.get('pnl_pct', 0):+.1f}%",
            '🎯 Raison'     : t.get('raison', 'N/A'),
            '⏱️ Durée'      : f"{t.get('duree_jours', 0)}j",
        } for t in tous_les_trades[:20]])
        st.dataframe(df_trades, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun trade n'a encore été clôturé par les bots.")

    # ── COURBE TOTALE ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Performance Cumulée du Fonds Global")
    st.caption("Cette courbe additionne la valeur de tes portefeuilles pour voir ta progression totale.")

    all_dates = set()
    for d in data_bots.values():
        all_dates.update(d['seen'].keys())

    if all_dates:
        sorted_dates = sorted(list(all_dates))
        df_total     = pd.DataFrame(index=sorted_dates)
        for nom, d in data_bots.items():
            df_total[nom] = pd.Series(d['seen'])
        df_total          = df_total.ffill().fillna(1000)
        df_total['TOTAL'] = df_total.sum(axis=1)
        capital_initial   = 1000 * len(data_bots)

        fig_total = go.Figure()
        fig_total.add_trace(go.Scatter(
            x=df_total.index, y=df_total['TOTAL'],
            name=f"Valeur totale ({len(data_bots)} bots)",
            line=dict(color='#00FF88', width=4),
            fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'
        ))
        fig_total.add_hline(y=capital_initial, line_dash="dash", line_color="white", opacity=0.5, annotation_text=f"Capital Initial Global ({capital_initial:.0f}€)")
        fig_total.update_layout(template='plotly_dark', height=450, yaxis_title="Capital Total (€)", hovermode="x unified")
        st.plotly_chart(fig_total, use_container_width=True)

        valeur_totale_actuelle = df_total['TOTAL'].iloc[-1]
        perf_globale           = (valeur_totale_actuelle - capital_initial) / capital_initial * 100
        f1, f2, f3             = st.columns(3)
        f1.metric("Capital initial global",  f"{capital_initial:.0f} €")
        f2.metric("Valeur actuelle globale", f"{valeur_totale_actuelle:.2f} €", f"{perf_globale:+.2f}%")
        f3.metric("Gain / Perte total",      f"{valeur_totale_actuelle - capital_initial:+.2f} €")
