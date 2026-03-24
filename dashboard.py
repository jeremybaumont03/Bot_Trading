import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from itertools import combinations

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant Dashboard V8", layout="wide", page_icon="📈")

TICKERS           = ["NVDA", "AAPL", "BTC-USD", "GLD", "TSLA", "MSFT"]
CAPITAL_TOTAL     = 1000
FRAIS_TRANSACTION = 0.001   # 0.1% par trade
SLIPPAGE          = 0.0005  # 0.05% de slippage réaliste (actions liquides)
VOL_TARGET        = 0.15    # Cible : 15% de volatilité annualisée

# ── MOTEUR PAR ACTIF ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_asset_data(ticker: str):
    """
    Télécharge et prépare les données pour UN actif.
    Retourne le df complet + les rendements du backtest walk-forward.
    """
    df = yf.download(ticker, period="8y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ── Indicateurs ──
    df['MA50']       = df['Close'].rolling(50).mean()
    df['MA200']      = df['Close'].rolling(200).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    df['Mom_20j']    = df['Close'] / df['Close'].shift(20)
    df['Drawdown']   = df['Close'] / df['Close'].cummax() - 1
    df['Target']     = (df['Close'].shift(-10) / df['Close'] - 1 > 0.03).astype(int)
    df = df.dropna()

    features = ['MA50', 'MA200', 'Volatility', 'Mom_20j', 'Drawdown']

    # ── WALK-FORWARD ─────────────────────────────────────────────────────────
    # Principe : on avance par fenêtres de 6 mois
    # À chaque fenêtre : on réentraîne sur le passé, on teste sur les 6 prochains mois
    # → Le modèle s'adapte à chaque régime de marché
    WINDOW_TRAIN = 504   # ~2 ans de trading (252 jours/an × 2)
    WINDOW_TEST  = 126   # ~6 mois

    all_results = []

    for start in range(WINDOW_TRAIN, len(df) - WINDOW_TEST, WINDOW_TEST):
        train = df.iloc[start - WINDOW_TRAIN : start]
        test  = df.iloc[start : start + WINDOW_TEST].copy()

        if len(train) < 200 or len(test) < 10:
            continue

        # Entraînement
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        model.fit(train[features], train['Target'])

        # Seuil sur le train de cette fenêtre
        seuil = np.percentile(model.predict_proba(train[features])[:, 1], 70)

        # Prédictions sur la fenêtre test
        test['Proba']  = model.predict_proba(test[features])[:, 1]
        test['Signal'] = (
            (test['Proba']   > seuil) &
            (test['Close']   > test['MA200']) &
            (test['Mom_20j'] > 1.02)
        ).astype(int)

        all_results.append(test)

    if not all_results:
        return df, pd.DataFrame(), None, features, 0.5

    test_wf = pd.concat(all_results).sort_index()
    test_wf = test_wf[~test_wf.index.duplicated(keep='first')]

    # ── Signal & rendements (avec slippage + frais) ──
    daily_ret = test_wf['Close'].pct_change().fillna(0)
    position  = test_wf['Signal'].shift(1).fillna(0)

    # Changements de position → frais + slippage
    changes         = position.diff().abs().fillna(0)
    cout_total      = changes * (FRAIS_TRANSACTION + SLIPPAGE)
    strat_net       = daily_ret * position - cout_total

    test_wf['Strat_Returns'] = (1 + strat_net).cumprod()
    test_wf['BH_Returns']    = (1 + daily_ret).cumprod()

    # ── Métriques ──
    final_strat = test_wf['Strat_Returns'].iloc[-1]
    final_bh    = test_wf['BH_Returns'].iloc[-1]
    sharpe      = strat_net.mean() / strat_net.std() * np.sqrt(252) if strat_net.std() > 0 else 0
    cumul       = test_wf['Strat_Returns']
    max_dd      = ((cumul - cumul.cummax()) / cumul.cummax()).min()
    nb_trades   = int(changes.sum())
    win_rate    = (strat_net[strat_net != 0] > 0).mean()

    # ── Modèle final (entraîné sur tout sauf les 6 derniers mois) ──
    split       = int(len(df) * 0.85)
    model_live  = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_live.fit(df[features].iloc[:split], df['Target'].iloc[:split])
    seuil_live  = np.percentile(model_live.predict_proba(df[features].iloc[:split])[:, 1], 70)

    return df, test_wf, model_live, features, seuil_live, final_strat, final_bh, sharpe, max_dd, nb_trades, win_rate


# ── PORTFOLIO : ALLOCATION PAR VOLATILITY TARGETING ──────────────────────────
@st.cache_data(show_spinner=False)
def get_portfolio(capital: float):
    """
    Pour chaque actif :
      1. Calcule le signal live
      2. Applique le Volatility Targeting : investit plus sur les actifs calmes, moins sur les volatils
      3. Calcule la matrice de corrélation pour alerter sur les risques
    """
    results   = []
    prices    = {}

    progress = st.progress(0, text="Chargement du portfolio…")

    for i, ticker in enumerate(TICKERS):
        progress.progress((i + 1) / len(TICKERS), text=f"Analyse : {ticker}")
        try:
            out = get_asset_data(ticker)
            df, test_wf, model, features, seuil_live, final_strat, final_bh, sharpe, max_dd, nb_trades, win_rate = out

            last       = df.iloc[-1]
            proba_live = model.predict_proba(df[features].iloc[[-1]])[0][1]

            trend_ok = float(last['Close'])   > float(last['MA200'])
            mom_ok   = float(last['Mom_20j']) > 1.02
            ia_ok    = proba_live > seuil_live
            signal   = trend_ok and mom_ok and ia_ok

            vol_annualisee = float(last['Volatility']) * np.sqrt(252)

            # ── VOLATILITY TARGETING ──────────────────────────────────────
            # Principe : on alloue un % du capital tel que la contribution
            # au risque du portefeuille soit constante (VOL_TARGET)
            # alloc = VOL_TARGET / vol_annualisee (plafonné à 25%)
            if signal and vol_annualisee > 0:
                alloc = min(0.25, VOL_TARGET / vol_annualisee)
            else:
                alloc = 0.0

            prices[ticker] = df['Close'].rename(ticker)

            results.append({
                'Ticker'      : ticker,
                'Signal'      : "🟢 ACHAT" if signal else "🔴 CASH",
                'Confiance IA': proba_live,
                'Volatilité'  : vol_annualisee,
                'Allocation'  : alloc,
                'Mise (€)'    : capital * alloc,
                'Strat (x)'   : final_strat,
                'B&H (x)'     : final_bh,
                'Sharpe'      : sharpe,
                'Max DD'      : max_dd,
                'Trades'      : nb_trades,
                'Win Rate'    : win_rate,
                '_trend'      : trend_ok,
                '_mom'        : mom_ok,
                '_ia'         : ia_ok,
                '_test_wf'    : test_wf,
                '_df'         : df,
            })
        except Exception as e:
            st.warning(f"{ticker} : erreur → {e}")

    progress.empty()

    # ── MATRICE DE CORRÉLATION ──────────────────────────────────────────────
    if len(prices) >= 2:
        price_df = pd.concat(prices.values(), axis=1).dropna()
        corr_matrix = price_df.pct_change().dropna().corr()
    else:
        corr_matrix = pd.DataFrame()

    # ── Risque de concentration ──────────────────────────────────────────
    total_alloc = sum(r['Allocation'] for r in results)

    return results, corr_matrix, total_alloc


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Quant Dashboard V8")
st.caption("Walk-Forward · Volatility Targeting · Corrélations · Slippage réaliste")

with st.sidebar:
    st.header("⚙️ Paramètres")
    capital    = st.number_input("Capital total (€)", 100, 1_000_000, CAPITAL_TOTAL, 100)
    vol_target = st.slider("Cible de volatilité (%)", 5, 40, int(VOL_TARGET * 100), 1)
    VOL_TARGET = vol_target / 100
    st.divider()
    st.caption(f"Frais : {FRAIS_TRANSACTION:.1%} · Slippage : {SLIPPAGE:.2%}")
    st.caption("Walk-forward : fenêtre 2 ans / pas 6 mois")

# ── CHARGEMENT PORTFOLIO ──────────────────────────────────────────────────────
results, corr_matrix, total_alloc = get_portfolio(capital)

# ── VUE PORTFOLIO GLOBALE ─────────────────────────────────────────────────────
st.subheader("💼 Vue Portefeuille")

p1, p2, p3, p4 = st.columns(4)
actifs_actifs = [r for r in results if r['Allocation'] > 0]
total_investi = sum(r['Mise (€)'] for r in results)

p1.metric("Capital total",    f"{capital:.0f} €")
p2.metric("Capital investi",  f"{total_investi:.0f} €",
          f"{total_investi/capital:.0%} du portfolio")
p3.metric("Cash gardé",       f"{capital - total_investi:.0f} €",
          f"{1 - total_investi/capital:.0%}")
p4.metric("Actifs en signal", f"{len(actifs_actifs)} / {len(TICKERS)}")

# Alerte si trop concentré
if total_investi / capital > 0.80:
    st.warning("⚠️ **Concentration élevée** : plus de 80% du capital est investi. Vérifie les corrélations ci-dessous.")

# ── TABLEAU DES SIGNAUX ────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Signaux & Allocations")

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
**Formule :** `Allocation = VOL_TARGET / Volatilité_annualisée` (plafonnée à 25%)

Avec une cible de **{vol_target}%** :
- Un actif à 20% de vol annuelle → allocation = {vol_target}/20 = **{vol_target/20:.0%}**
- Un actif à 60% de vol annuelle (ex: BTC) → allocation = {vol_target}/60 = **{vol_target/60:.0%}**
- Un actif à 10% de vol annuelle (ex: GLD) → allocation = {vol_target}/10 = **{min(25, vol_target/10):.0%}** (plafonné à 25%)

👉 Résultat : chaque actif contribue **autant de risque** au portefeuille, peu importe sa volatilité intrinsèque.
    """)

# ── CORRÉLATIONS ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔗 Matrice de Corrélation")
st.caption("Corrélation proche de 1.0 = actifs qui bougent ensemble = risque sous-estimé si tu les détiens tous les deux")

if not corr_matrix.empty:
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='RdYlGn',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    fig_corr.update_layout(
        template='plotly_dark', height=400,
        title="Corrélations des rendements journaliers (8 ans)"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Alertes corrélation élevée
    alertes = []
    for a, b in combinations(corr_matrix.columns, 2):
        c = corr_matrix.loc[a, b]
        if c > 0.75:
            alertes.append(f"**{a} / {b}** : corrélation {c:.2f} → très liés, ton risque est concentré")
    if alertes:
        st.warning("⚠️ Corrélations élevées détectées :\n\n" + "\n\n".join(alertes))
    else:
        st.success("✅ Aucune corrélation excessive détectée (< 0.75)")

# ── DÉTAIL PAR ACTIF ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Analyse détaillée par actif")
selected = st.selectbox("Choisir un actif", TICKERS)
r = next(x for x in results if x['Ticker'] == selected)
df      = r['_df']
test_wf = r['_test_wf']

# Signal détaillé
if r['Allocation'] > 0:
    st.success(f"🟢 **SIGNAL VERT** · Allocation : {r['Allocation']:.1%} → **{r['Mise (€)']:.0f} €**")
else:
    raisons = []
    if not r['_trend']: raisons.append("🛡️ Sous MA200")
    if not r['_mom']:   raisons.append("📉 Pas de momentum")
    if not r['_ia']:    raisons.append("🧠 IA hésitante")
    st.error("🔴 **SIGNAL ROUGE — CASH**")
    st.info("  ·  ".join(raisons) if raisons else "Conditions non réunies")

tab1, tab2 = st.tabs(["📈 Prix & MA", "📊 Walk-Forward Backtest"])

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
    fig.update_layout(template='plotly_dark', height=500,
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not test_wf.empty:
        st.caption("⚠️ Walk-forward : le modèle est réentraîné tous les 6 mois — chaque point de la courbe verte est une vraie prédiction out-of-sample")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=test_wf.index, y=test_wf['Strat_Returns'],
            line=dict(color='#00FF88', width=2.5), name="Stratégie IA (frais + slippage)"))
        fig2.add_trace(go.Scatter(x=test_wf.index, y=test_wf['BH_Returns'],
            line=dict(color='#888888', width=1.5, dash='dot'), name="Buy & Hold"))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.3)
        fig2.update_layout(template='plotly_dark', height=420,
                           yaxis_title="Multiplicateur de capital")
        st.plotly_chart(fig2, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stratégie",  f"{r['Strat (x)']:.2f}x")
        m2.metric("Sharpe",     f"{r['Sharpe']:.2f}")
        m3.metric("Max DD",     f"{r['Max DD']:.0%}")
        m4.metric("Win Rate",   f"{r['Win Rate']:.0%}")
    else:
        st.info("Pas assez de données pour le walk-forward sur cet actif.")