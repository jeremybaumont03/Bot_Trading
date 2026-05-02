"""
🏦 BROKER EXECUTION V6 APEX — FINAL CORRIGÉ
Corrections :
  ✅ FX Fix : values.flatten()[-1] pour éviter l'erreur numpy.float64
  ✅ TimeInForce.GTC : Les ordres s'exécutent à l'ouverture du lendemain
  ✅ Crypto Shield conservé
  ✅ Timeout 45s conservé
  ✅ Rapport Telegram conservé
  ✅ String Notional conservé (bloque les erreurs 422 Alpaca)
  ✅ Max 10 ordres par cycle conservé
"""

import json
import glob
import os
import sys
import time
import requests
from datetime import datetime
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ── 1. CHRONOMÈTRE GLOBAL ─────────────────────────────────────────────────────
START_TIME   = time.time()
MAX_RUNTIME  = 45.0
SESSION_LOGS = []

def check_timeout():
    elapsed = time.time() - START_TIME
    if elapsed > MAX_RUNTIME:
        log_action(f"🚨 TIMEOUT FATAL: {elapsed:.1f}s. Arrêt d'urgence.")
        send_telegram_summary()
        sys.exit(1)

# ── 2. CONFIGURATION ───────────────────────────────────────────────────────────
API_KEY          = os.environ.get("ALPACA_API_KEY")
SECRET_KEY       = os.environ.get("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not API_KEY or not SECRET_KEY:
    print("🚨 FATAL ERROR: Clés Alpaca manquantes. Vérifier GitHub Secrets.")
    sys.exit(1)

LIVE_EXECUTION = True   # True = ordres réels sur Alpaca
IS_PAPER       = True   # True = compte paper (argent virtuel Alpaca)

BASE_MAX_ORDER_SIZE     = 500.0
MIN_TRADE_SIZE          = 10.0
BASE_MAX_TOTAL_EXPOSURE = 2500.0
MAX_ORDERS_PER_CYCLE    = 10

CRYPTO_BLACKLIST = {
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
    "BTCUSD",  "ETHUSD",  "SOLUSD",  "XRPUSD"
}

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=IS_PAPER)

# ── 3. LOGS & TELEGRAM ────────────────────────────────────────────────────────
def log_action(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line  = f"[{timestamp}] {message}"
    print(log_line)
    SESSION_LOGS.append(message)
    try:
        with open("execution_log.txt", "a") as f:
            f.write(log_line + "\n")
    except:
        pass

def send_telegram_summary():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Pas de clés Telegram — logs uniquement console.")
        return
    mode_txt = "🟡 PAPER (Argent virtuel Alpaca)" if IS_PAPER else "🔴 LIVE (Vrai argent)"
    exec_txt = "ACTIF" if LIVE_EXECUTION else "DRY RUN"
    log_text = "\n".join(SESSION_LOGS[-30:])
    message  = f"🏦 *HEDGE FUND V6 APEX*\n_{mode_txt} | Exec: {exec_txt}_\n\n```\n{log_text}\n```"
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        print(f"⚠️ Telegram : {e}")

# ── 4. CERVEAU CENTRAL ────────────────────────────────────────────────────────
def get_market_context():
    check_timeout()
    context = {
        "panic_mode"      : False,
        "allow_buying"    : True,
        "risk_multiplier" : 1.0,
        "regime"          : "UNKNOWN"
    }
    try:
        if os.path.exists("global_settings.json"):
            with open("global_settings.json", "r") as f:
                meta = json.load(f)
            context["panic_mode"]      = meta.get("panic_mode", False)
            context["allow_buying"]    = meta.get("allow_buying", True)
            context["risk_multiplier"] = float(meta.get("global_risk_multiplier", 1.0))
            context["regime"]          = meta.get("market_regime", "UNKNOWN")
            log_action(f"🧠 MACRO: Régime={context['regime']} | Risk={context['risk_multiplier']}x | Achats={'✅' if context['allow_buying'] else '🛑'}")
        else:
            log_action("⚠️ global_settings.json introuvable — risque 1.0x par défaut")
    except Exception as e:
        log_action(f"⚠️ Erreur Cerveau ({e}) — fallback 0.5x")
        context["risk_multiplier"] = 0.5
    return context

# ── 5. TAUX EUR/USD (✅ Fix numpy.float64) ────────────────────────────────────
def get_eurusd_rate():
    """
    ✅ Fix : .values.flatten()[-1] évite l'erreur
    'numpy.float64 object has no attribute iloc'
    """
    check_timeout()
    try:
        df = yf.download("EURUSD=X", period="2d", progress=False)

        if df.empty or "Close" not in df.columns:
            raise ValueError("Données FX vides")

        # ✅ Extraction robuste sans .iloc sur un numpy scalar
        rate = float(df["Close"].values.flatten()[-1])
        log_action(f"💱 EUR/USD : {rate:.4f}")
        return rate

    except Exception as e:
        log_action(f"⚠️ FX ERROR ({e}) — Fallback 1.08")
        return 1.08

# ── 6. LECTURE DES PORTFOLIOS (Crypto exclue à la source) ─────────────────────
def get_target_positions():
    target_allocations = {}
    for file in glob.glob("portfolio_*.json"):
        check_timeout()
        if any(x in file for x in ["backup", "tmp", "temp"]):
            continue
        try:
            with open(file, "r") as f:
                data = json.load(f)
            for ticker, pos_data in data.get("positions", {}).items():
                if ticker in CRYPTO_BLACKLIST:
                    log_action(f"⏭️ SKIP CRYPTO: {ticker} ignoré à la source")
                    continue
                mise = pos_data.get("mise", 0)
                if mise > 0:
                    target_allocations[ticker] = target_allocations.get(ticker, 0) + mise
        except Exception as e:
            log_action(f"⚠️ Erreur lecture {file} : {e}")

    log_action(f"📋 {len(target_allocations)} positions actions chargées (cryptos exclues)")
    return target_allocations

# ── 7. MOTEUR DE REBALANCING ──────────────────────────────────────────────────
def execute_trades():
    log_action(f"🚀 DÉMARRAGE V6 APEX — {'PAPER' if IS_PAPER else 'LIVE'} | {'DRY RUN' if not LIVE_EXECUTION else 'EXECUTION'}")

    market_context = get_market_context()

    if market_context["panic_mode"]:
        log_action("🚨 PANIC MODE — Exécution annulée.")
        send_telegram_summary()
        sys.exit(0)

    if market_context["risk_multiplier"] < 0.2:
        log_action("🛑 Risque trop faible (< 0.2) — Exécution annulée.")
        send_telegram_summary()
        sys.exit(0)

    active_max_order    = BASE_MAX_ORDER_SIZE     * market_context["risk_multiplier"]
    active_max_exposure = BASE_MAX_TOTAL_EXPOSURE * market_context["risk_multiplier"]

    if not market_context["allow_buying"]:
        log_action("⚠️ ACHATS BLOQUÉS — Mode liquidations uniquement.")

    fx_rate     = get_eurusd_rate()
    targets_eur = get_target_positions()
    targets_usd = {
        ticker: round(mise * fx_rate, 2)
        for ticker, mise in targets_eur.items()
    }

    total_exposure = sum(targets_usd.values())
    log_action(f"📊 Exposition cible : {total_exposure:.0f}$ (limite : {active_max_exposure:.0f}$)")

    if total_exposure > active_max_exposure:
        log_action(f"🚨 FATAL : Exposition ({total_exposure:.0f}$) > Limite ({active_max_exposure:.0f}$). Arrêt.")
        send_telegram_summary()
        return

    try:
        current_positions = trading_client.get_all_positions()
        current_holdings  = {pos.symbol: float(pos.market_value) for pos in current_positions}
        log_action(f"📂 {len(current_holdings)} positions actuelles sur Alpaca")
    except Exception as e:
        log_action(f"❌ FATAL : API Alpaca inaccessible : {e}")
        send_telegram_summary()
        return

    # ── PHASE 1 : Liquidations ────────────────────────────────────────────────
    log_action("── PHASE 1 : LIQUIDATIONS ──")
    for symbol in list(current_holdings.keys()):
        check_timeout()
        factory_symbol = (
            symbol.replace("USD", "-USD")
            if symbol in ["BTCUSD", "ETHUSD", "SOLUSD"]
            else symbol
        )
        if factory_symbol not in targets_usd:
            log_action(f"🛑 LIQUIDATION : {symbol}")
            if LIVE_EXECUTION:
                try:
                    trading_client.close_position(symbol)
                except Exception as e:
                    log_action(f"❌ ECHEC Liquidation {symbol} : {e}")
            else:
                log_action(f"[DRY RUN] Aurait liquidé {symbol}")

    # ── PHASE 2 : Ordres ──────────────────────────────────────────────────────
    log_action("── PHASE 2 : ORDRES ──")
    pending_verifications = []
    orders_sent           = 0

    for factory_symbol, target_usd in targets_usd.items():
        check_timeout()

        if orders_sent >= MAX_ORDERS_PER_CYCLE:
            log_action("⚠️ MAX ORDRES ATTEINT (10) — Anti-spam Alpaca.")
            break

        alpaca_symbol = factory_symbol
        current_usd   = current_holdings.get(alpaca_symbol, 0.0)
        difference    = target_usd - current_usd

        if abs(difference) < MIN_TRADE_SIZE:
            continue

        side = OrderSide.BUY if difference > 0 else OrderSide.SELL

        if side == OrderSide.BUY and not market_context["allow_buying"]:
            log_action(f"🚫 ACHAT ANNULÉ : {alpaca_symbol} (allow_buying = False)")
            continue

        trade_amount = round(min(abs(difference), active_max_order), 2)
        notional_str = f"{trade_amount:.2f}"  # String pour éviter les erreurs 422 Alpaca

        if abs(difference) > active_max_order:
            log_action(f"⚠️ TRIM : {alpaca_symbol} réduit à ${active_max_order:.0f}")

        if LIVE_EXECUTION:
            try:
                order = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    notional=notional_str,
                    side=side,
                    # ✅ FIX : GTC au lieu de DAY
                    # DAY = expire si marché fermé (après 21h UTC)
                    # GTC = s'exécute à l'ouverture du lendemain matin
                    time_in_force=TimeInForce.GTC
                )
                submitted = trading_client.submit_order(order_data=order)
                log_action(f"⚡ ORDER : {side.name} ${notional_str} de {alpaca_symbol} (GTC)")
                pending_verifications.append((alpaca_symbol, trade_amount, submitted.id))
                orders_sent += 1
            except Exception as e:
                log_action(f"❌ ECHEC {alpaca_symbol} : {e}")
        else:
            log_action(f"[DRY RUN] {side.name} ${notional_str} de {alpaca_symbol} (GTC)")

    # ── PHASE 3 : Vérification ────────────────────────────────────────────────
    if LIVE_EXECUTION and pending_verifications:
        log_action("── PHASE 3 : VÉRIFICATION ──")
        time.sleep(2.0)
        for symbol, amount, order_id in pending_verifications:
            check_timeout()
            try:
                status = trading_client.get_order_by_id(order_id)
                # GTC = ACCEPTED est normal après clôture — pas une erreur
                emoji = "✅" if status.status == "filled" else "⏳"
                log_action(f"{emoji} {symbol} : {status.status} (GTC — s'exécutera à l'ouverture)")
            except Exception as e:
                log_action(f"⚠️ Vérification impossible {symbol} : {e}")

    elapsed = time.time() - START_TIME
    log_action(f"🏁 CYCLE TERMINÉ EN {elapsed:.1f}s")
    send_telegram_summary()

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    execute_trades()
