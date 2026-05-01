"""
🏦 BROKER EXECUTION V6 APEX — CLOUD FACTORY EDITION
Features:
  ✅ Live Execution = True (Will place orders on Alpaca Paper account)
  ✅ Paper = True (Safely isolated from real bank funds)
  ✅ Crypto Shield (Ignores BTC/ETH to prevent cash drag)
  ✅ 45s Hard Timeout (Protects GitHub Actions quotas)
  ✅ Telegram Telemetry (Sends full summary to your phone)
  ✅ PRO FIX: String Notional to bypass 100% of Alpaca Float Errors
  ✅ PRO FIX: Robust DataFrame validation for Yahoo Finance
  ✅ PRO FIX: Max Orders Limit to prevent API Spam
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

# ── 1. GLOBAL TIMER & SAFETY ─────────────────────────────────────────────────
START_TIME   = time.time()
MAX_RUNTIME  = 45.0   # Absolute hard stop at 45 seconds
SESSION_LOGS = []     # Stores logs for the final Telegram message

def check_timeout():
    elapsed = time.time() - START_TIME
    if elapsed > MAX_RUNTIME:
        log_action(f"🚨 FATAL TIMEOUT: Script running for {elapsed:.1f}s. Emergency Stop.")
        send_telegram_summary()
        sys.exit(1)

# ── 2. CONFIGURATION & KEYS ──────────────────────────────────────────────────
API_KEY          = os.environ.get("ALPACA_API_KEY")
SECRET_KEY       = os.environ.get("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not API_KEY or not SECRET_KEY:
    print("🚨 FATAL ERROR: Missing Alpaca API Keys. Check GitHub Secrets.")
    sys.exit(1)

# THE TRIGGERS
LIVE_EXECUTION = True  # True = Robot pulls the trigger on Alpaca
IS_PAPER       = True  # True = Shoots blanks (Virtual Money). False = Real Money.

# RISK PARAMETERS
BASE_MAX_ORDER_SIZE     = 500.0   # Max USD per single trade
MIN_TRADE_SIZE          = 10.0    # Min USD to avoid micro-transactions
BASE_MAX_TOTAL_EXPOSURE = 2500.0  # Max total portfolio exposure in USD
MAX_ORDERS_PER_CYCLE    = 10      # Safety cap to prevent Alpaca API spam

CRYPTO_BLACKLIST = {
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
    "BTCUSD",  "ETHUSD",  "SOLUSD",  "XRPUSD"
}

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=IS_PAPER)

# ── 3. STRUCTURED TELEMETRY (LOGS & TELEGRAM) ────────────────────────────────
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
    """Sends the execution summary to your phone."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ No Telegram keys found, logs sent to console only.")
        return
        
    mode_txt = "🟡 VIRTUAL MONEY (Paper)" if IS_PAPER else "🔴 REAL MONEY (Live)"
    exec_txt = "ACTIVE" if LIVE_EXECUTION else "DRY RUN"
    
    log_text = "\n".join(SESSION_LOGS)
    message  = f"🏦 *HEDGE FUND FACTORY V6*\n_{mode_txt} | Exec: {exec_txt}_\n\n```text\n{log_text}\n```"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"⚠️ Telegram sending failed: {e}")

# ── 4. THE CENTRAL BRAIN (META CONTROLLER) ───────────────────────────────────
def get_market_context():
    check_timeout()
    context = {"panic_mode": False, "allow_buying": True, "risk_multiplier": 1.0, "regime": "UNKNOWN"}
    
    try:
        file_path = "global_settings.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                meta = json.load(f)
                context["panic_mode"]      = meta.get("panic_mode", False)
                context["allow_buying"]    = meta.get("allow_buying", True)
                context["risk_multiplier"] = float(meta.get("global_risk_multiplier", 1.0))
                context["regime"]          = meta.get("market_regime", "UNKNOWN")
                log_action(f"🧠 MACRO: Regime {context['regime']} | Risk {context['risk_multiplier']}x")
        else:
            log_action("⚠️ Meta Controller not found. Default risk 1.0x")
    except Exception as e:
        log_action(f"⚠️ Brain Error ({e}), fallback risk 0.5x.")
        context["risk_multiplier"] = 0.5 
    return context

# ── 5. DATA & CONVERSION (ROBUST FX) ─────────────────────────────────────────
def get_eurusd_rate():
    check_timeout()
    try:
        df = yf.download("EURUSD=X", period="1d", progress=False)
        
        if df.empty or "Close" not in df:
            raise ValueError("Empty FX data")
            
        eurusd = float(df["Close"].squeeze().iloc[-1])
        return eurusd
        
    except Exception as e:
        log_action(f"⚠️ FX ERROR: ({e}) -> Fallback 1.08")
        return 1.08

def get_target_positions():
    target_allocations = {}
    for file in glob.glob("portfolio_*.json"):
        check_timeout()
        if "backup" in file or "tmp" in file: continue
        try:
            with open(file, "r") as f:
                data = json.load(f)
                for ticker, pos_data in data.get("positions", {}).items():
                    if ticker in CRYPTO_BLACKLIST:
                        log_action(f"⏭️ SKIP CRYPTO: {ticker} ignored at source.")
                        continue
                        
                    target_allocations[ticker] = target_allocations.get(ticker, 0) + pos_data.get("mise", 0)
        except Exception as e:
            log_action(f"⚠️ Error reading {file}: {e}")
            
    return target_allocations

# ── 6. THE REBALANCING ENGINE ────────────────────────────────────────────────
def execute_trades():
    log_action("🚀 STARTING V6 APEX EXECUTION ENGINE...")
    
    market_context = get_market_context()
    
    if market_context["panic_mode"]:
        log_action("🚨 PANIC MODE: Central system locked. Aborting.")
        send_telegram_summary()
        sys.exit(0)
        
    if market_context["risk_multiplier"] < 0.2:
        log_action("🛑 RISK TOO LOW (< 0.2). Aborting.")
        send_telegram_summary()
        sys.exit(0)

    active_max_order    = BASE_MAX_ORDER_SIZE * market_context["risk_multiplier"]
    active_max_exposure = BASE_MAX_TOTAL_EXPOSURE * market_context["risk_multiplier"]
    
    if not market_context["allow_buying"]:
        log_action("⚠️ BUYING BLOCKED: 'Liquidations Only' Mode Active.")

    fx_rate     = get_eurusd_rate()
    
    # PRO FIX: Round targets early to prevent cumulative floating point drift
    targets_eur = get_target_positions()
    targets_usd = {
        ticker: round(mise_eur * fx_rate, 2) 
        for ticker, mise_eur in targets_eur.items()
    }
    
    total_planned_exposure = sum(targets_usd.values())
    if total_planned_exposure > active_max_exposure:
        log_action(f"🚨 FATAL: Exposure (${total_planned_exposure:.0f}) > Limit (${active_max_exposure:.0f}).")
        send_telegram_summary()
        return

    try:
        current_alpaca_positions = trading_client.get_all_positions()
        current_holdings = {pos.symbol: float(pos.market_value) for pos in current_alpaca_positions}
    except Exception as e:
        log_action(f"❌ FATAL: Alpaca API unreachable: {e}")
        send_telegram_summary()
        return

    # ── PHASE 1: LIQUIDATIONS ──
    log_action("── PHASE 1: LIQUIDATIONS ──")
    for symbol in list(current_holdings.keys()):
        check_timeout()
        factory_symbol = symbol.replace("USD", "-USD") if symbol in ["BTCUSD", "ETHUSD"] else symbol
        
        if factory_symbol not in targets_usd:
            log_action(f"🛑 LIQUIDATING: {symbol}")
            if LIVE_EXECUTION:
                try: 
                    trading_client.close_position(symbol)
                except Exception as e: 
                    log_action(f"❌ FAIL Liquidation {symbol}: {e}")
            else:
                log_action(f"[DRY RUN] Would liquidate {symbol}")
            del current_holdings[symbol]

    # ── PHASE 2: BURST ORDERS ──
    log_action("── PHASE 2: ORDERS ──")
    pending_verifications = []
    orders_sent = 0
    
    for factory_symbol, target_usd in targets_usd.items():
        check_timeout()
        
        if orders_sent >= MAX_ORDERS_PER_CYCLE:
            log_action("⚠️ MAX ORDERS REACHED: Cap at 10 to prevent API spam.")
            break
            
        alpaca_symbol = factory_symbol
        current_usd   = current_holdings.get(alpaca_symbol, 0.0)
        difference    = target_usd - current_usd
        
        if abs(difference) < MIN_TRADE_SIZE: 
            continue
            
        side = OrderSide.BUY if difference > 0 else OrderSide.SELL

        if side == OrderSide.BUY and not market_context["allow_buying"]:
            log_action(f"🚫 BUY CANCELLED: {alpaca_symbol} (allow_buying = False)")
            continue

        if abs(difference) > active_max_order:
            log_action(f"⚠️ TRIM: Delta for {alpaca_symbol} reduced to ${active_max_order:.0f}")
            
        # PRO FIX: 1) Round mathematically, then 2) Format as exact String
        trade_amount = round(min(abs(difference), active_max_order), 2)
        notional_str = f"{trade_amount:.2f}"

        if LIVE_EXECUTION:
            try:
                order = MarketOrderRequest(
                    symbol=alpaca_symbol, 
                    notional=notional_str, # String injection blocks 422 API errors
                    side=side, 
                    time_in_force=TimeInForce.DAY
                )
                submitted_order = trading_client.submit_order(order_data=order)
                log_action(f"⚡ ORDER: {side.name} ${notional_str} of {alpaca_symbol}")
                pending_verifications.append((alpaca_symbol, trade_amount, submitted_order.id))
                orders_sent += 1
            except Exception as e:
                log_action(f"❌ FAIL {alpaca_symbol}: {e}")
        else:
            log_action(f"[DRY RUN] {side.name} ${notional_str} of {alpaca_symbol}")

    # ── PHASE 3: VERIFICATION ──
    if LIVE_EXECUTION and pending_verifications:
        log_action("── PHASE 3: VERIFICATION ──")
        time.sleep(2.0)
        for symbol, amount, order_id in pending_verifications:
            check_timeout()
            try:
                order_status = trading_client.get_order_by_id(order_id)
                if order_status.status == "filled":
                    log_action(f"✅ FILLED: {symbol}")
                else:
                    log_action(f"⚠️ PENDING: {symbol} ({order_status.status})")
            except: 
                pass

    elapsed_total = time.time() - START_TIME
    log_action(f"🏁 CYCLE COMPLETED IN {elapsed_total:.1f}s.")
    
    send_telegram_summary()

if __name__ == "__main__":
    execute_trades()
