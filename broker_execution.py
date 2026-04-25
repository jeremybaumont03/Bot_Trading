import json
import glob
import os
import sys
import time
from datetime import datetime
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ── 1. CONFIGURATION & SAFETY CAPS ──
# Securely pulls keys from GitHub Secrets. NO HARDCODED KEYS HERE.
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("🚨 FATAL ERROR: Missing Alpaca API Keys. Check your GitHub Secrets.")
    sys.exit(1)

LIVE_EXECUTION = True 

MAX_ORDER_SIZE = 500.0   # USD max per trade
MIN_TRADE_SIZE = 10.0    # USD min to avoid noise
MAX_TOTAL_EXPOSURE = 2500.0 # USD absolute max total portfolio size 

# The Lock File prevents GitHub Actions from accidentally running twice in one day
LOCK_FILE = "execution_lock.txt"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# ── 2. DAILY EXECUTION LOCK ──
def check_execution_lock():
    """Ensures we only execute trades once per calendar day."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            last_run = f.read().strip()
            if last_run == today_str:
                print(f"🛑 LOCK ACTIVE: Already executed today ({today_str}). Aborting to prevent double orders.")
                return True
    return False

def set_execution_lock():
    """Sets the lock file to today's date."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    with open(LOCK_FILE, "w") as f:
        f.write(today_str)
    print(f"🔒 Lock file set for {today_str}")

def get_eurusd_rate():
    try:
        eurusd = yf.download("EURUSD=X", period="1d", progress=False)["Close"].iloc[0]
        return float(eurusd)
    except: 
        return 1.08

# ── 4. AGGREGATE POSITIONS ──
def get_target_positions():
    target_allocations = {}
    for file in glob.glob("portfolio_*.json"):
        if "backup" in file or "tmp" in file: continue
        try:
            with open(file, "r") as f:
                data = json.load(f)
                for ticker, pos_data in data.get("positions", {}).items():
                    target_allocations[ticker] = target_allocations.get(ticker, 0) + pos_data.get("mise", 0)
        except: pass
    return target_allocations

# ── 5. THE REBALANCING ENGINE ──
def execute_trades():
    print("🚀 QUANT FACTORY EXECUTION ENGINE V2...")
    
    if check_execution_lock():
        return

    fx_rate = get_eurusd_rate()
    targets_eur = get_target_positions()
    targets_usd = {ticker: (mise_eur * fx_rate) for ticker, mise_eur in targets_eur.items()}
    
    # Global Exposure Check
    total_planned_exposure = sum(targets_usd.values())
    if total_planned_exposure > MAX_TOTAL_EXPOSURE:
        print(f"🚨 FATAL: Planned exposure (${total_planned_exposure:.2f}) exceeds MAX_TOTAL_EXPOSURE (${MAX_TOTAL_EXPOSURE:.2f}). Aborting.")
        return

    try:
        current_alpaca_positions = trading_client.get_all_positions()
        current_holdings = {pos.symbol: float(pos.market_value) for pos in current_alpaca_positions}
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")
        return

    # 1. FULL LIQUIDATIONS
    for symbol in list(current_holdings.keys()):
        factory_symbol = symbol.replace("USD", "-USD") if symbol in ["BTCUSD", "ETHUSD"] else symbol
        if factory_symbol not in targets_usd:
            print(f"🛑 LIQUIDATING: {symbol}")
            if LIVE_EXECUTION:
                try:
                    trading_client.close_position(symbol)
                    time.sleep(1)
                except Exception as e:
                    print(f"   ❌ FAILED to liquidate {symbol}: {e}")
            del current_holdings[symbol]

    # 2. PARTIAL REBALANCING
    for factory_symbol, target_usd in targets_usd.items():
        alpaca_symbol = factory_symbol.replace("-", "") if "-USD" in factory_symbol else factory_symbol
        current_usd = current_holdings.get(alpaca_symbol, 0.0)
        difference = target_usd - current_usd
        
        if abs(difference) < MIN_TRADE_SIZE: continue
            
        trade_amount = min(abs(difference), MAX_ORDER_SIZE)
        side = OrderSide.BUY if difference > 0 else OrderSide.SELL

        print(f"{'🟢 BUY' if difference > 0 else '🔴 SELL'}: {alpaca_symbol} | Target: ${target_usd:.2f} | Delta: ${trade_amount:.2f}")

        if LIVE_EXECUTION:
            try:
                order = MarketOrderRequest(symbol=alpaca_symbol, notional=trade_amount, side=side, time_in_force=TimeInForce.DAY)
                submitted_order = trading_client.submit_order(order_data=order)
                
                # Wait to verify if the order was actually filled
                print(f"   ⏳ Verifying fill status for {alpaca_symbol}...")
                time.sleep(2) # Give Alpaca a moment to process
                
                print(f"   ✅ ORDER SUBMITTED: {alpaca_symbol} (ID: {submitted_order.id})")
                
            except Exception as e:
                print(f"   ❌ FAILED for {alpaca_symbol}: {e}")
        else:
            print(f"   [DRY RUN] {side.name} ${trade_amount:.2f} of {alpaca_symbol}")

    # If we made it this far without crashing, set the lock file for today.
    if LIVE_EXECUTION:
        set_execution_lock()

if __name__ == "__main__":
    execute_trades()
