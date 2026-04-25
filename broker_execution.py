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

# ── 1. CHRONOMÈTRE GLOBAL & SÉCURITÉ ──
START_TIME = time.time()
MAX_RUNTIME = 45.0  # Hard stop absolu à 45 secondes
SESSION_LOGS = []   # Stocke les logs pour le message Telegram final

def check_timeout():
    elapsed = time.time() - START_TIME
    if elapsed > MAX_RUNTIME:
        log_action(f"🚨 TIMEOUT FATAL: Le script tourne depuis {elapsed:.1f}s. Arrêt d'urgence.")
        send_telegram_summary()
        sys.exit(1)

# ── 2. CONFIGURATION DE BASE & CLÉS ──
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not API_KEY or not SECRET_KEY:
    print("🚨 FATAL ERROR: Missing Alpaca API Keys. Check GitHub Secrets.")
    sys.exit(1)

LIVE_EXECUTION = True 
BASE_MAX_ORDER_SIZE = 500.0   # USD max de base par trade
MIN_TRADE_SIZE = 10.0         # USD min to avoid noise
BASE_MAX_TOTAL_EXPOSURE = 2500.0 # USD max portfolio de base

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# ── 3. STRUCTURED TELEMETRY (LOGS & TELEGRAM) ──
def log_action(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    SESSION_LOGS.append(message) # On stocke sans le timestamp pour Telegram (plus propre)
    
    # Écriture locale (utile uniquement si testé sur ton PC)
    try:
        with open("execution_log.txt", "a") as f:
            f.write(log_line + "\n")
    except: pass

def send_telegram_summary():
    """Envoie le résumé de l'exécution sur ton téléphone avant que GitHub ne s'éteigne."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Pas de clés Telegram trouvées, logs envoyés uniquement dans la console.")
        return
        
    log_text = "\n".join(SESSION_LOGS)
    message = f"🏦 *RAPPORT D'EXÉCUTION ALPACA*\n\n```text\n{log_text}\n```"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"⚠️ Échec de l'envoi Telegram : {e}")

# ── 4. LE CERVEAU CENTRAL (META CONTROLLER) ──
def get_market_context():
    check_timeout()
    context = {"panic_mode": False, "allow_buying": True, "risk_multiplier": 1.0, "regime": "UNKNOWN"}
    
    try:
        file_path = "meta_controller.json" if os.path.exists("meta_controller.json") else "global_settings.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                meta = json.load(f)
                context["panic_mode"] = meta.get("panic_mode", False)
                context["allow_buying"] = meta.get("allow_buying", True)
                context["risk_multiplier"] = float(meta.get("global_risk_multiplier", 1.0))
                context["regime"] = meta.get("market_regime", "UNKNOWN")
                log_action(f"🧠 MACRO: Régime {context['regime']} | Risque {context['risk_multiplier']}x")
        else:
            log_action("⚠️ Meta Controller introuvable. Risque par défaut 1.0x")
    except Exception as e:
        log_action(f"⚠️ Erreur Cerveau ({e}), fallback risque 0.5x.")
        context["risk_multiplier"] = 0.5 
    return context

# ── 5. DATA & CONVERSION ──
def get_eurusd_rate():
    check_timeout()
    try:
        eurusd = yf.download("EURUSD=X", period="1d", progress=False)["Close"].iloc[0]
        return float(eurusd)
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
                    target_allocations[ticker] = target_allocations.get(ticker, 0) + pos_data.get("mise", 0)
        except: pass
    return target_allocations

# ── 6. THE REBALANCING ENGINE (V6 APEX) ──
def execute_trades():
    log_action("🚀 DÉMARRAGE MOTEUR D'EXÉCUTION V6...")
    
    market_context = get_market_context()
    
    if market_context["panic_mode"]:
        log_action("🚨 PANIC MODE: Système central bloqué. Aborting.")
        send_telegram_summary()
        sys.exit(0)
        
    if market_context["risk_multiplier"] < 0.2:
        log_action("🛑 RISQUE TROP FAIBLE (< 0.2). Aborting.")
        send_telegram_summary()
        sys.exit(0)

    active_max_order = BASE_MAX_ORDER_SIZE * market_context["risk_multiplier"]
    active_max_exposure = BASE_MAX_TOTAL_EXPOSURE * market_context["risk_multiplier"]
    
    if not market_context["allow_buying"]:
        log_action("⚠️ ACHATS BLOQUÉS: Mode 'Liquidations Uniquement'.")

    fx_rate = get_eurusd_rate()
    targets_eur = get_target_positions()
    targets_usd = {ticker: (mise_eur * fx_rate) for ticker, mise_eur in targets_eur.items()}
    
    total_planned_exposure = sum(targets_usd.values())
    if total_planned_exposure > active_max_exposure:
        log_action(f"🚨 FATAL: Exposition ({total_planned_exposure:.0f}$) > Limite Dynamique ({active_max_exposure:.0f}$).")
        send_telegram_summary()
        return

    try:
        current_alpaca_positions = trading_client.get_all_positions()
        current_holdings = {pos.symbol: float(pos.market_value) for pos in current_alpaca_positions}
    except Exception as e:
        log_action(f"❌ FATAL: API Alpaca inaccessible: {e}")
        send_telegram_summary()
        return

    # ── PHASE 1: LIQUIDATIONS ──
    for symbol in list(current_holdings.keys()):
        check_timeout()
        factory_symbol = symbol.replace("USD", "-USD") if symbol in ["BTCUSD", "ETHUSD"] else symbol
        
        if factory_symbol not in targets_usd:
            log_action(f"🛑 LIQUIDATION: {symbol}")
            if LIVE_EXECUTION:
                try: trading_client.close_position(symbol)
                except Exception as e: log_action(f"❌ ECHEC Liquidation {symbol}: {e}")
            del current_holdings[symbol]

    # ── PHASE 2: ORDRES EN RAFALE ──
    pending_verifications = []
    
    for factory_symbol, target_usd in targets_usd.items():
        check_timeout()
        
        # FIX CRYPTO: On bloque proprement les ordres Notional sur les cryptos
        if factory_symbol in ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]:
            log_action(f"⏭️ SKIP CRYPTO: {factory_symbol} (Nécessite calcul QTY)")
            continue
            
        alpaca_symbol = factory_symbol
        current_usd = current_holdings.get(alpaca_symbol, 0.0)
        difference = target_usd - current_usd
        
        if abs(difference) < MIN_TRADE_SIZE: continue
            
        side = OrderSide.BUY if difference > 0 else OrderSide.SELL

        if side == OrderSide.BUY and not market_context["allow_buying"]:
            log_action(f"🚫 ACHAT ANNULÉ: {alpaca_symbol} (allow_buying = False)")
            continue

        if abs(difference) > active_max_order:
            log_action(f"⚠️ TRIM: Delta {alpaca_symbol} réduit à {active_max_order:.0f}$")
            
        trade_amount = min(abs(difference), active_max_order)

        if LIVE_EXECUTION:
            try:
                order = MarketOrderRequest(symbol=alpaca_symbol, notional=trade_amount, side=side, time_in_force=TimeInForce.DAY)
                submitted_order = trading_client.submit_order(order_data=order)
                log_action(f"⚡ ORDER: {side.name} {trade_amount:.0f}$ de {alpaca_symbol}")
                pending_verifications.append((alpaca_symbol, trade_amount, submitted_order.id))
            except Exception as e:
                log_action(f"❌ ECHEC {alpaca_symbol}: {e}")
        else:
            log_action(f"[DRY RUN] {side.name} {trade_amount:.0f}$ de {alpaca_symbol}")

    # ── PHASE 3: VÉRIFICATION ──
    if LIVE_EXECUTION and pending_verifications:
        time.sleep(2.0)
        for symbol, amount, order_id in pending_verifications:
            check_timeout()
            try:
                order_status = trading_client.get_order_by_id(order_id)
                if order_status.status == "filled":
                    log_action(f"✅ FILLED: {symbol}")
                else:
                    log_action(f"⚠️ PENDING: {symbol} ({order_status.status})")
            except: pass

    elapsed_total = time.time() - START_TIME
    log_action(f"🏁 CYCLE TERMINÉ EN {elapsed_total:.1f}s.")
    
    # Envoi final du rapport vers Telegram
    send_telegram_summary()

if __name__ == "__main__":
    execute_trades()
