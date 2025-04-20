import time
import base58
import requests
import os
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from solana.rpc.api import Client
from solana.rpc.types import TxOpts, TokenAccountOpts
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from base64 import b64decode
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Logging setup
logger = logging.getLogger('TradingBot')
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear all handlers
handler = RotatingFileHandler('bot.log', maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S'))
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S'))
logger.addHandler(console_handler)

def log(msg):
    logger.info(msg)

log("Starting bot...")

# Configuration
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
if not HELIUS_API_KEY:
    log("ERROR: HELIUS_API_KEY not set")
    raise ValueError("HELIUS_API_KEY must be set in environment variables")

RPC_URLS = [
    f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}",
    "https://solana-rpc.publicnode.com",
    "https://api.mainnet-beta.solana.com",
]
log(f"RPC URLs: {RPC_URLS}")

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
if not PRIVATE_KEY:
    log("ERROR: PRIVATE_KEY not set")
    raise ValueError("PRIVATE_KEY must be set in environment variables")
log("Loading private key...")
try:
    keypair = Keypair.from_bytes(base58.b58decode(PRIVATE_KEY))
    wallet_pub = keypair.pubkey()
    log(f"Wallet public key: {wallet_pub}")
except Exception as e:
    log(f"ERROR: Failed to load private key: {e}")
    raise

USDC_MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
SOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")
TRADE_INTERVAL = 30  # Adjusted dynamically
BASE_BUY_TRIGGER = 2.0  # %, adjusted dynamically
BASE_SELL_TRIGGER = 3.0  # %, adjusted dynamically
STOP_LOSS_DROP = 5.0  # %
TRAILING_STOP = 2.5  # %, activated at 3.5% profit
SLIPPAGE = 0.003  # 0.3%
MIN_TRADE_USD = 1.0
MAX_POSITION_SOL = 3.0
MAX_DRAWDOWN = 15.0  # %
MIN_SOL_THRESHOLD = 0.01  # Ensure SOL balance never goes below 0.01 SOL

# Client setup
log("Initializing Solana client...")
client = Client(RPC_URLS[0])
log(f"Connected to RPC: {client._provider.endpoint_uri}")

# State
state = {
    'position': 0.0,  # SOL amount
    'entry_price': 0.0,
    'sell_targets': [],  # List of (amount, target_price)
    'highest_price': 0.0,
    'peak_portfolio': 0.0,
    'pause_until': 0,
    'price_history': [],
    'atr_history': [],
    'last_fetch_time': 0,
    'current_rpc_index': 0,
    'total_trades': 0,
    'wins': 0,
    'losses': 0,
    'total_profit': 0.0,
    'last_fee_update': 0,
    'cached_fee': 0.002,  # Default fee estimate
    'recent_trades': [],  # For win streak tracking
}

# Initialize Price History
def initialize_price_history():
    log("Initializing price history...")
    price_file = 'price_history.json'
    required_prices = 34  # Enough for MACD (26 + 9 - 1)

    # Try loading from file if recent
    if os.path.exists(price_file):
        try:
            with open(price_file, 'r') as f:
                data = json.load(f)
                prices = data.get('prices', [])
                timestamp = data.get('timestamp', 0)
                if time.time() - timestamp < 3600 and len(prices) >= required_prices:  # Use if <1 hour old
                    state['price_history'] = prices[-required_prices:]
                    log(f"Loaded {len(state['price_history'])} prices from file")
                    return
                else:
                    log("Price history outdated or insufficient, fetching new data")
        except Exception as e:
            log(f"Failed to load price history: {e}")

    # Fetch live prices
    prices = []
    attempts = 0
    max_attempts = 5
    while len(prices) < required_prices and attempts < max_attempts:
        price = fetch_current_price()
        if price:
            prices.append(price)
            log(f"Fetched price {len(prices)}/{required_prices}: ${price:.2f}")
            time.sleep(0.5)  # Small delay to avoid rate limits
        else:
            attempts += 1
            log(f"Price fetch failed, attempt {attempts}/{max_attempts}")
            time.sleep(2)

    if len(prices) < required_prices:
        log(f"ERROR: Could not fetch enough prices, got {len(prices)}/{required_prices}")
        raise RuntimeError("Failed to initialize price history")

    state['price_history'] = prices[-required_prices:]
    log(f"Initialized {len(state['price_history'])} prices")

    # Save to file
    try:
        with open(price_file, 'w') as f:
            json.dump({'prices': state['price_history'], 'timestamp': time.time()}, f)
        log("Saved price history to file")
    except Exception as e:
        log(f"Failed to save price history: {e}")

# Helper Functions
def fetch_current_price():
    log("Fetching USDC/SOL price...")
    url = f"https://quote-api.jup.ag/v6/quote?inputMint={SOL_MINT}&outputMint={USDC_MINT}&amount=1000000000&slippageBps=0"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            out_amount = int(data['outAmount'])
            price = out_amount / 1e6  # USDC per SOL
            log(f"Price fetched: ${price:.2f}")

            # Update price history file
            state['price_history'].append(price)
            if len(state['price_history']) > 200:
                state['price_history'].pop(0)
            try:
                with open('price_history.json', 'w') as f:
                    json.dump({'prices': state['price_history'], 'timestamp': time.time()}, f)
            except Exception as e:
                log(f"Failed to save price history: {e}")

            return price
        else:
            log(f"Price fetch failed: Status {response.status_code}")
    except Exception as e:
        log(f"Price fetch error: {e}")
    return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_sol_balance():
    global state
    log("Fetching SOL balance...")
    try:
        resp = client.get_balance(wallet_pub)
        balance = resp.value / 1e9
        log(f"SOL balance: {balance:.4f}")
        return balance
    except Exception as e:
        log(f"RPC {client._provider.endpoint_uri} failed for SOL balance: {e}")
        state['current_rpc_index'] = (state['current_rpc_index'] + 1) % len(RPC_URLS)
        client._provider.endpoint_uri = RPC_URLS[state['current_rpc_index']]
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_usdc_balance():
    global state, client
    log("Fetching USDC balance...")
    token_account = Pubkey.from_string("8usswUTuW7Sdr7K3FZ145SwBHcVyEyvg2p9R3Q1fnFiv")
    try:
        resp = client.get_token_account_balance(token_account)
        balance = resp.value.ui_amount
        log(f"USDC balance: {balance:.4f}")
        return balance if balance is not None else 0
    except Exception as e:
        log(f"RPC {client._provider.endpoint_uri} failed for USDC balance: {e}")
        state['current_rpc_index'] = (state['current_rpc_index'] + 1) % len(RPC_URLS)
        client._provider.endpoint_uri = RPC_URLS[state['current_rpc_index']]
        raise

def get_portfolio_value(price):
    log("Calculating portfolio value...")
    sol_balance = get_sol_balance()
    usdc_balance = get_usdc_balance()
    value = usdc_balance + sol_balance * price if price else usdc_balance
    log(f"Portfolio value: ${value:.2f}")
    return value

def get_fee_estimate():
    current_time = time.time()
    if current_time - state['last_fee_update'] < 180:
        log(f"Using cached fee: {state['cached_fee']*100:.2f}%")
        return state['cached_fee']
    log("Fetching new fee estimate...")
    try:
        url = f"https://quote-api.jup.ag/v6/quote?inputMint={USDC_MINT}&outputMint={SOL_MINT}&amount=1000000&slippageBps={int(SLIPPAGE * 10000)}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            in_amount = int(data['inAmount'])
            out_amount = int(data['outAmount'])
            current_price = fetch_current_price()
            if current_price:
                fee = (in_amount - out_amount * current_price) / in_amount
                state['cached_fee'] = max(0.002, min(fee, 0.005))
                state['last_fee_update'] = current_time
                log(f"New fee: {state['cached_fee']*100:.2f}%")
                return state['cached_fee']
        log(f"Fee fetch failed: Status {response.status_code}")
    except Exception as e:
        log(f"Fee fetch error: {e}")
    return state['cached_fee']

# Indicator Functions
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        log("Not enough prices for RSI")
        return None
    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    log(f"RSI: {rsi:.2f}")
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal - 1:
        log("Not enough prices for MACD")
        return None, None
    ema_fast = np.convolve(prices, np.ones(fast) / fast, mode='valid')[-1]
    ema_slow = np.convolve(prices, np.ones(slow) / slow, mode='valid')[-1]
    macd_line = ema_fast - ema_slow
    macd_history = [np.convolve(prices[i:], np.ones(fast) / fast, mode='valid')[0] - 
                    np.convolve(prices[i:], np.ones(slow) / slow, mode='valid')[0] 
                    for i in range(len(prices) - slow - signal + 1, len(prices) - slow + 1)]
    signal_line = np.mean(macd_history[-signal:])
    log(f"MACD: line={macd_line:.4f}, signal={signal_line:.4f}")
    return macd_line, signal_line

def calculate_vwap(prices, period=20):
    if len(prices) < period:
        log("Not enough prices for VWAP")
        return None
    vwap = np.mean(prices[-period:])
    log(f"VWAP: ${vwap:.2f}")
    return vwap

def calculate_bollinger_bands(prices, period=20, stddev=2):
    if len(prices) < period:
        log("Not enough prices for Bollinger Bands")
        return None, None
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = sma + stddev * std
    lower = sma - stddev * std
    log(f"Bollinger Bands: upper=${upper:.2f}, lower=${lower:.2f}")
    return upper, lower

def calculate_atr(prices, period=20):
    if len(prices) < period + 1:
        log("Not enough prices for ATR")
        return None
    tr = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    atr = np.mean(tr[-period:])
    log(f"ATR: ${atr:.4f}")
    return atr

def calculate_momentum(prices, period=10):
    if len(prices) < period + 1:
        log("Not enough prices for momentum")
        return None
    momentum = (prices[-1] / prices[-period - 1] - 1) * 100
    log(f"Momentum: {momentum:.2f}%")
    return momentum

# Trade Logic Functions
def adjust_triggers(atr, avg_atr, rsi):
    global BASE_BUY_TRIGGER, BASE_SELL_TRIGGER
    if atr is not None and avg_atr is not None and avg_atr > 0 and atr > 2 * avg_atr:
        BASE_BUY_TRIGGER = min(2.5, BASE_BUY_TRIGGER + 0.7)
        BASE_SELL_TRIGGER = min(4.0, BASE_SELL_TRIGGER + 0.7)
        log(f"High ATR, triggers: buy={BASE_BUY_TRIGGER}%, sell={BASE_SELL_TRIGGER}%")
    if rsi is not None and (rsi < 25 or rsi > 68):
        BASE_BUY_TRIGGER = max(1.5, BASE_BUY_TRIGGER - 0.5)
        BASE_SELL_TRIGGER = max(2.5, BASE_SELL_TRIGGER - 0.5)
        log(f"Extreme RSI, triggers: buy={BASE_BUY_TRIGGER}%, sell={BASE_SELL_TRIGGER}%")

def check_buy_signal(price, rsi, macd_line, signal_line, vwap, lower_bb, momentum, atr, avg_atr):
    if any(x is None for x in [rsi, macd_line, signal_line, vwap, lower_bb, momentum, atr, avg_atr]):
        log("Missing indicators, skipping buy")
        return False
    fee = get_fee_estimate()
    if fee > 0.0025:
        log(f"Fees too high ({fee*100:.2f}%), skipping buy")
        return False
    # Bid-ask spread check
    bid_ask_spread = abs(fetch_current_price() - price) / price if price else 0.01
    spread_condition = bid_ask_spread < 0.005
    if not spread_condition:
        log(f"Bid-ask spread too high ({bid_ask_spread*100:.2f}%), skipping buy")
        return False
    # Core requirements
    if rsi >= 35 or macd_line <= signal_line:
        log(f"Core conditions failed: RSI={rsi:.2f}, MACD={macd_line:.4f}<={signal_line:.4f}")
        return False
    # Weighted scoring for additional indicators
    momentum_weight = 0.4
    vwap_weight = 0.3
    bb_weight = 0.3
    momentum_score = 1 if momentum > 0.4 else 0
    vwap_score = 1 if price < vwap * 0.995 else 0
    bb_score = 1 if price < lower_bb * 1.01 else 0
    total_score = (momentum_weight * momentum_score) + (vwap_weight * vwap_score) + (bb_weight * bb_score)
    signal = total_score >= 0.6  # At least 2/3 conditions met
    log(f"Buy signal check: {'True' if signal else 'False'} (RSI={rsi:.2f}, MACD={macd_line:.4f}>{signal_line:.4f}, Score={total_score:.2f})")
    return signal

def calculate_position_size(portfolio_value, atr, avg_atr):
    log("Calculating position size...")
    if atr is None or avg_atr is None or avg_atr == 0:
        fraction = 0.1
    else:
        fraction = min(0.2, max(0.05, 0.1 * (avg_atr / atr)))
        if atr > 2.5 * avg_atr:
            fraction = 0.05
        elif atr < 0.7 * avg_atr and sum(1 for t in state['recent_trades'][-10:] if t > 0) >= 5:
            fraction = 0.2
    size_usd = fraction * portfolio_value
    price = fetch_current_price()
    if not price:
        log("No price for sizing, skipping")
        return 0
    min_size_sol = max(MIN_TRADE_USD / price, 0.0001)
    max_size_sol = min(MAX_POSITION_SOL, 0.2 * portfolio_value / price)
    size_sol = max(min_size_sol, min(size_usd / price, max_size_sol))
    usdc_needed = size_sol * price * (1 + SLIPPAGE + get_fee_estimate())
    if get_usdc_balance() < usdc_needed:
        log(f"Not enough USDC: need {usdc_needed:.2f}, have {get_usdc_balance():.2f}")
        return 0
    log(f"Position size: {size_sol:.4f} SOL (~${size_usd:.2f})")
    return size_sol

def get_route(from_mint, to_mint, amount):
    log(f"Fetching swap route: {from_mint} -> {to_mint}, amount={amount}")
    url = f"https://quote-api.jup.ag/v6/quote?inputMint={from_mint}&outputMint={to_mint}&amount={int(amount)}&slippageBps={int(SLIPPAGE * 10000)}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            route = data['data'][0] if 'data' in data and data['data'] else None
            if route:
                log("Route found")
                return route
            log("No routes available")
        else:
            log(f"Route fetch failed: Status {response.status_code}")
    except Exception as e:
        log(f"Route fetch error: {e}")
    return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def send_trade(route, current_price):
    log("Sending trade...")
    url = "https://quote-api.jup.ag/v6/swap"
    payload = {
        "userPublicKey": str(wallet_pub),
        "quoteResponse": route,
        "wrapAndUnwrapSol": True,
        "computeUnitPriceMicroLamports": 700
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            tx_raw = data['swapTransaction']
            tx = Transaction.deserialize(b64decode(tx_raw))
            tx.sign(keypair)
            result = client.send_transaction(tx, keypair, opts=TxOpts(skip_preflight=False))
            tx_id = result.value
            time.sleep(5)
            in_amount = int(route['inAmount'])
            out_amount = int(route['outAmount'])
            log(f"Trade sent: tx_id={tx_id}")
            return tx_id, in_amount, out_amount
        log(f"Trade failed: Status {response.status_code}")
    except Exception as e:
        log(f"Trade execution error: {e}")
        raise
    return None, 0, 0

def execute_buy(position_size):
    log(f"Executing buy: {position_size:.4f} SOL")
    price = fetch_current_price()
    if not price:
        log("No price, aborting buy")
        return
    input_amount_usdc = int(position_size * price * (1 + SLIPPAGE) * 1e6)
    route = get_route(str(USDC_MINT), str(SOL_MINT), input_amount_usdc)
    if route:
        tx_id, in_amount, out_amount = send_trade(route, price)
        if tx_id:
            sol_bought = out_amount / 1e9
            state['position'] = sol_bought
            state['entry_price'] = price
            state['highest_price'] = price
            state['total_trades'] += 1
            log(f"✅ Bought {sol_bought:.4f} SOL @ ${price:.2f}")
            set_sell_targets(sol_bought, price)

def set_sell_targets(position_size, entry_price):
    log(f"Setting sell targets for {position_size:.4f} SOL")
    if position_size < 1:
        state['sell_targets'] = [
            (position_size * 0.5, entry_price * 1.025),
            (position_size * 0.3, entry_price * 1.04),
            (position_size * 0.2, entry_price * 1.06)
        ]
    else:
        state['sell_targets'] = [
            (position_size * 0.4, entry_price * 1.025),
            (position_size * 0.4, entry_price * 1.04),
            (position_size * 0.2, entry_price * 1.06)
        ]
    log(f"Sell targets: {state['sell_targets']}")

def execute_sell(amount, price):
    log(f"Executing sell: {amount:.4f} SOL @ ${price:.2f}")
    total_sol_balance = get_sol_balance()  # Fetch total SOL balance
    if not price:
        log("No price, aborting sell")
        return
    # Check if selling the amount would bring balance below MIN_SOL_THRESHOLD
    remaining_sol = total_sol_balance - amount
    if remaining_sol < MIN_SOL_THRESHOLD:
        amount_to_sell = max(0, total_sol_balance - MIN_SOL_THRESHOLD)  # Adjust to leave at least 0.01 SOL
        log(f"Adjusted sell to {amount_to_sell:.4f} SOL to maintain minimum balance of {MIN_SOL_THRESHOLD:.4f} SOL")
    else:
        amount_to_sell = amount
    amount_sol = int(amount_to_sell * 1e9)
    route = get_route(str(SOL_MINT), str(USDC_MINT), amount_sol)
    if route:
        tx_id, in_amount, out_amount = send_trade(route, price)
        if tx_id:
            sol_sold = in_amount / 1e9
            usdc_received = out_amount / 1e6
            profit = usdc_received - sol_sold * state['entry_price']
            state['position'] -= sol_sold
            state['total_trades'] += 1
            if profit > 0:
                state['wins'] += 1
                state['total_profit'] += profit
                state['recent_trades'].append(profit)
            else:
                state['losses'] += 1
                state['recent_trades'].append(profit)
            if len(state['recent_trades']) > 10:
                state['recent_trades'].pop(0)
            log(f"✅ Sold {sol_sold:.4f} SOL @ ${price:.2f}, Profit: ${profit:.2f}")
            if state['position'] <= 0:
                state['position'] = 0
                state['sell_targets'] = []
                state['highest_price'] = 0

def log_performance(portfolio_value):
    log("Logging performance...")
    try:
        with open('stats.csv', 'a') as f:
            win_rate = state['wins'] / state['total_trades'] * 100 if state['total_trades'] > 0 else 0
            profit_factor = state['total_profit'] / abs(sum(t for t in state['recent_trades'] if t < 0)) if any(t < 0 for t in state['recent_trades']) else float('inf')
            drawdown = (state['peak_portfolio'] - portfolio_value) / state['peak_portfolio'] * 100 if state['peak_portfolio'] > 0 else 0
            f.write(f"{time.time()},{portfolio_value},{state['total_trades']},{state['wins']},{state['losses']},{state['total_profit']},{win_rate},{profit_factor},{drawdown}\n")
        log(f"Stats: Trades={state['total_trades']}, WinRate={win_rate:.2f}%, Profit=${state['total_profit']:.2f}, Drawdown={drawdown:.2f}%")
    except Exception as e:
        log(f"Failed to log performance: {e}")

# Main Loop
def main():
    log("Entering main loop...")
    last_stats_time = time.time()
    while True:
        loop_start = time.time()
        current_time = time.time()
        log("Loop iteration...")

        # Fetch price every 60 seconds
        if current_time - state['last_fetch_time'] >= 60:
            price = fetch_current_price()
            if price:
                state['last_fetch_time'] = current_time
            else:
                price = state['price_history'][-1] if state['price_history'] else None
        else:
            price = state['price_history'][-1] if state['price_history'] else None

        # Calculate indicators
        rsi = calculate_rsi(state['price_history']) if len(state['price_history']) >= 15 else None
        macd_line, signal_line = calculate_macd(state['price_history']) if len(state['price_history']) >= 34 else (None, None)
        vwap = calculate_vwap(state['price_history']) if len(state['price_history']) >= 20 else None
        upper_bb, lower_bb = calculate_bollinger_bands(state['price_history']) if len(state['price_history']) >= 20 else (None, None)
        atr = calculate_atr(state['price_history']) if len(state['price_history']) >= 21 else None
        momentum = calculate_momentum(state['price_history']) if len(state['price_history']) >= 11 else None

        # Calculate avg_atr safely
        if atr is not None:
            state['atr_history'].append(atr)
            if len(state['atr_history']) > 50:
                state['atr_history'].pop(0)
            avg_atr = np.mean(state['atr_history']) if state['atr_history'] else atr
        else:
            avg_atr = None

        # Adjust TRADE_INTERVAL
        global TRADE_INTERVAL
        if atr is not None and avg_atr is not None and avg_atr > 0:
            TRADE_INTERVAL = max(5, min(45, 30 * (avg_atr / atr)))
            if atr > 2 * avg_atr or (rsi is not None and (rsi < 35 or rsi > 68)):
                TRADE_INTERVAL = 5
            elif atr < 0.5 * avg_atr or (rsi is not None and 40 <= rsi <= 60):
                TRADE_INTERVAL = 45
            log(f"TRADE_INTERVAL: {TRADE_INTERVAL}s")

        # Portfolio pause check
        if current_time < state['pause_until']:
            log(f"Paused until {time.strftime('%H:%M:%S', time.localtime(state['pause_until']))}")
            time.sleep(TRADE_INTERVAL)
            continue

        # Update portfolio stats
        portfolio_value = get_portfolio_value(price)
        if portfolio_value > state['peak_portfolio']:
            state['peak_portfolio'] = portfolio_value
        drawdown = (state['peak_portfolio'] - portfolio_value) / state['peak_portfolio'] * 100 if state['peak_portfolio'] > 0 else 0
        log(f"Portfolio: ${portfolio_value:.2f}, Drawdown: {drawdown:.2f}%")
        if drawdown > 10:
            state['pause_until'] = current_time + 36 * 3600
            log("Drawdown >7%, pausing for 36 hours")
            continue
        elif drawdown > 7:
            state['pause_until'] = current_time + 48 * 3600
            log("Drawdown >5%, pausing for 48 hours")
            continue

        # Adjust triggers
        adjust_triggers(atr, avg_atr, rsi)

        # Buy logic
        if state['position'] == 0 and price:
            if check_buy_signal(price, rsi, macd_line, signal_line, vwap, lower_bb, momentum, atr, avg_atr):
                position_size = calculate_position_size(portfolio_value, atr, avg_atr)
                if position_size > 0:
                    execute_buy(position_size)

        # Sell logic
        elif state['position'] > 0 and price:
            if price <= state['entry_price'] * (1 - STOP_LOSS_DROP / 100):
                log("Hit stop-loss, selling position")
                execute_sell(state['position'], price)
            elif price >= state['entry_price'] * 1.035:
                state['highest_price'] = max(state['highest_price'], price)
                if rsi is not None and rsi > 68:  # Direct RSI sell condition
                    log("RSI overbought, selling position")
                    execute_sell(state['position'], price)
                elif price <= state['highest_price'] * (1 - TRAILING_STOP / 100):
                    log("Hit trailing stop, selling position")
                    execute_sell(state['position'], price)
            else:
                sma_slope = (vwap - calculate_vwap(state['price_history'][:-1])) / vwap * 100 if vwap and len(state['price_history']) > 1 else 0
                hold_final = sma_slope > 0.7 and macd_line is not None and signal_line is not None and macd_line > signal_line and (rsi is not None and rsi <= 68)
                for i, (amount, target_price) in enumerate(state['sell_targets'][:]):
                    if price >= target_price and (not hold_final or i < len(state['sell_targets']) - 1):
                        min_profit = 0.02 if portfolio_value < 100 else 1
                        if (price - state['entry_price']) * amount > min_profit:
                            execute_sell(amount, price)
                            del state['sell_targets'][i]
                            break

        # Log performance every 4 hours
        if current_time - last_stats_time >= 4 * 3600:
            log_performance(portfolio_value)
            last_stats_time = current_time

        # Sleep
        elapsed = time.time() - loop_start
        sleep_time = max(0, TRADE_INTERVAL - elapsed)
        log(f"Sleeping for {sleep_time:.1f}s")
        time.sleep(sleep_time)

def tcp_health_check():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 8000))
    server.listen(1)
    log("TCP health check listening on port 8000...")
    while True:
        try:
            client, addr = server.accept()
            client.close()
        except Exception as e:
            log(f"TCP health error: {e}")

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')
        log("Ping received to keep bot awake")

def http_server():
    server = HTTPServer(('0.0.0.0', 80), SimpleHandler)
    log("HTTP server listening on port 80 for pings...")
    server.serve_forever()


if __name__ == "__main__":
    log("Bot initializing...")
    try:
        # Start TCP health check thread
        health_thread = threading.Thread(target=tcp_health_check, daemon=True)
        health_thread.start()
        # Start HTTP server thread for pings
        http_thread = threading.Thread(target=http_server, daemon=True)
        http_thread.start()
        initialize_price_history()
        with open('stats.csv', 'w') as f:
            f.write("timestamp,portfolio_value,total_trades,wins,losses,total_profit,win_rate,profit_factor,drawdown\n")
        while True:
            try:
                main()
            except Exception as e:
                log(f"Main loop crashed, restarting: {e}")
                time.sleep(10)
                initialize_price_history()
                continue
    except KeyboardInterrupt:
        log("Bot stopped by user")
        portfolio_value = get_portfolio_value(state['price_history'][-1] if state['price_history'] else 0)
        log_performance(portfolio_value)
        log(f"Final Stats: Total Return: {state['total_profit'] / portfolio_value * 100:.2f}%, "
            f"Win Rate: {state['wins'] / state['total_trades'] * 100 if state['total_trades'] > 0 else 0:.2f}%, "
            f"Adverse Months: Not Calculated")
    except Exception as e:
        log(f"Bot crashed: {e}")
        raise
