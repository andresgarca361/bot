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
from solders.transaction import VersionedTransaction
from base64 import b64decode
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from solders.message import to_bytes_versioned
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
SLIPPAGE = 0.01
MIN_TRADE_USD = 1.0
MAX_POSITION_SOL = 3.0
MAX_DRAWDOWN = 15.0  # %
MIN_SOL_THRESHOLD = 0.01  # Ensure SOL balance never goes below 0.01 SOL

# Client setup
log("Initializing Solana client...")
client = Client(RPC_URLS[0])
log(f"Connected to RPC: {client._provider.endpoint_uri}")

state = {
    'position': 0.0,
    'entry_price': 0.0,
    'sell_targets': [],
    'highest_price': 0.0,
    'peak_portfolio': 0.0,
    'peak_market_value': 0.0,  # Ensure this is present
    'pause_until': 0,
    'buy_pause_until': 0,
    'sell_pause_until': 0,
    'price_history': [],
    'atr_history': [],
    'last_fetch_time': 0,
    'current_rpc_index': 0,
    'total_trades': 0,
    'wins': 0,
    'losses': 0,
    'total_profit': 0.0,
    'last_fee_update': 0,
    'cached_fee': 0.002,
    'recent_trades': [],
    'last_price': None,
    'last_sol_balance': 0.0,
    'last_usdc_balance': 0.0,
    'last_balance_update': 0,
    'trade_cooldown_until': 0,
    'peak_timestamp': 0,
    'version': "1.0",
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
    max_attempts = 10
    while len(prices) < required_prices and attempts < max_attempts:
        price = fetch_current_price()
        if price:
            prices.append(price)
            log(f"Fetched price {len(prices)}/{required_prices}: ${price:.2f}")
            time.sleep(1)  # Increased delay to avoid rate limits
        else:
            attempts += 1
            log(f"Price fetch failed, attempt {attempts}/{max_attempts}")
            time.sleep(5)  # Longer delay on failure

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
@sleep_and_retry
@limits(calls=90, period=60)
def fetch_current_price():
    log("Fetching USDC/SOL price...")
    current_time = time.time()
    if 'last_price' in state and current_time - state['last_fetch_time'] < 1:  # Reduced from 5 to 1 second
        log(f"Using cached price: ${state['last_price']:.2f}")
        return state['last_price']
        
    url = f"https://quote-api.jup.ag/v6/quote?inputMint={SOL_MINT}&outputMint={USDC_MINT}&amount=1000000000&slippageBps=0"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            out_amount = int(data['outAmount'])
            price = out_amount / 1e6  # USDC per SOL
            state['last_price'] = price
            state['last_fetch_time'] = current_time
            log(f"Price fetched: ${price:.2f}")
            return price
        else:
            log(f"Price fetch failed: Status {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        log(f"Price fetch error: {str(e)}")
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
# Add to state initialization (line 84)


# Update calculate_rsi (line 248)
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        log("Not enough prices for RSI")
        return None
    if len(set(prices[-period-1:])) < 2:  # Check for price variation
        log("Price history has insufficient variation for RSI")
        return None
    relevant_prices = prices[-(period + 1):]
    changes = np.diff(relevant_prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    for i in range(1, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss < 0.0001:
        avg_loss = 0.0001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    log(f"RSI: {rsi:.2f}")
    return rsi



def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal - 1:
        log("Not enough prices for MACD")
        return None, None
    if len(set(prices[-slow-signal+1:])) < 2:  # Check for price variation
        log("Price history has insufficient variation for MACD")
        return None, None
    
    prices = np.array(prices)
    def calculate_ema(prices, period):
        k = 2 / (period + 1)
        ema = prices[0]
        ema_values = [ema]
        for price in prices[1:]:
            ema = (price * k) + (ema * (1 - k))
            ema_values.append(ema)
        return ema_values
    
    ema_fast_values = calculate_ema(prices, fast)
    ema_slow_values = calculate_ema(prices, slow)
    
    if len(ema_fast_values) < slow or len(ema_slow_values) < slow:
        log("Not enough EMA values for MACD")
        return None, None
    
    macd_line = ema_fast_values[-1] - ema_slow_values[-1]
    
    macd_history = []
    for i in range(len(prices) - slow + 1, len(prices)):
        macd_value = ema_fast_values[i] - ema_slow_values[i]
        macd_history.append(macd_value)
    
    if len(macd_history) < signal:
        log("Not enough MACD history for signal line")
        return None, None
    
    signal_values = calculate_ema(np.array(macd_history), signal)
    signal_line = signal_values[-1]
    
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
    if rsi is not None and (rsi < 25 or rsi > 70):  # Changed from 66 to 70
        BASE_BUY_TRIGGER = max(1.5, BASE_BUY_TRIGGER - 0.5)
        BASE_SELL_TRIGGER = max(2.5, BASE_SELL_TRIGGER - 0.5)
        log(f"Extreme RSI, triggers: buy={BASE_BUY_TRIGGER}%, sell={BASE_SELL_TRIGGER}%")

def check_buy_signal(price, rsi, macd_line, signal_line, vwap, lower_bb, momentum, atr, avg_atr):
    if any(x is None for x in [rsi, macd_line, signal_line, vwap, lower_bb, momentum, atr, avg_atr]):
        log("Missing or invalid indicators, skipping buy")
        return False
    if rsi <= 0 or macd_line == 0 or signal_line == 0:  # Prevent invalid calculations
        log(f"Invalid indicator values: RSI={rsi}, MACD={macd_line}, Signal={signal_line}, skipping buy")
        return False
    fee = get_fee_estimate()
    if fee > 0.0035:
        log(f"Fees too high ({fee*100:.2f}%), skipping buy")
        return False
    bid_ask_spread = abs(fetch_current_price() - price) / price if price else 0.01
    spread_condition = bid_ask_spread < 0.005
    if not spread_condition:
        log(f"Bid-ask spread too high ({bid_ask_spread*100:.2f}%), skipping buy")
        return False
    multiplier = 1.1 if signal_line < 0 else 0.9
    if rsi >= 35 or macd_line <= signal_line * multiplier:
        log(f"Core conditions failed: RSI={rsi:.2f}, MACD={macd_line:.4f}<={signal_line:.4f}")
        return False
    momentum_weight = 0.4
    vwap_weight = 0.3
    bb_weight = 0.3
    momentum_score = 1 if momentum > 0.4 else 0
    vwap_score = 1 if price < vwap * 0.995 else 0
    bb_score = 1 if price < lower_bb * 1.01 else 0
    total_score = (momentum_weight * momentum_score) + (vwap_weight * vwap_score) + (bb_weight * bb_score)
    signal = total_score >= 0.3
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

@sleep_and_retry
@limits(calls=90, period=60)
def get_route(from_mint, to_mint, amount):
    log(f"Fetching swap route: {from_mint} -> {to_mint}, amount={amount}")
    url = f"https://quote-api.jup.ag/v6/quote?inputMint={from_mint}&outputMint={to_mint}&amount={int(amount)}&slippageBps={int(SLIPPAGE * 10000)}&swapMode=ExactIn"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            log(f"API response: {data}")
            # Check if response has 'routePlan' directly or nested under 'data'
            route = None
            if isinstance(data, dict):
                if 'routePlan' in data:
                    route = data  # Route is directly in the response
                elif 'data' in data and data['data']:
                    route = data['data'][0]  # Route is nested under 'data'
            if route:
                log("Route found")
                return route
            log("No routes available")
            return None
        log(f"Route fetch failed: Status {response.status_code}, Response: {response.text}")
        return None
    except Exception as e:
        log(f"Route fetch error: {e}, Response: {response.text if 'response' in locals() else 'No response'}")
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
    
    # Step 1: Send request to Jupiter API
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            log(f"Trade failed: Status {response.status_code}, Response: {response.text}")
            return None, 0, 0
        data = response.json()
        if 'swapTransaction' not in data:
            log(f"Missing swapTransaction in response: {data}")
            return None, 0, 0
        tx_raw = data['swapTransaction']
    except Exception as e:
        log(f"API request failed: {e}")
        return None, 0, 0

    # Step 2: Decode the raw transaction
    try:
        tx_data = b64decode(tx_raw)
    except Exception as e:
        log(f"Failed to decode transaction: {e}, Raw data: {tx_raw}")
        return None, 0, 0

    # Step 3: Parse into VersionedTransaction
    try:
        tx = VersionedTransaction.from_bytes(tx_data)
    except Exception as e:
        log(f"Failed to parse transaction: {e}")
        return None, 0, 0

    # Step 4: Serialize message for signing using to_bytes_versioned
    try:
        message_bytes = to_bytes_versioned(tx.message)  # Correct method for solders==0.21.0
    except Exception as e:
        log(f"Message serialization failed: {e}")
        return None, 0, 0

    # Step 5: Sign the message
    try:
        signature = keypair.sign_message(message_bytes)
    except Exception as e:
        log(f"Signing failed: {e}")
        return None, 0, 0

    # Step 6: Attach signature to transaction
    try:
        tx.signatures = [signature]
    except Exception as e:
        log(f"Failed to set signature: {e}")
        return None, 0, 0

    # Step 7: Serialize signed transaction
    try:
        signed_tx_data = bytes(tx)
    except Exception as e:
        log(f"Transaction serialization failed: {e}")
        return None, 0, 0

    # Step 8: Send transaction
    try:
        result = client.send_raw_transaction(signed_tx_data, opts=TxOpts(skip_preflight=False))
        tx_id = result.value
        log(f"Trade sent: tx_id={tx_id}")
        time.sleep(10)  # Increased from 5 to 10 seconds for network sync
        in_amount = int(route['inAmount'])
        out_amount = int(route['outAmount'])
        return tx_id, in_amount, out_amount
    except Exception as e:
        log(f"Transaction submission failed: {e}")
        return None, 0, 0

def execute_buy(position_size):
    log(f"Executing buy: {position_size:.4f} SOL")
    price = fetch_current_price()
    if not price:
        log("No price, aborting buy")
        return
    input_amount_usdc = int(position_size * price * 1e6)  # Convert to USDC lamports (ExactIn)
    log(f"Buying with {input_amount_usdc} USDC lamports (~${input_amount_usdc / 1e6:.2f})")
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
            save_state()
            set_sell_targets(sol_bought, price)
            time.sleep(10)

def set_sell_targets(position_size, entry_price):
    log(f"Setting sell targets for {position_size:.4f} SOL")
    if position_size < 1:
        state['sell_targets'] = [
            (position_size * 0.5, entry_price * 1.02),  # 50% at 2% above entry
            (position_size * 0.3, entry_price * 1.05),  # 30% at 5% above entry
            (position_size * 0.2, entry_price * 1.08)   # 20% at 8% above entry
        ]
    else:
        state['sell_targets'] = [
            (position_size * 0.4, entry_price * 1.02),  # 40% at 2% above entry
            (position_size * 0.4, entry_price * 1.05),  # 40% at 5% above entry
            (position_size * 0.2, entry_price * 1.08)   # 20% at 8% above entry
        ]
    log(f"Sell targets: {state['sell_targets']}")

def execute_sell(amount, price):
    log(f"Executing sell: {amount:.4f} SOL @ ${price:.2f}")
    total_sol_balance = get_sol_balance()  # Fetch total SOL balance
    if not price:
        log("No price, aborting sell")
        return
    remaining_sol = total_sol_balance - amount
    if remaining_sol < MIN_SOL_THRESHOLD:
        amount_to_sell = max(0, total_sol_balance - MIN_SOL_THRESHOLD)
        log(f"Adjusted sell to {amount_to_sell:.4f} SOL to maintain minimum balance of {MIN_SOL_THRESHOLD:.4f} SOL")
    else:
        amount_to_sell = amount
    amount_sol = int(amount_to_sell * 1e9)  # Convert SOL to lamports
    log(f"Selling {amount_sol} SOL lamports (~${amount_to_sell * price:.2f})")
    route = get_route(str(SOL_MINT), str(USDC_MINT), amount_sol)
    if route:
        tx_id, in_amount, out_amount = send_trade(route, price)
        if tx_id:
            sol_sold = in_amount / 1e9
            usdc_received = out_amount / 1e6
            fee = get_fee_estimate()
            fee_amount = usdc_received * fee
            profit = (usdc_received - fee_amount) - (sol_sold * state['entry_price'])
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
            save_state()
            if state['position'] <= 0:
                state['position'] = 0
                state['sell_targets'] = []
                state['highest_price'] = 0
                time.sleep(10)

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

def save_state():
    try:
        with open('state.json', 'w') as f:
            json.dump(state, f)
    except Exception as e:
        log(f"Failed to save state: {e}")

def load_state():
    try:
        if os.path.exists('state.json'):
            with open('state.json', 'r') as f:
                state.update(json.load(f))
    except Exception as e:
        log(f"Failed to load state: {e}")


def main():
    global TRADE_INTERVAL
    log("Entering main loop...")
    if 'peak_timestamp' not in state:
        state['peak_timestamp'] = time.time()
    if 'trade_cooldown_until' not in state:
        state['trade_cooldown_until'] = 0
    save_state()

    # Validate and reset state on startup
    current_time = time.time()
    last_save_time = os.path.getmtime('state.json') if os.path.exists('state.json') else 0
    if last_save_time > 0 and current_time - last_save_time > 48 * 3600:
        log(f"State file older than 48 hours, resetting")
    price = fetch_current_price()
    if price:
        portfolio_value = get_portfolio_value(price)
        if state.get('peak_portfolio', 0) == 0 or abs(state['peak_portfolio'] - portfolio_value) / portfolio_value > 0.5:
            log(f"Resetting peak_portfolio from ${state.get('peak_portfolio', 0):.2f} to ${portfolio_value:.2f}")
            state['peak_portfolio'] = portfolio_value
            state['peak_timestamp'] = current_time
        if state.get('pause_until', 0) > current_time and (current_time - state['peak_timestamp'] > 6 * 3600 or last_save_time == 0):
            log(f"Clearing stale pause_until {state.get('pause_until', 0)} as it’s outdated")
            state['pause_until'] = 0
        save_state()

    if current_time - state['peak_timestamp'] > 604800:
        log(f"Peak portfolio ${state['peak_portfolio']:.2f} is over 7 days old, resetting")
        if price is None:
            price = state.get('last_price', state['price_history'][-1] if state['price_history'] else None)
        if price:
            portfolio_value = get_portfolio_value(price)
            state['peak_portfolio'] = portfolio_value
            state['peak_timestamp'] = current_time
            save_state()

    last_stats_time = time.time()
    last_indicator_time = 0
    cached_rsi = None
    cached_macd_line = None
    cached_signal_line = None
    cached_vwap = None
    cached_upper_bb = None
    cached_lower_bb = None
    cached_atr = None
    cached_momentum = None
    cached_avg_atr = None
    last_sol_balance = state.get('last_sol_balance', 0.0)
    last_usdc_balance = state.get('last_usdc_balance', 0.0)
    peak_market_value = state.get('peak_market_value', 0.0)

    def get_updated_portfolio(price, max_retries=3, wait_time=3):
        for attempt in range(max_retries):
            sol_balance = get_sol_balance()
            usdc_balance = get_usdc_balance()
            if sol_balance is not None and usdc_balance is not None:
                portfolio = usdc_balance + (sol_balance * price)
                expected_portfolio = last_usdc_balance + (last_sol_balance * price)
                if abs(portfolio - expected_portfolio) > 0.1 and attempt < max_retries - 1:  # Allow small variance
                    log(f"Portfolio mismatch: Expected ${expected_portfolio:.2f}, got ${portfolio:.2f}, retrying (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)  # Wait for network sync
                    continue
                log(f"Portfolio fetch: SOL={sol_balance:.4f}, USDC={usdc_balance:.4f}, Value=${portfolio:.2f}")
                return portfolio, sol_balance, usdc_balance
        log(f"Failed to fetch consistent portfolio after {max_retries} attempts")
        return None, None, None

    while True:
        try:
            loop_start = time.time()
            current_time = time.time()
            log("Loop iteration...")

            max_pause_duration = 48 * 3600
            if current_time < state['pause_until']:
                remaining_pause = state['pause_until'] - current_time
                if state['pause_until'] - state['peak_timestamp'] > max_pause_duration:
                    log(f"Pause exceeded max duration of 48 hours, resuming")
                    state['pause_until'] = 0
                    if price:
                        portfolio_value = get_portfolio_value(price)
                        state['peak_portfolio'] = portfolio_value
                        state['peak_market_value'] = portfolio_value
                        state['peak_timestamp'] = current_time
                        log(f"Reset peak_portfolio and peak_market_value to ${portfolio_value:.2f} after long pause")
                    save_state()
                else:
                    rsi = calculate_rsi(state['price_history']) if len(state['price_history']) >= 15 else None
                    if rsi is not None and rsi > 50 and remaining_pause > 6 * 3600:
                        state['pause_until'] = current_time + 6 * 3600
                        log("RSI > 50, reducing pause to 6 hours")
                        save_state()
                    log(f"Paused until {time.strftime('%H:%M:%S', time.localtime(state['pause_until']))}")
                    time.sleep(TRADE_INTERVAL)
                    continue
            else:
                if state['pause_until'] != 0:
                    log("Pause time passed, resuming trading")
                    state['pause_until'] = 0
                    if price:
                        portfolio_value = get_portfolio_value(price)
                        state['peak_portfolio'] = portfolio_value
                        state['peak_market_value'] = portfolio_value
                        state['peak_timestamp'] = current_time
                        log(f"Reset peak_portfolio and peak_market_value to ${portfolio_value:.2f} after pause ended")
                    save_state()

            price = fetch_current_price()
            if price is None:
                if not state['price_history']:
                    log("No price history available, cannot proceed without a price")
                    time.sleep(TRADE_INTERVAL)
                    continue
                price = state['price_history'][-1]
                log(f"Price fetch failed, using last known price from history: ${price:.2f}")
            state['last_price'] = price
            state['price_history'].append(price)
            if len(state['price_history']) > 200:
                state['price_history'].pop(0)
            state['last_fetch_time'] = current_time
            save_state()

            try:
                with open('price_history.json', 'w') as f:
                    json.dump({'prices': state['price_history'], 'timestamp': time.time()}, f)
            except Exception as e:
                log(f"Failed to save price history: {e}")

            if len(state['price_history']) < 34 or len(set(state['price_history'][-34:])) < 2:
                log(f"Waiting for price data: {len(state['price_history'])}/34 prices collected, or insufficient variation")
                log(f"Last 34 prices: {[round(p, 2) for p in state['price_history'][-34:]]}")
                time.sleep(TRADE_INTERVAL)
                continue

            if current_time - last_indicator_time >= 60 or any(x is None for x in [cached_rsi, cached_macd_line, cached_signal_line, cached_vwap, cached_upper_bb, cached_lower_bb, cached_atr, cached_momentum, cached_avg_atr]):
                rsi = calculate_rsi(state['price_history'])
                macd_line, signal_line = calculate_macd(state['price_history'])
                vwap = calculate_vwap(state['price_history'])
                upper_bb, lower_bb = calculate_bollinger_bands(state['price_history'])
                atr = calculate_atr(state['price_history'])
                momentum = calculate_momentum(state['price_history'])
                if atr is not None:
                    state['atr_history'].append(atr)
                    if len(state['atr_history']) > 50:
                        state['atr_history'].pop(0)
                    avg_atr = np.mean(state['atr_history']) if state['atr_history'] else atr
                else:
                    avg_atr = 2.0
                cached_rsi, cached_macd_line, cached_signal_line = rsi, macd_line, signal_line
                cached_vwap, cached_upper_bb, cached_lower_bb = vwap, upper_bb, lower_bb
                cached_atr, cached_momentum, cached_avg_atr = atr, momentum, avg_atr
                last_indicator_time = current_time
            else:
                rsi, macd_line, signal_line = cached_rsi, cached_macd_line, cached_signal_line
                vwap, upper_bb, lower_bb = cached_vwap, cached_upper_bb, cached_lower_bb
                atr, momentum, avg_atr = cached_atr, cached_momentum, cached_avg_atr

            if any(x is None for x in [rsi, macd_line, signal_line, vwap, lower_bb, momentum, atr, avg_atr]):
                log("Indicators not ready or invalid, skipping trade logic")
                time.sleep(TRADE_INTERVAL)
                continue

            cooldown_duration = 1800
            if atr is not None and avg_atr is not None and avg_atr > 0:
                TRADE_INTERVAL = max(5, min(45, 30 * (avg_atr / atr)))
                if atr > 2 * avg_atr or (rsi is not None and (rsi < 35 or rsi > 66)):
                    TRADE_INTERVAL = 4
                    cooldown_duration = 900
                elif atr < 0.5 * avg_atr or (rsi is not None and 40 <= rsi <= 60):
                    TRADE_INTERVAL = 45
                log(f"TRADE_INTERVAL: {TRADE_INTERVAL}s, Cooldown: {cooldown_duration//60} min")
            else:
                log(f"TRADE_INTERVAL: {TRADE_INTERVAL}s (default), Cooldown: {cooldown_duration//60} min")

            portfolio_value, sol_balance, usdc_balance = get_updated_portfolio(price, wait_time=10)
            if portfolio_value is None:
                log("Skipping iteration due to portfolio fetch failure")
                time.sleep(TRADE_INTERVAL)
                continue
            if sol_balance is not None and usdc_balance is not None:
                if abs(sol_balance - last_sol_balance) > 0.0001 or abs(usdc_balance - last_usdc_balance) > 0.01:
                    log(f"Balance changed: SOL {last_sol_balance:.4f} -> {sol_balance:.4f}, USDC {last_usdc_balance:.4f} -> {usdc_balance:.4f}")
                last_sol_balance = sol_balance
                last_usdc_balance = usdc_balance
                state['last_sol_balance'] = sol_balance
                state['last_usdc_balance'] = usdc_balance
                save_state()

            if sol_balance is not None and price and len(state['price_history']) >= 10:
                peak_price = max(state['price_history'][-100:])
                current_market_value = usdc_balance + (sol_balance * price)
                peak_market_value_with_current_holdings = usdc_balance + (sol_balance * peak_price)
                if peak_market_value_with_current_holdings > peak_market_value:
                    peak_market_value = peak_market_value_with_current_holdings
                    state['peak_market_value'] = peak_market_value
                market_drawdown = (peak_market_value - current_market_value) / peak_market_value * 100 if peak_market_value > 0 else 0
                market_drawdown_usd = peak_market_value - current_market_value
            else:
                market_drawdown = 0
                market_drawdown_usd = 0

            if portfolio_value > 0:
                if state['peak_portfolio'] == 0 or abs(state['peak_portfolio'] - portfolio_value) / portfolio_value > 0.5:
                    state['peak_portfolio'] = portfolio_value
                    state['peak_timestamp'] = current_time
                    state['peak_market_value'] = portfolio_value
                    log(f"Reset peak_portfolio and peak_market_value to current value: ${portfolio_value:.2f}")
                elif portfolio_value > state['peak_portfolio']:
                    state['peak_portfolio'] = portfolio_value
                    state['peak_timestamp'] = current_time
                    state['peak_market_value'] = portfolio_value
                    log(f"Peak portfolio updated to ${portfolio_value:.2f}")
            save_state()

            log(f"Portfolio: ${portfolio_value:.2f}, Market Drawdown: {market_drawdown:.2f}% (${market_drawdown_usd:.2f})")
            atr_adjust = atr * 2 if atr else 0
            pause_threshold = max(10, min(20, 10 + (portfolio_value * 0.0001) + atr_adjust))
            min_usd_loss = 5.0
            price_momentum = (price - state['price_history'][-10]) / state['price_history'][-10] * 100 if len(state['price_history']) >= 10 else 0
            log(f"Pause check: market_drawdown={market_drawdown:.2f}%, pause_threshold={pause_threshold:.2f}%, market_drawdown_usd=${market_drawdown_usd:.2f}, min_usd_loss=${min_usd_loss:.2f}, price_momentum={price_momentum:.2f}%")
            if market_drawdown > pause_threshold and market_drawdown_usd >= min_usd_loss and price_momentum < 2.0:
                pause_duration = 6 * 3600 if rsi and rsi > 50 else 48 * 3600
                state['pause_until'] = current_time + pause_duration
                state['peak_timestamp'] = current_time
                log(f"Pausing due to market drawdown: {market_drawdown:.2f}% > {pause_threshold:.2f}% and loss ${market_drawdown_usd:.2f} >= ${min_usd_loss:.2f}, pausing for {pause_duration/3600} hours")
                save_state()
                continue

            adjust_triggers(atr, avg_atr, rsi)
            if not price:
                log("No valid price, skipping trade logic")
                time.sleep(TRADE_INTERVAL)
                continue

            total_sol_balance = get_sol_balance()
            if total_sol_balance is None:
                log("Failed to fetch SOL balance, skipping trade logic")
                time.sleep(TRADE_INTERVAL)
                continue

            if current_time >= state['trade_cooldown_until']:
                if check_buy_signal(price, rsi, macd_line, signal_line, vwap, lower_bb, momentum, atr, avg_atr):
                    position_size = calculate_position_size(portfolio_value, atr, avg_atr)
                    if position_size > 0:
                        new_position = state['position'] + position_size
                        if new_position <= MAX_POSITION_SOL:
                            old_sol_balance = sol_balance
                            old_usdc_balance = usdc_balance
                            cost = position_size * price * (1 + 0.002)  # Include fee
                            execute_buy(position_size)
                            time.sleep(10)  # Wait for network sync
                            portfolio_value_after, sol_balance_after, usdc_balance_after = get_updated_portfolio(price, wait_time=10)
                            if portfolio_value_after is None:
                                log("Trade failed: Unable to fetch portfolio after buy, skipping update")
                                time.sleep(TRADE_INTERVAL)
                                continue
                            if sol_balance_after is not None and usdc_balance_after is not None:
                                expected_sol = old_sol_balance + (position_size * (1 - 0.002))
                                expected_usdc = old_usdc_balance - cost
                                if abs(sol_balance_after - expected_sol) > 0.0001 or abs(usdc_balance_after - expected_usdc) > 0.1:
                                    log(f"Trade mismatch: Expected SOL {expected_sol:.4f}, got {sol_balance_after:.4f}; Expected USDC {expected_usdc:.2f}, got {usdc_balance_after:.2f}")
                                    log("Skipping trade update due to mismatch")
                                    time.sleep(TRADE_INTERVAL)
                                    continue
                                state['position'] += position_size
                                state['trade_cooldown_until'] = current_time + cooldown_duration
                                last_sol_balance = sol_balance_after
                                last_usdc_balance = usdc_balance_after
                                state['last_sol_balance'] = sol_balance_after
                                state['last_usdc_balance'] = usdc_balance_after
                                log(f"New position: {state['position']:.4f} SOL, cooldown for {cooldown_duration//60} min")
                                save_state()
                        else:
                            log(f"Cannot buy: New position {new_position:.4f} SOL exceeds max {MAX_POSITION_SOL} SOL")
            else:
                log(f"Trade on cooldown until {time.strftime('%H:%M:%S', time.localtime(state['trade_cooldown_until']))}")

            if current_time >= state['trade_cooldown_until']:
                if total_sol_balance > MIN_SOL_THRESHOLD and price:
                    multiplier = 1.1 if macd_line < 0 else 0.9
                    if (rsi is not None and rsi > 70) and \
                       (macd_line is not None and signal_line is not None and macd_line < signal_line * multiplier and signal_line < 0):
                        amount_to_sell = min(total_sol_balance - MIN_SOL_THRESHOLD, total_sol_balance * 0.1)
                        if amount_to_sell > 0:
                            log(f"Selling due to RSI > 70 or bearish MACD ({macd_line:.4f} < {signal_line:.4f} * {multiplier})")
                            if state['position'] == 0:
                                state['entry_price'] = price
                            old_sol_balance = sol_balance
                            old_usdc_balance = usdc_balance
                            execute_sell(amount_to_sell, price)
                            time.sleep(10)  # Wait for network sync
                            portfolio_value_after, sol_balance_after, usdc_balance_after = get_updated_portfolio(price, wait_time=10)
                            if portfolio_value_after is None:
                                log("Trade failed: Unable to fetch portfolio after sell, skipping update")
                                time.sleep(TRADE_INTERVAL)
                                continue
                            if sol_balance_after is not None and usdc_balance_after is not None:
                                expected_sol = old_sol_balance - amount_to_sell
                                expected_usdc = old_usdc_balance + (amount_to_sell * price * (1 - 0.002))
                                if abs(sol_balance_after - expected_sol) > 0.0001 or abs(usdc_balance_after - expected_usdc) > 0.1:
                                    log(f"Trade mismatch: Expected SOL {expected_sol:.4f}, got {sol_balance_after:.4f}; Expected USDC {expected_usdc:.2f}, got {usdc_balance_after:.2f}")
                                    log("Skipping trade update due to mismatch")
                                    time.sleep(TRADE_INTERVAL)
                                    continue
                            state['trade_cooldown_until'] = current_time + cooldown_duration
                            last_sol_balance = sol_balance_after
                            last_usdc_balance = usdc_balance_after
                            state['last_sol_balance'] = sol_balance_after
                            state['last_usdc_balance'] = usdc_balance_after
                            new_portfolio_value = get_portfolio_value(price)
                            state['peak_portfolio'] = new_portfolio_value
                            state['peak_market_value'] = new_portfolio_value
                            state['peak_timestamp'] = current_time
                            log(f"Reset peak_portfolio and peak_market_value to ${new_portfolio_value:.2f} after RSI or MACD sell")
                            save_state()
                    elif state['position'] > 0 and price:
                        if price <= state['entry_price'] * (1 - STOP_LOSS_DROP / 100):
                            log("Hit stop-loss, selling")
                            old_sol_balance = sol_balance
                            old_usdc_balance = usdc_balance
                            execute_sell(state['position'], price)
                            time.sleep(10)  # Wait for network sync
                            portfolio_value_after, sol_balance_after, usdc_balance_after = get_updated_portfolio(price, wait_time=10)
                            if portfolio_value_after is None:
                                log("Trade failed: Unable to fetch portfolio after sell, skipping update")
                                time.sleep(TRADE_INTERVAL)
                                continue
                            if sol_balance_after is not None and usdc_balance_after is not None:
                                expected_sol = old_sol_balance - state['position']
                                expected_usdc = old_usdc_balance + (state['position'] * price * (1 - 0.002))
                                if abs(sol_balance_after - expected_sol) > 0.0001 or abs(usdc_balance_after - expected_usdc) > 0.1:
                                    log(f"Trade mismatch: Expected SOL {expected_sol:.4f}, got {sol_balance_after:.4f}; Expected USDC {expected_usdc:.2f}, got {usdc_balance_after:.2f}")
                                    log("Skipping trade update due to mismatch")
                                    time.sleep(TRADE_INTERVAL)
                                    continue
                            state['trade_cooldown_until'] = current_time + cooldown_duration
                            last_sol_balance = sol_balance_after
                            last_usdc_balance = usdc_balance_after
                            state['last_sol_balance'] = sol_balance_after
                            state['last_usdc_balance'] = usdc_balance_after
                            new_portfolio_value = get_portfolio_value(price)
                            state['peak_portfolio'] = new_portfolio_value
                            state['peak_market_value'] = new_portfolio_value
                            state['peak_timestamp'] = current_time
                            log(f"Reset peak_portfolio and peak_market_value to ${new_portfolio_value:.2f} after stop-loss")
                            save_state()
                        elif price >= state['entry_price'] * 1.035:
                            state['highest_price'] = max(state['highest_price'], price)
                            if rsi is not None and rsi > 70:
                                log("RSI overbought, selling")
                                old_sol_balance = sol_balance
                                old_usdc_balance = usdc_balance
                                execute_sell(state['position'], price)
                                time.sleep(10)  # Wait for network sync
                                portfolio_value_after, sol_balance_after, usdc_balance_after = get_updated_portfolio(price, wait_time=10)
                                if portfolio_value_after is None:
                                    log("Trade failed: Unable to fetch portfolio after sell, skipping update")
                                    time.sleep(TRADE_INTERVAL)
                                    continue
                                if sol_balance_after is not None and usdc_balance_after is not None:
                                    expected_sol = old_sol_balance - state['position']
                                    expected_usdc = old_usdc_balance + (state['position'] * price * (1 - 0.002))
                                    if abs(sol_balance_after - expected_sol) > 0.0001 or abs(usdc_balance_after - expected_usdc) > 0.1:
                                        log(f"Trade mismatch: Expected SOL {expected_sol:.4f}, got {sol_balance_after:.4f}; Expected USDC {expected_usdc:.2f}, got {usdc_balance_after:.2f}")
                                        log("Skipping trade update due to mismatch")
                                        time.sleep(TRADE_INTERVAL)
                                        continue
                                state['trade_cooldown_until'] = current_time + cooldown_duration
                                last_sol_balance = sol_balance_after
                                last_usdc_balance = usdc_balance_after
                                state['last_sol_balance'] = sol_balance_after
                                state['last_usdc_balance'] = usdc_balance_after
                                new_portfolio_value = get_portfolio_value(price)
                                state['peak_portfolio'] = new_portfolio_value
                                state['peak_market_value'] = new_portfolio_value
                                state['peak_timestamp'] = current_time
                                log(f"Reset peak_portfolio and peak_market_value to ${new_portfolio_value:.2f} after RSI sell")
                                save_state()
                            elif price <= state['highest_price'] * (1 - TRAILING_STOP / 100):
                                log("Hit trailing stop, selling")
                                old_sol_balance = sol_balance
                                old_usdc_balance = usdc_balance
                                execute_sell(state['position'], price)
                                time.sleep(10)  # Wait for network sync
                                portfolio_value_after, sol_balance_after, usdc_balance_after = get_updated_portfolio(price, wait_time=10)
                                if portfolio_value_after is None:
                                    log("Trade failed: Unable to fetch portfolio after sell, skipping update")
                                    time.sleep(TRADE_INTERVAL)
                                    continue
                                if sol_balance_after is not None and usdc_balance_after is not None:
                                    expected_sol = old_sol_balance - state['position']
                                    expected_usdc = old_usdc_balance + (state['position'] * price * (1 - 0.002))
                                    if abs(sol_balance_after - expected_sol) > 0.0001 or abs(usdc_balance_after - expected_usdc) > 0.1:
                                        log(f"Trade mismatch: Expected SOL {expected_sol:.4f}, got {sol_balance_after:.4f}; Expected USDC {expected_usdc:.2f}, got {usdc_balance_after:.2f}")
                                        log("Skipping trade update due to mismatch")
                                        time.sleep(TRADE_INTERVAL)
                                        continue
                                state['trade_cooldown_until'] = current_time + cooldown_duration
                                last_sol_balance = sol_balance_after
                                last_usdc_balance = usdc_balance_after
                                state['last_sol_balance'] = sol_balance_after
                                state['last_usdc_balance'] = usdc_balance_after
                                new_portfolio_value = get_portfolio_value(price)
                                state['peak_portfolio'] = new_portfolio_value
                                state['peak_market_value'] = new_portfolio_value
                                state['peak_timestamp'] = current_time
                                log(f"Reset peak_portfolio and peak_market_value to ${new_portfolio_value:.2f} after trailing stop")
                                save_state()
                        else:
                            sma_slope = (vwap - calculate_vwap(state['price_history'][:-1])) / vwap * 100 if vwap and len(state['price_history']) > 1 else 0
                            hold_final = sma_slope > 0.7 and macd_line is not None and signal_line is not None and macd_line > signal_line and (rsi is not None and rsi <= 70)
                            for i, (amount, target_price) in enumerate(state['sell_targets'][:]):
                                if price >= target_price and (not hold_final or i < len(state['sell_targets']) - 1):
                                    min_profit = 0.02 if portfolio_value < 100 else 1
                                    if (price - state['entry_price']) * amount > min_profit:
                                        old_sol_balance = sol_balance
                                        old_usdc_balance = usdc_balance
                                        execute_sell(amount, price)
                                        time.sleep(10)  # Wait for network sync
                                        portfolio_value_after, sol_balance_after, usdc_balance_after = get_updated_portfolio(price, wait_time=10)
                                        if portfolio_value_after is None:
                                            log("Trade failed: Unable to fetch portfolio after sell, skipping update")
                                            time.sleep(TRADE_INTERVAL)
                                            continue
                                        if sol_balance_after is not None and usdc_balance_after is not None:
                                            expected_sol = old_sol_balance - amount
                                            expected_usdc = old_usdc_balance + (amount * price * (1 - 0.002))
                                            if abs(sol_balance_after - expected_sol) > 0.0001 or abs(usdc_balance_after - expected_usdc) > 0.1:
                                                log(f"Trade mismatch: Expected SOL {expected_sol:.4f}, got {sol_balance_after:.4f}; Expected USDC {expected_usdc:.2f}, got {usdc_balance_after:.2f}")
                                                log("Skipping trade update due to mismatch")
                                                time.sleep(TRADE_INTERVAL)
                                                continue
                                        state['trade_cooldown_until'] = current_time + cooldown_duration
                                        last_sol_balance = sol_balance_after
                                        last_usdc_balance = usdc_balance_after
                                        state['last_sol_balance'] = sol_balance_after
                                        state['last_usdc_balance'] = usdc_balance_after
                                        new_portfolio_value = get_portfolio_value(price)
                                        state['peak_portfolio'] = new_portfolio_value
                                        state['peak_market_value'] = new_portfolio_value
                                        state['peak_timestamp'] = current_time
                                        log(f"Reset peak_portfolio and peak_market_value to ${new_portfolio_value:.2f} after target sell")
                                        save_state()
                                        del state['sell_targets'][i]
                                        break

            if current_time - last_stats_time >= 4 * 3600:
                log_performance(portfolio_value)
                last_stats_time = current_time
                save_state()

            elapsed = time.time() - loop_start
            sleep_time = max(0, TRADE_INTERVAL - elapsed)
            log(f"Sleeping for {sleep_time:.1f}s")
            time.sleep(sleep_time)
        except Exception as e:
            log(f"Error in main loop, continuing: {e}")
            time.sleep(TRADE_INTERVAL)
            continue
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
        health_thread = threading.Thread(target=tcp_health_check, daemon=True)
        health_thread.start()
        http_thread = threading.Thread(target=http_server, daemon=True)
        http_thread.start()
        load_state()  # Load state
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
        save_state()  # Save state on shutdown
        log(f"Final Stats: Total Return: {state['total_profit'] / portfolio_value * 100:.2f}%, "
            f"Win Rate: {state['wins'] / state['total_trades'] * 100 if state['total_trades'] > 0 else 0:.2f}%, "
            f"Adverse Months: Not Calculated")
    except Exception as e:
        log(f"Bot crashed: {e}")
        raise
