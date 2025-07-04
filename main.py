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
TRADE_INTERVAL = 60  # Adjusted dynamically
BASE_BUY_TRIGGER = 2.0  # %, adjusted dynamically
BASE_SELL_TRIGGER = 3.0  # %, adjusted dynamically
STOP_LOSS_DROP = 5.0  # %
TRAILING_STOP = 1.5  # %, activated at 3.5% profit
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
    'peak_market_value': 0.0,
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
    'buy_orders': []  # Added
}




@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def initialize_price_history():
    log("Initializing price history with CryptoCompare...")
    price_file = 'price_history.json'
    required_prices = 34  # Enough for 20-period RSI + buffer

    # Load recent cached prices if available
    if os.path.exists(price_file):
        try:
            with open(price_file, 'r') as f:
                data = json.load(f)
                prices = data.get('prices', [])
                timestamp = data.get('timestamp', 0)
                if time.time() - timestamp < 3600 and len(prices) >= required_prices:
                    state['price_history'] = prices[-required_prices:]
                    log(f"Loaded {len(state['price_history'])} prices from file")
                    return
                else:
                    log("Price history outdated or insufficient, fetching from CryptoCompare")
        except Exception as e:
            log(f"Failed to load price history: {e}")

    # Fetch historical prices from CryptoCompare
    API_KEY = os.getenv("CMC_KEY")
    if not API_KEY:
        log("ERROR: CMC_KEY (used as CryptoCompare API key) is not set")
        raise RuntimeError("CryptoCompare API key not found")

    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {
        "fsym": "SOL",
        "tsym": "USD",
        "limit": required_prices - 1,
        "api_key": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        log(f"CryptoCompare response status: {response.status_code}, Text length: {len(response.text)}...")
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") != "Success":
                log(f"CryptoCompare returned error: {data.get('Message')}")
                raise RuntimeError("CryptoCompare response not successful")

            candles = data.get("Data", {}).get("Data", [])
            if len(candles) < required_prices:
                log(f"Insufficient data from CryptoCompare: {len(candles)}/{required_prices}")
                raise RuntimeError("Not enough historical data from CryptoCompare")

            closes = [c["close"] for c in candles if c.get("close") is not None]
            if len(closes) < required_prices:
                log(f"Insufficient valid closes: {len(closes)}/{required_prices}")
                raise RuntimeError("Invalid or incomplete close data")

            state['price_history'] = closes[-required_prices:]
            log(f"Initialized {len(state['price_history'])} prices from CryptoCompare")

            # ✅ SAVE TO FILE
            try:
                with open(price_file, 'w') as f:
                    json.dump({'prices': state['price_history'], 'timestamp': time.time()}, f)
                log("Saved price history to file")
            except Exception as e:
                log(f"Failed to save price history: {e}")

            return  # ✅ PREVENT FALLBACK ON SUCCESS
        else:
            log(f"CryptoCompare request failed: Status {response.status_code}, Response: {response.text}")
            raise RuntimeError("CryptoCompare request failed")
    except Exception as e:
        log(f"Error fetching price data from CryptoCompare: {e}")

    # ⛔ FALLBACK to Jupiter API
    log("Falling back to Jupiter API...")
    prices = []
    attempts = 0
    max_attempts = 10
    while len(prices) < required_prices and attempts < max_attempts:
        price = fetch_current_price()
        if price:
            prices.append(price)
            log(f"Fetched price {len(prices)}/{required_prices}: ${price:.2f}")
            time.sleep(60)
        else:
            attempts += 1
            log(f"Price fetch failed, attempt {attempts}/{max_attempts}")
            time.sleep(60)

    if len(prices) < required_prices:
        log(f"ERROR: Could not fetch enough prices, got {len(prices)}/{required_prices}")
        raise RuntimeError("Failed to initialize price history")

    state['price_history'] = prices[-required_prices:]

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
    return 0.002

# Indicator Functions
def get_current_rsi(prices, period=14):
    if len(prices) < period or len(set(prices)) < 2:
        return 50.0  # Silently return 50.0, no spam logging
    prices = np.array(prices)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0 and avg_gain == 0:
        return 50.0  # No movement, still silent
    if avg_loss == 0:
        avg_loss = 0.0001  # Avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    log(f"RSI calculated: {rsi:.2f} with period {period}")
    return rsi
    
def calculate_rsi(prices, period=20):  # Increased from 14 to 20
    if len(prices) < period + 1:
        log("Not enough prices for RSI calculation")
        return None

    prices = np.array(prices)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Initial 20-period SMA for gains and losses
    avg_gain = np.mean(gains[:period]) if period <= len(gains) else 0
    avg_loss = np.mean(losses[:period]) if period <= len(losses) else 0

    # Wilder’s smoothing (1/20 factor) for remaining periods
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Prevent division by zero
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

def check_buy_signal(price, rsi, macd_line, signal_line, vwap, lower_bb, momentum, atr, avg_atr, timeframe):  # Added timeframe parameter
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
    if timeframe != 'eagle' and rsi >= 35 or macd_line <= signal_line * multiplier:  # Bypass rsi >= 35 for Eagle
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
            state['position'] += sol_bought  # Changed to accumulate position
            state['entry_price'] = price
            state['highest_price'] = price
            state['total_trades'] += 1
            log(f"✅ Bought {sol_bought:.4f} SOL @ ${price:.2f}, Total Position: {state['position']:.4f} SOL")
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
    global TRADE_INTERVAL, MAX_POSITION_SOL, BASE_BUY_TRIGGER, BASE_SELL_TRIGGER
    MAX_POSITION_SOL = 25.0  # Capacity for massive uptrends
    TRADE_INTERVAL = 30  # Adjusted to 30s for EagleEye
    BASE_BUY_TRIGGER = 3.0  # Base value, overridden per timeframe
    BASE_SELL_TRIGGER = 3.0  # Base value, overridden per timeframe

    log(f"Entering main loop at {time.strftime('%H:%M:%S', time.localtime(time.time()))}")
    if 'peak_timestamp' not in state:
        state['peak_timestamp'] = time.time()
    if 'trade_cooldown_until' not in state:
        state['trade_cooldown_until'] = 0
    if 'rsi_price_history_eagle' not in state:
        state['rsi_price_history_eagle'] = []
    if 'rsi_price_history_medium' not in state:
        state['rsi_price_history_medium'] = []
    if 'rsi_price_history_long' not in state:
        state['rsi_price_history_long'] = []
    if 'position' not in state:
        state['position'] = 0
    if 'last_buy_price' not in state:
        state['last_buy_price'] = 0
    if 'trailing_stop_price' not in state:
        state['trailing_stop_price'] = 0
    save_state()

    # Fetch initial prices from CryptoCompare for extended period (1000 minutes)
    log(f"Fetching initial prices from CryptoCompare at {time.strftime('%H:%M:%S', time.localtime(time.time()))}")
    API_KEY = os.getenv("CMC_KEY")
    if not API_KEY:
        log("ERROR: CMC_KEY (used as CryptoCompare API key) is not set")
        raise RuntimeError("CryptoCompare API key not found")
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {"fsym": "SOL", "tsym": "USD", "limit": 1000, "api_key": API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("Response") != "Success":
            log(f"CryptoCompare returned error: {data.get('Message')}")
            raise RuntimeError("CryptoCompare response not successful")
        candles = data.get("Data", {}).get("Data", [])
        if len(candles) < 50:
            log(f"Insufficient data from CryptoCompare: {len(candles)}/50")
            raise RuntimeError("Not enough historical data")
        initial_prices = [c["close"] for c in candles if c.get("close") is not None and c["close"] > 0][-1000:]  # Ensure positive prices
        if len(initial_prices) < 50 or len(set(initial_prices)) < 2:
            log(f"Insufficient valid or varied closes: {len(initial_prices)} prices, unique values: {len(set(initial_prices))}")
            raise RuntimeError("Invalid or insufficient price variation")
        state['price_history'] = initial_prices
        state['last_price'] = initial_prices[-1]
        log(f"Initial prices fetched: {len(state['price_history'])} prices, last price ${state['last_price']:.2f}")

        # Initialize timeframe-specific price histories with full initial data
        state['rsi_price_history_eagle'] = initial_prices[-50:]  # Use last 50 prices for eagle
        state['rsi_price_history_medium'] = [initial_prices[i] for i in range(0, len(initial_prices), 10)][-50:]  # ~100 prices, last 50
        state['rsi_price_history_long'] = [initial_prices[i] for i in range(0, len(initial_prices), 15)][-50:]  # ~67 prices, last 50
        log(f"Initialized rsi_price_history_eagle with {len(state['rsi_price_history_eagle'])} prices")
        log(f"Initialized rsi_price_history_medium with {len(state['rsi_price_history_medium'])} prices")
        log(f"Initialized rsi_price_history_long with {len(state['rsi_price_history_long'])} prices")

        # Precompute warmup indicators for eagle (single pass with valid data)
        if len(state['rsi_price_history_eagle']) >= 14:
            rsi_values = []
            for i in range(len(state['rsi_price_history_eagle']) - 14 + 1):
                slice_prices = state['rsi_price_history_eagle'][i:i+14]
                rsi = get_current_rsi(slice_prices, period=14)
                if rsi != 50.0:  # Only append if there's variation
                    rsi_values.append(rsi)
            if rsi_values:
                state['eagle_avg_rsi'] = np.mean(rsi_values[-20:]) if len(rsi_values) >= 20 else np.mean(rsi_values)
                log(f"Precomputed Eagle avg_rsi: {state['eagle_avg_rsi']:.2f} from {len(rsi_values)} valid RSI values")
            else:
                state['eagle_avg_rsi'] = 50.0
                log("No valid RSI values precomputed, setting Eagle avg_rsi to 50.0")

        with open('price_history.json', 'w') as f:
            json.dump({'prices': state['price_history'], 'timestamp': time.time()}, f)
        save_state()
    except Exception as e:
        log(f"Failed to fetch initial prices from CryptoCompare: {e}, using fallback with variation")
        # Fallback with simulated variation
        initial_prices = [150.0 + i * 0.1 for i in range(1000)]  # Simulated prices with slight variation
        state['price_history'] = initial_prices
        state['last_price'] = initial_prices[-1]
        state['rsi_price_history_eagle'] = initial_prices[-50:]
        state['rsi_price_history_medium'] = [initial_prices[i] for i in range(0, len(initial_prices), 10)][-50:]
        state['rsi_price_history_long'] = [initial_prices[i] for i in range(0, len(initial_prices), 15)][-50:]
        log(f"Fallback initial price set to ${state['last_price']:.2f} with variation")
        with open('price_history.json', 'w') as f:
            json.dump({'prices': state['price_history'], 'timestamp': time.time()}, f)
        save_state()

    current_time = time.time()
    last_save_time = os.path.getmtime('state.json') if os.path.exists('state.json') else 0
    if last_save_time > 0 and current_time - last_save_time > 48 * 3600:
        log(f"State file older than 48 hours, resetting")
    price = state['last_price']
    if price:
        portfolio_value = get_portfolio_value(price)
        if state.get('peak_portfolio', 0) == 0 or abs(state['peak_portfolio'] - portfolio_value) / portfolio_value > 0.5:
            state['peak_portfolio'] = portfolio_value
            state['peak_timestamp'] = current_time
        if state.get('pause_until', 0) > current_time and (current_time - state['peak_timestamp'] > 6 * 3600 or last_save_time == 0):
            state['pause_until'] = 0
        save_state()

    if current_time - state['peak_timestamp'] > 604800:
        log(f"Peak portfolio over 7 days old, resetting to ${portfolio_value:.2f}")
        if price:
            portfolio_value = get_portfolio_value(price)
            state['peak_portfolio'] = portfolio_value
            state['peak_timestamp'] = current_time
            save_state()

    last_stats_time = time.time()
    last_indicator_time = {'eagle': 0, 'medium': 0, 'long': 0}
    cached_indicators = {
        'eagle': {'rsi': None, 'macd_line': None, 'signal_line': None, 'vwap': None, 'upper_bb': None, 'lower_bb': None, 'atr': None, 'momentum': None, 'avg_rsi': state.get('eagle_avg_rsi', 50.0), 'avg_atr': 2.5},
        'medium': {'rsi': None, 'macd_line': None, 'signal_line': None, 'vwap': None, 'upper_bb': None, 'lower_bb': None, 'atr': None, 'momentum': None, 'avg_atr': 2.5},
        'long': {'rsi': None, 'macd_line': None, 'signal_line': None, 'vwap': None, 'upper_bb': None, 'lower_bb': None, 'atr': None, 'momentum': None, 'avg_atr': 2.5}
    }
    last_sol_balance = state.get('last_sol_balance', 0.0)
    last_usdc_balance = state.get('last_usdc_balance', 0.0)
    peak_market_value = state.get('peak_market_value', 0.0)

    def get_updated_portfolio(price, max_retries=5, wait_time=30):
        for attempt in range(max_retries):
            try:
                sol_balance = get_sol_balance()
                usdc_balance = get_usdc_balance()
                if sol_balance is not None and usdc_balance is not None:
                    portfolio = usdc_balance + (sol_balance * price)
                    expected_portfolio = last_usdc_balance + (last_sol_balance * price)
                    if abs(portfolio - expected_portfolio) > 0.1 and attempt < max_retries - 1:
                        log(f"Portfolio mismatch, retrying (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    log(f"Portfolio: SOL={sol_balance:.4f}, USDC={usdc_balance:.4f}, Value=${portfolio:.2f}")
                    return portfolio, sol_balance, usdc_balance
            except Exception as e:
                log(f"Error fetching portfolio (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    log(f"Portfolio fetch failed after {max_retries} attempts, using last known value")
                    return last_usdc_balance + (last_sol_balance * price) if last_sol_balance is not None and last_usdc_balance is not None else (0, 0, 0)
                time.sleep(wait_time)
        log(f"Failed to fetch portfolio after {max_retries} attempts")
        return None, None, None

    def fetch_backup_price():
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            log(f"Backup price fetch failed: Status {response.status_code}")
            return None
        except Exception as e:
            log(f"Backup price fetch error: {e}")
            return None

    while True:
        try:
            loop_start = time.time()
            current_time = time.time()
            log(f"Loop iteration started at {time.strftime('%H:%M:%S', time.localtime(current_time))}")

            if current_time < state['pause_until']:
                remaining_pause = state['pause_until'] - current_time
                if state['pause_until'] - state['peak_timestamp'] > 48 * 3600:
                    log(f"Pause exceeded 48 hours, resuming")
                    state['pause_until'] = 0
                    if price:
                        portfolio_value = get_portfolio_value(price)
                        state['peak_portfolio'] = portfolio_value
                        state['peak_timestamp'] = current_time
                    save_state()
                else:
                    log(f"Paused until {time.strftime('%H:%M:%S', time.localtime(state['pause_until']))}, remaining {remaining_pause:.1f}s")
                    time.sleep(TRADE_INTERVAL)
                    continue
            else:
                if state['pause_until'] != 0:
                    log(f"Resuming trading after pause")
                    state['pause_until'] = 0
                    if price:
                        portfolio_value = get_portfolio_value(price)
                        state['peak_portfolio'] = portfolio_value
                        state['peak_timestamp'] = current_time
                    save_state()

            price = fetch_current_price()
            if price is None:
                backup_price = fetch_backup_price()
                if backup_price:
                    price = backup_price
                    log(f"Using backup price: ${price:.2f}")
                elif state['price_history']:
                    price = state['price_history'][-1]
                    log(f"Using last cached price: ${price:.2f}")
                else:
                    log(f"No price data available, skipping iteration")
                    time.sleep(TRADE_INTERVAL)
                    continue
            state['last_price'] = price
            state['price_history'].append(price)
            if len(state['price_history']) > 1000:
                state['price_history'].pop(0)

            # Trim to 50 prices after first cycle
            if current_time - last_indicator_time['eagle'] >= 30 and len(state['rsi_price_history_eagle']) > 50:
                state['rsi_price_history_eagle'] = state['rsi_price_history_eagle'][-50:]
            if current_time - last_indicator_time['medium'] >= 600 and len(state['rsi_price_history_medium']) > 50:
                state['rsi_price_history_medium'] = state['rsi_price_history_medium'][-50:]
            if current_time - last_indicator_time['long'] >= 900 and len(state['rsi_price_history_long']) > 50:
                state['rsi_price_history_long'] = state['rsi_price_history_long'][-50:]

            # Update timeframe-specific price histories
            if current_time - last_indicator_time['eagle'] >= 30:
                state['rsi_price_history_eagle'].append(price)
            if current_time - last_indicator_time['medium'] >= 600:
                state['rsi_price_history_medium'].append(price)
            if current_time - last_indicator_time['long'] >= 900:
                state['rsi_price_history_long'].append(price)

            state['last_fetch_time'] = current_time
            save_state()

            try:
                with open('price_history.json', 'w') as f:
                    json.dump({'prices': state['price_history'], 'timestamp': time.time()}, f)
            except Exception as e:
                log(f"Failed to save price history: {e}")

            # Calculate indicators only for current window
            for timeframe, period in [('eagle', 30), ('medium', 600), ('long', 900)]:
                if current_time - last_indicator_time[timeframe] >= period or all(cached_indicators[timeframe][k] is None for k in cached_indicators[timeframe]):
                    prices = state[f'rsi_price_history_{timeframe}'][-14:]  # Always use last 14 prices
                    if len(prices) < 14 or len(set(prices)) < 2:
                        cached_indicators[timeframe]['rsi'] = 50.0
                        log(f"{timeframe.capitalize()} RSI set to 50.0 due to insufficient data or variation")
                    else:
                        cached_indicators[timeframe]['rsi'] = get_current_rsi(prices, period=14)
                    if timeframe == 'eagle' and len(state['rsi_price_history_eagle']) >= 34:
                        valid_rsi_values = [get_current_rsi(state['rsi_price_history_eagle'][i:i+14], period=14) 
                                          for i in range(max(0, len(state['rsi_price_history_eagle']) - 34), len(state['rsi_price_history_eagle']) - 14 + 1) 
                                          if len(set(state['rsi_price_history_eagle'][i:i+14])) >= 2]
                        cached_indicators[timeframe]['avg_rsi'] = np.mean(valid_rsi_values[-20:]) if valid_rsi_values and len(valid_rsi_values) >= 20 else (np.mean(valid_rsi_values) if valid_rsi_values else state.get('eagle_avg_rsi', 50.0))
                        log(f"{timeframe.capitalize()} avg_rsi: {cached_indicators[timeframe]['avg_rsi']:.2f} with {len(valid_rsi_values)} values")
                    else:
                        cached_indicators[timeframe]['avg_rsi'] = state.get('eagle_avg_rsi', 50.0) if timeframe == 'eagle' else 50.0
                    if timeframe == 'eagle':
                        macd_result = calculate_macd(state[f'rsi_price_history_{timeframe}'], fast=6, slow=12, signal=3) if len(state[f'rsi_price_history_{timeframe}']) >= 12 else (None, None)
                    else:
                        macd_result = calculate_macd(state[f'rsi_price_history_{timeframe}']) if len(state[f'rsi_price_history_{timeframe}']) >= 34 else (None, None)
                    cached_indicators[timeframe]['macd_line'], cached_indicators[timeframe]['signal_line'] = macd_result
                    cached_indicators[timeframe]['vwap'] = calculate_vwap(state[f'rsi_price_history_{timeframe}']) if len(state[f'rsi_price_history_{timeframe}']) >= 20 else None
                    bb_result = calculate_bollinger_bands(state[f'rsi_price_history_{timeframe}']) if len(state[f'rsi_price_history_{timeframe}']) >= 20 else (None, None)
                    cached_indicators[timeframe]['upper_bb'], cached_indicators[timeframe]['lower_bb'] = bb_result
                    cached_indicators[timeframe]['atr'] = calculate_atr(state[f'rsi_price_history_{timeframe}']) if len(state[f'rsi_price_history_{timeframe}']) >= 20 else None
                    cached_indicators[timeframe]['momentum'] = calculate_momentum(state[f'rsi_price_history_{timeframe}']) if len(state[f'rsi_price_history_{timeframe}']) >= 2 else 0.0
                    if cached_indicators[timeframe]['atr'] is not None:
                        state[f'atr_history_{timeframe}'] = state.get(f'atr_history_{timeframe}', []) + [cached_indicators[timeframe]['atr']]
                        if len(state[f'atr_history_{timeframe}']) > 50:
                            state[f'atr_history_{timeframe}'].pop(0)
                        cached_indicators[timeframe]['avg_atr'] = np.mean(state[f'atr_history_{timeframe}']) if state[f'atr_history_{timeframe}'] else 2.5
                    else:
                        cached_indicators[timeframe]['avg_atr'] = 2.5
                    last_indicator_time[timeframe] = current_time
                    rsi_str = f"{cached_indicators[timeframe]['rsi']:.2f}" if cached_indicators[timeframe]['rsi'] is not None else "N/A"
                    macd_line_str = f"{cached_indicators[timeframe]['macd_line']:.2f}" if cached_indicators[timeframe]['macd_line'] is not None else "N/A"
                    signal_line_str = f"{cached_indicators[timeframe]['signal_line']:.2f}" if cached_indicators[timeframe]['signal_line'] is not None else "N/A"
                    vwap_str = f"{cached_indicators[timeframe]['vwap']:.2f}" if cached_indicators[timeframe]['vwap'] is not None else "N/A"
                    upper_bb_str = f"{cached_indicators[timeframe]['upper_bb']:.2f}" if cached_indicators[timeframe]['upper_bb'] is not None else "N/A"
                    lower_bb_str = f"{cached_indicators[timeframe]['lower_bb']:.2f}" if cached_indicators[timeframe]['lower_bb'] is not None else "N/A"
                    atr_str = f"{cached_indicators[timeframe]['atr']:.2f}" if cached_indicators[timeframe]['atr'] is not None else "N/A"
                    momentum_str = f"{cached_indicators[timeframe]['momentum']:.2f}" if cached_indicators[timeframe]['momentum'] is not None else "N/A"
                    log(f"{timeframe.capitalize()} Indicators ({period//60 if period >= 60 else period}s) - RSI: {rsi_str}, avg_rsi: {cached_indicators[timeframe]['avg_rsi']:.2f}, MACD: {macd_line_str}/{signal_line_str}, VWAP: {vwap_str}, BB: {upper_bb_str}/{lower_bb_str}, ATR: {atr_str}, Momentum: {momentum_str}")

            if any(ind is None for timeframe in cached_indicators for ind in ['rsi', 'vwap', 'upper_bb', 'lower_bb', 'atr', 'momentum'] if timeframe in cached_indicators):
                log("Missing critical indicators, skipping")
                time.sleep(TRADE_INTERVAL)
                continue

            portfolio_value, sol_balance, usdc_balance = get_updated_portfolio(price, wait_time=30)
            if portfolio_value is None:
                log("Portfolio fetch failed, skipping")
                time.sleep(TRADE_INTERVAL)
                continue
            if sol_balance is not None and usdc_balance is not None:
                if abs(sol_balance - last_sol_balance) > 0.0001 or abs(usdc_balance - last_usdc_balance) > 0.01:
                    log(f"Balance updated: SOL {last_sol_balance:.4f} -> {sol_balance:.4f}, USDC {last_usdc_balance:.4f} -> {usdc_balance:.4f}")
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
                log(f"Market metrics - Current Value: ${current_market_value:.2f}, Peak Value: ${peak_market_value:.2f}, Drawdown: {market_drawdown:.2f}%")

            if portfolio_value > 0:
                if state.get('peak_portfolio', 0) == 0 or abs(state.get('peak_portfolio', 0) - portfolio_value) / portfolio_value > 0.5:
                    state['peak_portfolio'] = portfolio_value
                    state['peak_timestamp'] = current_time
                    state['peak_market_value'] = portfolio_value
                    log(f"Peak portfolio reset to: ${portfolio_value:.2f}")
                elif portfolio_value > state['peak_portfolio']:
                    state['peak_portfolio'] = portfolio_value
                    state['peak_timestamp'] = current_time
                    state['peak_market_value'] = portfolio_value
                    log(f"Peak portfolio updated to: ${portfolio_value:.2f}")
            save_state()

            log(f"Portfolio Value: ${portfolio_value:.2f}, Market Drawdown: {market_drawdown:.2f}%")
            if market_drawdown > MAX_DRAWDOWN:
                log(f"Drawdown {market_drawdown:.2f}% exceeds {MAX_DRAWDOWN}%, pausing for 1 hour")
                state['pause_until'] = current_time + 3600
                save_state()
                continue

            adjust_triggers(cached_indicators['eagle']['atr'], cached_indicators['eagle']['avg_atr'], cached_indicators['eagle']['rsi'])
            total_sol_balance = get_sol_balance()
            total_usdc_balance = get_usdc_balance()
            log(f"Current balances - SOL: {total_sol_balance:.4f}, USDC: {total_usdc_balance:.4f}")

            # Buy Logic
            if current_time >= state['trade_cooldown_until'] and total_usdc_balance > MIN_TRADE_USD:
                for timeframe, buy_factor, profit_target in [('eagle', 0.3, 3), ('medium', 0.4, 10), ('long', 0.5, 25)]:
                    ind = cached_indicators[timeframe]
                    fee = get_fee_estimate()
                    bid_ask_spread = abs(fetch_current_price() - price) / price if price else 0.01
                    if timeframe == 'eagle':
                        rsi_condition = ind['rsi'] < (ind['avg_rsi'] - 5) if ind['avg_rsi'] is not None else False
                    else:
                        rsi_condition = ind['rsi'] < 35
                    if rsi_condition and check_buy_signal(price, ind['rsi'], ind['macd_line'], ind['signal_line'], ind['vwap'], ind['lower_bb'], ind['momentum'], ind['atr'], ind['avg_atr'], timeframe):
                        target_usdc = portfolio_value * 0.1
                        position_size = min(target_usdc / price, total_usdc_balance / (price * (1 + SLIPPAGE + get_fee_estimate() / price)))  # Adjusted for flat fee
                        if position_size < 0.001 and total_usdc_balance > MIN_TRADE_USD:
                            position_size = total_usdc_balance / (price * (1 + SLIPPAGE + get_fee_estimate() / price))
                        if position_size > 0.001 and position_size * price * (1 + SLIPPAGE + get_fee_estimate() / price) <= total_usdc_balance and bid_ask_spread < 0.005:
                            execute_buy(position_size)
                            time.sleep(10)
                            state['position'] += position_size
                            state.setdefault('buy_orders', []).append({'amount': position_size, 'buy_price': price, 'timeframe': timeframe})
                            state['trade_cooldown_until'] = current_time + 2
                            state['trailing_stop_price'] = price * (1 - 0.03)
                            state['highest_price'] = price
                            log(f"{timeframe.capitalize()} Buy: {position_size:.4f} SOL, Position: {state['position']:.4f} SOL, RSI: {ind['rsi']:.2f}")
                            save_state()

            # Sell Logic
            if total_sol_balance > MIN_SOL_THRESHOLD and state['position'] > 0:
                fee = get_fee_estimate()
                if price > state['highest_price']:
                    state['highest_price'] = price
                    state['trailing_stop_price'] = max(state['trailing_stop_price'], price * (1 - 0.01))
                for timeframe, sell_factor, profit_target in [('eagle', 0.2, 1), ('medium', 0.3, 5), ('long', 0.5, 20)]:
                    ind = cached_indicators[timeframe]
                    sell_condition = False
                    sell_amount = 0
                    buy_to_sell = None
                    for buy_order in state.get('buy_orders', []):
                        buy_profit_target = {'eagle': 1, 'medium': 5, 'long': 20}[buy_order['timeframe']]
                        net_profit_percent = ((price - buy_order['buy_price'] - fee * 2) / buy_order['buy_price'] * 100) if buy_order['buy_price'] else 0
                        if net_profit_percent >= buy_profit_target or (net_profit_percent > 0 and buy_order['timeframe'] in ['eagle', 'medium']):
                            sell_condition = True
                            sell_amount = buy_order['amount']
                            buy_to_sell = buy_order
                            break
                    if not sell_condition and price <= state['trailing_stop_price'] and state['buy_orders']:
                        sell_condition = True
                        sell_amount = min(state['buy_orders'][0]['amount'], state['position'])
                        buy_to_sell = state['buy_orders'][0]
                    if sell_condition:
                        sell_amount = min(sell_amount, state['position'])
                        if sell_amount > 0.001:
                            execute_sell(sell_amount, price)
                            time.sleep(10)
                            profit = (price - buy_to_sell['buy_price']) * sell_amount - (get_fee_estimate() * 2)  # Flat fee for buy + sell
                            state['position'] -= sell_amount
                            state['total_profit'] = state.get('total_profit', 0) + profit
                            state['buy_orders'].remove(buy_to_sell)
                            state['trade_cooldown_until'] = current_time + 2
                            log(f"{timeframe.capitalize()} Sell: {sell_amount:.4f} SOL @ ${price:.2f}, Profit: ${profit:.2f}, Buy Price: ${buy_to_sell['buy_price']:.2f}, Timeframe: {buy_to_sell['timeframe']}")
                            save_state()
                            if state['position'] <= 0.001:
                                state['position'] = 0
                                state['highest_price'] = 0
                                state['trailing_stop_price'] = 0
                                state['buy_orders'] = []

            if current_time - last_stats_time >= 3600:
                log_performance(portfolio_value)
                last_stats_time = current_time

            elapsed = time.time() - loop_start
            sleep_time = max(0, TRADE_INTERVAL - elapsed)
            log(f"Sleeping for {sleep_time:.1f}s")
            time.sleep(sleep_time)
        except Exception as e:
            log(f"Error in main loop, continuing: {str(e)}")
            time.sleep(TRADE_INTERVAL)
            continue

# Ensure get_current_rsi supports custom period (already updated)
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
        load_state()
        with open('stats.csv', 'w') as f:
            f.write("timestamp,portfolio_value,total_trades,wins,losses,total_profit,win_rate,profit_factor,drawdown\n")
        while True:
            try:
                main()
            except Exception as e:
                log(f"Main loop crashed, restarting: {e}")
                time.sleep(10)
                continue
    except KeyboardInterrupt:
        log("Bot stopped by user")
        portfolio_value = get_portfolio_value(state['price_history'][-1] if state['price_history'] else 0)
        log_performance(portfolio_value)
        save_state()
        log(f"Final Stats: Total Return: {state['total_profit'] / portfolio_value * 100:.2f}%, "
            f"Win Rate: {state['wins'] / state['total_trades'] * 100 if state['total_trades'] > 0 else 0:.2f}%, "
            f"Adverse Months: Not Calculated")
    except Exception as e:
        log(f"Bot crashed: {e}")
        raise
