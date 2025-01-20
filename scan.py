import streamlit as st
import aiohttp
import numpy as np
import asyncio
import pandas as pd
import threading
import json
import os
import requests
import time
from datetime import datetime, timedelta

# Default values will be overridden by UI inputs
STATE_FILE = "scanner_state.json"
RESULTS_FILE = "results.csv"
LOG_FILE = "scanner_logs.json"

# ======================
# HTTP Client Utilities
# ======================
async def fetch_with_retry(session, url, headers, params, max_retries):
    for attempt in range(max_retries):
        try:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise e
    return None

# ======================
# Data Fetching Functions
# ======================
def fetch_top_tokens(api_key, min_liquidity):
    """Fetch tokens with highest 24h volume using BirdEye's tokenlist endpoint"""
    url = "https://public-api.birdeye.so/defi/tokenlist"
    params = {"sort_by": "v24hUSD", "sort_type": "desc", "offset": 0, "min_liquidity": min_liquidity}
    headers = {"accept": "application/json", "x-chain": "solana", "X-API-KEY": api_key}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('data', {}).get('tokens', [])
    except Exception as e:
        return []

async def get_dex_data(session, token_address, api_key, max_retries):
    """Get current price data from BirdEye's API"""
    url = "https://public-api.birdeye.so/defi/price"
    params = {'address': token_address}
    headers = {"accept": "application/json", "X-API-KEY": api_key, "x-chain": "solana"}
    
    try:
        data = await fetch_with_retry(session, url, headers, params, max_retries)
        return data['data'] if data.get('success') else None
    except Exception as e:
        return None

async def get_historical_data(session, token_address, hours, interval, api_key, max_retries):
    """Get historical price data"""
    url = "https://public-api.birdeye.so/defi/history_price"
    now = datetime.now()
    params = {
        'address': token_address,
        'address_type': 'token',
        'type': interval,
        'time_from': int((now - timedelta(hours=hours)).timestamp()),
        'time_to': int(now.timestamp())
    }
    headers = {"accept": "application/json", "X-API-KEY": api_key, "x-chain": "solana"}
    
    try:
        data = await fetch_with_retry(session, url, headers, params, max_retries)
        return data.get('data', {}).get('items', []) if data.get('success') else None
    except Exception as e:
        return None

# ======================
# Analysis Functions
# ======================
def calculate_metrics(historical_data):
    """Calculate volatility and momentum from historical prices"""
    if not historical_data or len(historical_data) < 4:
        return None
    
    prices = [x['value'] for x in historical_data]
    changes = np.diff(prices) / prices[:-1] * 100
    
    return {
        'volatility': np.std(changes),
        '3period_change': sum(changes[-3:]),
        '6h_ma': np.mean(prices[-3:])
    }

async def analyze_token(session, token, scanner):
    """Full analysis for a single token"""
    if token['symbol'] in scanner.parameters['stablecoins']:
        scanner.add_log(f"Skipping stablecoin {token['symbol']}")
        return None
    
    dex_data = await get_dex_data(session, token['address'], 
                                scanner.parameters['api_key'],
                                scanner.parameters['max_retries'])
    if dex_data is None:
        scanner.add_log(f"Failed to fetch DEX data for {token['symbol']}")
        return None
    
    await asyncio.sleep(scanner.parameters['delay_between_tokens'])
    historical_data = await get_historical_data(session, token['address'],
                                              scanner.parameters['historical_hours'],
                                              scanner.parameters['historical_interval'],
                                              scanner.parameters['api_key'],
                                              scanner.parameters['max_retries'])
    if not historical_data:
        scanner.add_log(f"Failed to fetch historical data for {token['symbol']}")
        return None
    
    metrics = calculate_metrics(historical_data)
    if not metrics:
        scanner.add_log(f"Insufficient data for metrics calculation: {token['symbol']}")
        return None
    
    meets_criteria = all([
        token['liquidity'] >= scanner.parameters['min_liquidity'],
        token['v24hUSD'] >= scanner.parameters['min_volume'],
        dex_data['priceChange24h'] >= scanner.parameters['min_price_change_24h'],
        dex_data['value'] > scanner.parameters['min_price'],
        metrics['volatility'] < scanner.parameters['max_volatility'],
        metrics['volatility'] > scanner.parameters['min_volatility'],
        metrics['3period_change'] > scanner.parameters['min_3period_change'],
    ])
    
    if not meets_criteria:
        scanner.add_log(f"Token {token['symbol']} does not meet criteria")
        return None
    
    scanner.add_log(f"New opportunity detected: {token['symbol']}")
    return {
        'symbol': token['symbol'],
        'address': token['address'],
        'price': dex_data['value'],
        '24h_chg': f"{dex_data['priceChange24h']:.1f}%",
        'volatility': f"{metrics['volatility']:.1f}%",
        '6h_momentum': f"{metrics['3period_change']:.1f}%",
        'volume': f"${token['v24hUSD']/1000:.1f}K",
        'liquidity': f"${token['liquidity']/1000:.1f}K",
        'market_cap': f"${token.get('mc',0)/1e6:.2f}M",
        'timestamp': datetime.now().isoformat()
    }

# ======================
# State Management
# ======================
def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"running": False, "last_run": None, "processed": 0}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def load_results():
    try:
        updated = pd.read_csv(RESULTS_FILE)
        updated = updated.sort_values(by='timestamp').drop_duplicates(subset='address', keep='last')
        return updated
    except FileNotFoundError:
        return pd.DataFrame()

def save_results(df):
    df.to_csv(RESULTS_FILE, index=False)

def load_processed_addresses():
    try:
        df = pd.read_csv(RESULTS_FILE)
        return set(df['address'])
    except FileNotFoundError:
        return set()

# ======================
# Scanner Service
# ======================
class ScannerService:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.logs = []
        self.log_lock = threading.Lock()
        self._load_logs()
        self.parameters = {
            'api_key': "c9053344ac544522896e3e0814047d48",
            'min_liquidity_top': 1000,
            'min_liquidity': 5000,
            'min_volume': 25000,
            'min_price_change_24h': 3.0,
            'min_price': 0.00001,
            'max_volatility': 30.0,
            'min_volatility': 5.0,
            'min_3period_change': 1.5,
            'historical_hours': 24,
            'historical_interval': '2H',
            'stablecoins': ['USDC', 'USDT', 'USDH'],
            'max_retries': 10,
            'delay_between_tokens': 3,
            'delay_between_cycles':3 
        }

    def _load_logs(self):
        try:
            with open(LOG_FILE, "r") as f:
                self.logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.logs = []

    def _save_logs(self):
        with open(LOG_FILE, "w") as f:
            json.dump(self.logs[-100:], f)

    def add_log(self, message):
        with self.log_lock:
            timestamp = datetime.now().isoformat()
            self.logs.append({"timestamp": timestamp, "message": message})
            self._save_logs()

    def get_logs(self):
        with self.log_lock:
            return self.logs.copy()

    def start(self):
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()
                save_state({"running": True, "last_run": str(datetime.now()), "processed": 0})
                self.add_log("Service started")

    def stop(self):
        with self.lock:
            if self.running:
                self.running = False
                save_state({"running": False, "last_run": load_state().get("last_run"), "processed": 0})
                self.add_log("Service stopped")

    def run(self):
        async def async_main():
            processed_addresses = load_processed_addresses()
            async with aiohttp.ClientSession() as session:
                while self.running:
                    try:
                        tokens = fetch_top_tokens(self.parameters['api_key'],
                                                self.parameters['min_liquidity_top'])
                        if not tokens:
                            self.add_log("Warning: No tokens fetched")
                            await asyncio.sleep(1)
                            continue
                        
                        new_tokens = [t for t in tokens if t['address'] not in processed_addresses]
                        if not new_tokens:
                            self.add_log("No new tokens to process")
                            await asyncio.sleep(1)
                            continue
                        
                        self.add_log(f"Processing {len(new_tokens)} new tokens")
                        opportunities = []
                        for token in new_tokens:
                            result = await analyze_token(session, token, self)
                            if result:
                                to_del = None
                                for o in opportunities:
                                    if result['address'] == o['address']:
                                        to_del = o
                                if to_del is not None:
                                    opportunities.pop(to_del)
                                
                                opportunities.append(result)
                                df = pd.DataFrame(opportunities)
                                existing = load_results()
                                updated = pd.concat([existing, df], ignore_index=True)

                                save_results(updated)
                                processed_addresses.update(df['address'].tolist())

                            await asyncio.sleep(self.parameters['delay_between_tokens'])

                        await asyncio.sleep(self.parameters['delay_between_cycles'])
                    except Exception as e:
                        self.add_log(f"Critical error: {str(e)}")
                        await asyncio.sleep(1)

        asyncio.run(async_main())

# ======================
# Streamlit UI
# ======================
if 'scanner' not in st.session_state:
    st.session_state.scanner = ScannerService()

st.title("🔍 Solana Token Scanner (24/7 Service)")

# Sidebar controls
with st.sidebar:
    st.header("Configuration Panel")
    
    # API Settings
    st.subheader("API Settings")
    api_key = st.text_input("BirdEye API Key", 
                          value=st.session_state.scanner.parameters['api_key'],
                          type="password")
    
    # Top Tokens Filter
    st.subheader("Top Tokens Filter")
    min_liquidity_top = st.number_input("Minimum Liquidity ($)", 
                                      value=st.session_state.scanner.parameters['min_liquidity_top'],
                                      min_value=0)
    
    # Analysis Criteria
    st.subheader("Analysis Criteria")
    min_liquidity = st.number_input("Minimum Liquidity ($)", 
                                  value=st.session_state.scanner.parameters['min_liquidity'],
                                  min_value=0)
    min_volume = st.number_input("Minimum 24h Volume ($)", 
                               value=st.session_state.scanner.parameters['min_volume'],
                               min_value=0)
    min_price_change_24h = st.number_input("Minimum 24h Price Change (%)", 
                                         value=st.session_state.scanner.parameters['min_price_change_24h'],
                                         min_value=0.0,
                                         format="%.1f")
    min_price = st.number_input("Minimum Price ($)", 
                              value=st.session_state.scanner.parameters['min_price'],
                              format="%.5f")
    min_volatility = st.number_input("Minimum Volatility (%)", 
                                   value=st.session_state.scanner.parameters['min_volatility'],
                                   min_value=0.0,
                                   format="%.1f")
    max_volatility = st.number_input("Maximum Volatility (%)", 
                                   value=st.session_state.scanner.parameters['max_volatility'],
                                   min_value=0.0,
                                   format="%.1f")
    min_3period_change = st.number_input("Minimum 3-Period Momentum (%)", 
                                       value=st.session_state.scanner.parameters['min_3period_change'],
                                       min_value=0.0,
                                       format="%.1f")
    
    # Historical Data
    st.subheader("Historical Data")
    historical_hours = st.slider("Analysis Window (hours)", 
                               min_value=1, max_value=48, 
                               value=st.session_state.scanner.parameters['historical_hours'])
    historical_interval = st.selectbox("Candle Interval", 
                                     options=['1H', '2H', '4H', '6H', '8H', '12H', '1D'],
                                     index=1)
    
    # Advanced Settings
    st.subheader("Advanced Settings")
    max_retries = st.number_input("API Retry Attempts", 
                                min_value=1, max_value=10, 
                                value=st.session_state.scanner.parameters['max_retries'])
    delay_between_tokens = st.number_input("Delay Between Tokens (sec)", 
                                         min_value=0, max_value=60, 
                                         value=st.session_state.scanner.parameters['delay_between_tokens'])
    delay_between_cycles = st.number_input("Delay Between Cycles (sec)", 
                                         min_value=0, max_value=3600, 
                                         value=st.session_state.scanner.parameters['delay_between_cycles'])
    stablecoins = st.multiselect("Exclude Stablecoins", 
                               options=['USDC', 'USDT', 'USDH', 'BUSD', 'DAI', 'PAI'],
                               default=st.session_state.scanner.parameters['stablecoins'])

# Control buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start Service"):
        st.session_state.scanner.parameters.update({
            'api_key': api_key,
            'min_liquidity_top': min_liquidity_top,
            'min_liquidity': min_liquidity,
            'min_volume': min_volume,
            'min_price_change_24h': min_price_change_24h,
            'min_price': min_price,
            'max_volatility': max_volatility,
            'min_volatility': min_volatility,
            'min_3period_change': min_3period_change,
            'historical_hours': historical_hours,
            'historical_interval': historical_interval,
            'max_retries': max_retries,
            'delay_between_tokens': delay_between_tokens,
            'delay_between_cycles': delay_between_cycles,
            'stablecoins': stablecoins
        })
        st.session_state.scanner.start()
with col2:
    if st.button("⏹️ Stop Service"):
        st.session_state.scanner.stop()
with col3:
    show_logs = st.button("📜 View Logs")

# Status display
current_state = load_state()
status_info = []
if current_state.get("running", False):
    status_info.append("**Status:** 🟢 Running")
else:
    status_info.append("**Status:** 🔴 Stopped")

if current_state.get("last_run"):
    status_info.append(f"**Last scan:** {current_state['last_run']}")
if current_state.get("processed"):
    status_info.append(f"**Processed:** {current_state['processed']} tokens")

st.markdown("\n\n".join(status_info))

# Results display
try:
    df = load_results()
    if not df.empty:
        st.dataframe(
            df.sort_values('timestamp', ascending=False).head(20),
            height=500,
            use_container_width=True
        )
    else:
        st.info("No results found yet")
except Exception as e:
    st.error(f"Error loading results: {str(e)}")

# Logs display
if show_logs:
    st.subheader("System Logs")
    logs = st.session_state.scanner.get_logs()
    if logs:
        for log in reversed(logs[-20:]):
            st.code(f"{log['timestamp']} - {log['message']}")
    else:
        st.info("No logs available yet")

# Background service management
if current_state.get("running") and not st.session_state.scanner.running:
    st.session_state.scanner.start()

if __name__ == "__main__":
    while True:
        try:
            state = load_state()
            if state.get("running"):
                if not st.session_state.scanner.running:
                    st.session_state.scanner.start()
        except KeyboardInterrupt:
            break