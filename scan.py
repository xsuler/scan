import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
from solders.pubkey import Pubkey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import queue

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
UI_REFRESH_INTERVAL = 0.1
ROW_HEIGHT = 35
HEADER_HEIGHT = 50

class TokenAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def get_all_tokens(self, strict_checks=True):
        try:
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
            response.raise_for_status()
            tokens = response.json()
            return [t for t in tokens if self.validate_token(t, strict_checks)]
        except Exception as e:
            st.error(f"Token fetch error: {str(e)}")
            return []

    def validate_token(self, token, strict_checks):
        try:
            if not Pubkey.from_string(token.get('address', '')):
                return False

            required_fields = ['symbol', 'name', 'decimals', 'logoURI']
            for field in required_fields:
                if not token.get(field):
                    return False

            if strict_checks:
                if "community" in token.get('tags', []) or token.get('chainId') != 101:
                    return False
            return True
        except:
            return False

    def safe_float(self, value, default=0.0):
        try:
            return float(value) if value not in [None, np.nan, ''] else default
        except:
            return default

    def analyze_token(self, token):
        try:
            price_data = self.get_token_price_data(token)
            analysis = {
                'symbol': token.get('symbol', 'Unknown'),
                'address': token['address'],
                'price': self.safe_float(price_data.get('price', 0)),
                'liquidity': self.calculate_liquidity_score(token),
                'confidence': price_data.get('confidence', 'medium'),
                'explorer': f"https://solscan.io/token/{token['address']}",
                'score': 0
            }
            analysis['score'] = self.calculate_score(analysis)
            return analysis
        except:
            return None

    def calculate_liquidity_score(self, token):
        try:
            decimals = token.get('decimals', 9)
            amount = int(1000 * (10 ** decimals))
            response = self.session.get(
                f"{JUPITER_QUOTE_API}?inputMint={token['address']}&outputMint={USDC_MINT}&amount={amount}",
                timeout=15
            )
            response.raise_for_status()
            quote = response.json()
            price_impact = self.safe_float(quote.get('priceImpactPct', 1))
            return max(0.0, 100.0 - (price_impact * 10000))
        except:
            return 0.0

    def calculate_score(self, analysis):
        try:
            liquidity = analysis['liquidity'] / 100
            price = analysis['price']
            
            spread = abs(analysis.get('buy_price', 0) - analysis.get('sell_price', 0))
            price_stability = 1 - (spread / price) if price != 0 else 0
            price_stability = np.clip(price_stability, 0, 1)
            
            impact_10 = analysis.get('price_impact_10', 0)
            impact_100 = analysis.get('price_impact_100', 0)
            avg_impact = (impact_10 + impact_100) / 2
            market_depth = 1 - np.clip(avg_impact, 0, 1)
            
            weights = {
                'liquidity': 0.4,
                'price_stability': 0.35,
                'market_depth': 0.25
            }
            
            penalties = 0
            if price <= 0:
                penalties += 0.3
            if analysis['liquidity'] <= 0:
                penalties += 0.2
            if analysis['confidence'] != 'high':
                penalties += 0.1
                
            raw_score = (
                weights['liquidity'] * liquidity +
                weights['price_stability'] * price_stability +
                weights['market_depth'] * market_depth
            )
            
            return (raw_score - penalties) * 100
        except:
            return 0.0

    def get_token_price_data(self, token):
        try:
            response = self.session.get(
                f"{JUPITER_PRICE_API}?ids={token['address']}&showExtraInfo=true",
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            price_data = data.get('data', {}).get(token['address'], {})
            
            return {
                'price': self.safe_float(price_data.get('price', 0)),
                'confidence': price_data.get('extraInfo', {}).get('confidenceLevel', 'medium'),
                'buy_price': self.safe_float(price_data.get('extraInfo', {}).get('quotedPrice', {}).get('buyPrice', 0)),
                'sell_price': self.safe_float(price_data.get('extraInfo', {}).get('quotedPrice', {}).get('sellPrice', 0)),
                'price_impact_10': self.safe_float(price_data.get('extraInfo', {}).get('depth', {}).get('sellPriceImpactRatio', {}).get('depth', {}).get('10', 0)),
                'price_impact_100': self.safe_float(price_data.get('extraInfo', {}).get('depth', {}).get('sellPriceImpactRatio', {}).get('depth', {}).get('100', 0)),
            }
        except:
            return {}

class AnalysisWorker:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.thread = None
        self.running = False

    def start(self, tokens):
        self.running = True
        for token in tokens:
            self.input_queue.put(token)
        self.thread = threading.Thread(target=self.process_tokens, daemon=True)
        self.thread.start()

    def process_tokens(self):
        while self.running and not self.input_queue.empty():
            token = self.input_queue.get()
            result = self.analyzer.analyze_token(token)
            if result:
                self.output_queue.put(result)
            time.sleep(UI_REFRESH_INTERVAL)
        self.running = False

    def get_results(self):
        results = []
        while not self.output_queue.empty():
            results.append(self.output_queue.get())
        return results

class UIManager:
    def __init__(self):
        self.analyzer = TokenAnalyzer()
        self.worker = AnalysisWorker(self.analyzer)
        self.init_session()
        self.inject_styles()

    def init_session(self):
        if 'analysis' not in st.session_state:
            st.session_state.analysis = {
                'running': False,
                'processed': 0,
                'total': 0,
                'results': pd.DataFrame()
            }

    def inject_styles(self):
        st.markdown("""
        <style>
            .stDataFrame { width: 100% !important; }
            .stProgress > div > div > div > div {
                background-color: #4CAF50;
            }
            @media (max-width: 768px) {
                .stButton button { width: 100% !important; }
                .stDataFrame { font-size: 12px; }
            }
            .st-emotion-cache-1hgxyac { padding: 0.5rem; }
        </style>
        """, unsafe_allow_html=True)

    def render_controls(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Start Analysis") and not st.session_state.analysis['running']:
                tokens = self.analyzer.get_all_tokens()
                if tokens:
                    self.worker.start(tokens)
                    st.session_state.analysis.update({
                        'running': True,
                        'total': len(tokens),
                        'results': pd.DataFrame()
                    })
                    st.rerun()
        with col2:
            if st.button("⏹ Stop") and st.session_state.analysis['running']:
                self.worker.running = False
                st.session_state.analysis['running'] = False
                st.rerun()
        with col3:
            if st.button("🧹 Clear Results"):
                st.session_state.analysis['results'] = pd.DataFrame()
                st.rerun()

    def render_progress(self):
        if st.session_state.analysis['running']:
            progress = len(st.session_state.analysis['results']) / st.session_state.analysis['total']
            st.progress(min(progress, 1.0), text=f"Analyzed {len(st.session_state.analysis['results'])}/{st.session_state.analysis['total']} tokens")

    def update_results(self):
        new_results = self.worker.get_results()
        if new_results:
            new_df = pd.DataFrame(new_results)
            st.session_state.analysis['results'] = pd.concat(
                [st.session_state.analysis['results'], new_df],
                ignore_index=True
            ).sort_values('score', ascending=False)
            st.rerun()

    def render_results_table(self):
        if not st.session_state.analysis['results'].empty:
            st.dataframe(
                st.session_state.analysis['results'],
                column_config={
                    "score": st.column_config.NumberColumn(
                        "Score",
                        format="%.1f",
                        width="small"
                    ),
                    "price": st.column_config.NumberColumn(
                        "Price",
                        format="$%.4f"
                    ),
                    "liquidity": st.column_config.ProgressColumn(
                        "Liquidity",
                        format="%.1f",
                        min_value=0,
                        max_value=100
                    ),
                    "confidence": st.column_config.SelectboxColumn(
                        "Confidence",
                        options=['low', 'medium', 'high']
                    ),
                    "explorer": st.column_config.LinkColumn(
                        "Explorer",
                        display_text="View"
                    )
                },
                use_container_width=True,
                height=self.calculate_table_height(),
                hide_index=True
            )

    def calculate_table_height(self):
        num_rows = len(st.session_state.analysis['results'])
        return min(HEADER_HEIGHT + (num_rows * ROW_HEIGHT), 600)

    def render(self):
        st.title("🪙 Solana Token Analyzer")
        self.render_controls()
        self.render_progress()
        self.update_results()
        self.render_results_table()

        if st.session_state.analysis['running']:
            time.sleep(UI_REFRESH_INTERVAL)
            st.rerun()

if __name__ == "__main__":
    ui = UIManager()
    ui.render()
