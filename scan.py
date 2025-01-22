import streamlit as st
import pandas as pd
import traceback
import requests
import time
import numpy as np
from solders.pubkey import Pubkey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from threading import Thread, Lock
import queue
import copy

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
UI_REFRESH_INTERVAL = 0.01
ROW_HEIGHT = 35
HEADER_HEIGHT = 50

class BackgroundAnalyzer:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = Lock()
        self.worker = None
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
    def start_analysis(self, tokens, params):
        with self.lock:
            if self.worker and self.worker.is_alive():
                return False
            
            # Clear queues
            while not self.task_queue.empty():
                self.task_queue.get()
            while not self.result_queue.empty():
                self.result_queue.get()
            
            # Add all tokens to task queue
            for token in tokens:
                self.task_queue.put((token, params))
            
            # Start worker thread
            self.worker = Thread(target=self._process_tokens)
            self.worker.daemon = True
            self.worker.start()
            return True

    def _process_tokens(self):
        while not self.task_queue.empty():
            try:
                token, params = self.task_queue.get()
                result = self._analyze_token(token, params)
                if result:
                    self.result_queue.put(result)
            except Exception as e:
                self.result_queue.put({'error': str(e)})
            finally:
                self.task_queue.task_done()
                time.sleep(0.01)  # Prevent resource hogging

    def _analyze_token(self, token, params):
        try:
            price_data = self._get_token_price_data(token)
            analysis = {
                'symbol': token.get('symbol', 'Unknown'),
                'address': token['address'],
                'price': self._safe_float(price_data.get('price', 0)),
                'price_type': price_data.get('type', 'N/A'),
                'liquidity': self._calculate_liquidity_score(token),
                'confidence': price_data.get('confidence', 'medium'),
                'buy_price': self._safe_float(price_data.get('buy_price', 0)),
                'sell_price': self._safe_float(price_data.get('sell_price', 0)),
                'price_impact_10': self._safe_float(price_data.get('price_impact_10', 0)),
                'price_impact_100': self._safe_float(price_data.get('price_impact_100', 0)),
                'explorer': f"https://solscan.io/token/{token['address']}",
            }
            analysis['score'] = self._calculate_score(analysis)
            return analysis
        except Exception as e:
            return {'error': f"{token.get('symbol')}: {str(e)}"}

    def get_progress(self):
        with self.lock:
            if self.task_queue.empty():
                return 1.0
            size = self.task_queue.qsize()
            total = size + self.task_queue.unfinished_tasks
            return 1 - (size / total) if total > 0 else 0

    def get_results(self):
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results

    def _safe_float(self, value, default=0.0):
        try:
            return float(value) if value not in [None, np.nan, ''] else default
        except:
            return default

    def _calculate_liquidity_score(self, token):
        try:
            decimals = token.get('decimals', 9)
            amount = int(1000 * (10 ** decimals))
            response = self.session.get(
                f"{JUPITER_QUOTE_API}?inputMint={token['address']}&outputMint={USDC_MINT}&amount={amount}",
                timeout=15
            )
            response.raise_for_status()
            quote = response.json()
            price_impact = self._safe_float(quote.get('priceImpactPct', 1))
            return max(0.0, 100.0 - (price_impact * 10000))
        except Exception as e:
            return 0.0

    def _calculate_score(self, analysis):
        try:
            liquidity = analysis['liquidity'] / 100
            price = analysis['price']
            buy_price = analysis['buy_price']
            sell_price = analysis['sell_price']
            
            if price <= 0 or (buy_price == 0 and sell_price == 0):
                price_stability = 0.0
            else:
                spread = abs(buy_price - sell_price)
                price_stability = 1 - (spread / price) if price != 0 else 0
                price_stability = np.clip(price_stability, 0, 1)
            
            impact_10 = analysis['price_impact_10']
            impact_100 = analysis['price_impact_100']
            avg_impact = (impact_10 + impact_100) / 2
            market_depth = 1 - np.clip(avg_impact, 0, 1)
            
            weights = {'liquidity': 0.4, 'price_stability': 0.35, 'market_depth': 0.25}
            penalties = 0
            if price <= 0: penalties += 0.3
            if analysis['liquidity'] <= 0: penalties += 0.2
            if analysis['confidence'] != 'high': penalties += 0.1
                
            raw_score = (weights['liquidity'] * liquidity +
                        weights['price_stability'] * price_stability +
                        weights['market_depth'] * market_depth)
            
            return (raw_score - penalties) * 100
        except KeyError as e:
            return 0.0

    def _get_token_price_data(self, token):
        try:
            response = self.session.get(
                f"{JUPITER_PRICE_API}?ids={token['address']}&showExtraInfo=true",
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            price_data = data.get('data', {}).get(token['address'], {})
            return {
                'price': self._safe_float(price_data.get('price', 0)),
                'type': price_data.get('type', 'N/A'),
                'confidence': price_data.get('extraInfo', {}).get('confidenceLevel', 'medium'),
                'buy_price': self._safe_float(price_data.get('extraInfo', {}).get('quotedPrice', {}).get('buyPrice', 0)),
                'sell_price': self._safe_float(price_data.get('extraInfo', {}).get('quotedPrice', {}).get('sellPrice', 0)),
                'price_impact_10': self._safe_float(price_data.get('extraInfo', {}).get('depth', {}).get('sellPriceImpactRatio', {}).get('depth', {}).get('10', 0)),
                'price_impact_100': self._safe_float(price_data.get('extraInfo', {}).get('depth', {}).get('sellPriceImpactRatio', {}).get('depth', {}).get('100', 0)),
            }
        except Exception as e:
            return {}

class AnalysisManager:
    def __init__(self):
        self.bg_analyzer = BackgroundAnalyzer()
        self.init_session()
        
    def init_session(self):
        if 'analysis' not in st.session_state:
            st.session_state.analysis = {
                'running': False,
                'params': None,
                'tokens': [],
                'results': pd.DataFrame(),
                'start_time': None,
                'total_tokens': 0
            }

    def start_analysis(self, tokens, params):
        st.session_state.analysis.update({
            'running': True,
            'params': params,
            'tokens': copy.deepcopy(tokens),
            'results': pd.DataFrame(),
            'start_time': time.time(),
            'total_tokens': len(tokens)
        })
        self.bg_analyzer.start_analysis(tokens, params)

    def update_results(self):
        if st.session_state.analysis['running']:
            new_results = self.bg_analyzer.get_results()
            if new_results:
                valid_results = [r for r in new_results if 'error' not in r]
                if valid_results:
                    new_df = pd.DataFrame(valid_results)
                    st.session_state.analysis['results'] = pd.concat(
                        [st.session_state.analysis['results'], new_df],
                        ignore_index=True
                    )
                
                errors = [r['error'] for r in new_results if 'error' in r]
                for error in errors:
                    st.error(f"Analysis error: {error}")

            # Check completion
            progress = self.bg_analyzer.get_progress()
            if progress >= 1.0:
                st.session_state.analysis['running'] = False
                if st.session_state.analysis['results'].empty:
                    st.warning("No qualified tokens found during analysis")

class UIManager:
    def __init__(self):
        self.manager = AnalysisManager()
        self.inject_responsive_css()

    def inject_responsive_css(self):
        st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                width: 100% !important;
                min-width: 100% !important;
                transform: translateX(0) !important;
                z-index: 999999;
            }
            [data-testid="collapsedControl"] {
                display: none !important;
            }
            @media (max-width: 768px) {
                div.stButton > button { width: 100% !important; }
                div[data-testid="column"] { width: 100% !important; }
                div[data-testid="stDataFrame"] { font-size: 14px; }
            }
        </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        with st.sidebar:
            st.title("🪙 Solana Token Analyzer")
            st.image("https://jup.ag/svg/jupiter-logo.svg", width=200)
            
            with st.expander("⚙️ Control Panel", expanded=True):
                params = {
                    'strict_mode': st.checkbox("Strict Validation", True),
                    'live_sorting': st.checkbox("Real-time Sorting", True)
                }

            cols = st.columns([1, 1, 1])
            with cols[0]:
                if st.button("🚀 Start"):
                    self.start_analysis(params)
            with cols[1]:
                if st.button("⏹ Stop"):
                    st.session_state.analysis['running'] = False
            with cols[2]:
                if st.button("🧹 Clear"):
                    st.session_state.analysis['results'] = pd.DataFrame()

            if st.session_state.analysis['running']:
                self.render_progress()

            if not st.session_state.analysis['results'].empty:
                st.divider()
                self.render_sidebar_results()

    def start_analysis(self, params):
        try:
            response = requests.get(JUPITER_TOKEN_LIST, timeout=15)
            response.raise_for_status()
            tokens = response.json()
            valid_tokens = [t for t in tokens if self.validate_token(t, params['strict_mode'])]
            
            if valid_tokens:
                self.manager.start_analysis(valid_tokens, params)
            else:
                st.error("No valid tokens found")
        except Exception as e:
            st.error(f"Failed to start analysis: {str(e)}")

    def validate_token(self, token, strict_checks):
        try:
            if not Pubkey.from_string(token.get('address', '')):
                return False

            required_fields = ['symbol', 'name', 'decimals', 'logoURI']
            for field in required_fields:
                if not token.get(field):
                    return False

            if strict_checks:
                if "community" in token.get('tags', []):
                    return False
                if token.get('chainId') != 101:
                    return False
            return True
        except:
            return False

    def render_progress(self):
        analysis = st.session_state.analysis
        progress = self.manager.bg_analyzer.get_progress()
        
        st.progress(progress)
        cols = st.columns(3)
        with cols[0]:
            st.metric("Processed", f"{int(progress * analysis['total_tokens'])}/{analysis['total_tokens']}")
        with cols[1]:
            elapsed = time.time() - analysis['start_time']
            speed = analysis['total_tokens'] * progress / elapsed if elapsed > 0 else 0
            st.metric("Speed", f"{speed:.1f} tkn/s")
        with cols[2]:
            st.metric("Found", len(analysis['results']))

    def render_sidebar_results(self):
        st.subheader("📊 Live Results")
        df = st.session_state.analysis['results']
        
        if df.empty:
            return

        sorted_df = df.sort_values(by='score', ascending=False)
        sorted_df = sorted_df[['score', 'symbol', 'price', 'liquidity', 'confidence', 'explorer']]

        st.data_editor(
            sorted_df,
            column_config={
                'score': st.column_config.NumberColumn('Score', format="%.1f"),
                'symbol': st.column_config.TextColumn('Token'),
                'price': st.column_config.NumberColumn('Price', format="$%.4f"),
                'liquidity': st.column_config.ProgressColumn('Liquidity', format="%.1f", min_value=0, max_value=100),
                'confidence': st.column_config.SelectboxColumn('Confidence', options=['low', 'medium', 'high']),
                'explorer': st.column_config.LinkColumn('Explorer'),
            },
            height=self.calculate_table_height(sorted_df),
            use_container_width=True,
            hide_index=True,
        )

    def calculate_table_height(self, df):
        return min(HEADER_HEIGHT + len(df) * ROW_HEIGHT, 600)

    def render_main(self):
        self.manager.update_results()

if __name__ == "__main__":
    ui = UIManager()
    ui.render_sidebar()
    ui.render_main()
