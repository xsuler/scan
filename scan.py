import streamlit as st
import pandas as pd
import traceback
import requests
import time
import numpy as np
from solders.pubkey import Pubkey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
UI_REFRESH_INTERVAL = 0.01
ROW_HEIGHT = 35
HEADER_HEIGHT = 50

class TokenAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.debug_info = []

    def get_all_tokens(self, strict_checks=True):
        self.debug_info = []
        try:
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
            response.raise_for_status()
            tokens = response.json()
            return [t for t in tokens if self.validate_token(t, strict_checks)]
        except Exception as e:
            st.error(f"Token fetch error: {str(e)}")
            return []

    def validate_token(self, token, strict_checks):
        debug_entry = {
            'symbol': token.get('symbol', 'Unknown'),
            'address': token.get('address', ''),
            'valid': False,
            'reasons': []
        }

        try:
            if not Pubkey.from_string(token.get('address', '')):
                debug_entry['reasons'].append('Invalid address')
                return False

            required_fields = ['symbol', 'name', 'decimals', 'logoURI']
            for field in required_fields:
                if not token.get(field):
                    debug_entry['reasons'].append(f'Missing {field}')
                    return False

            checks = [
                ('tags', lambda: "community" in token.get('tags', [])),
                ('chain', lambda: token.get('chainId') == 101)
            ]

            for check_name, check_func in checks:
                if not check_func():
                    debug_entry['reasons'].append(f'Failed {check_name}')
                    if strict_checks: return False

            debug_entry['valid'] = True
            return True
        except Exception as e:
            debug_entry['reasons'].append(f'Validation error: {str(e)}')
            return False
        finally:
            self.debug_info.append(debug_entry)

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
                'price_type': price_data.get('type', 'N/A'),
                'liquidity': self.calculate_liquidity_score(token),
                'confidence': price_data.get('confidence', 'medium'),
                'buy_price': self.safe_float(price_data.get('buy_price', 0)),
                'sell_price': self.safe_float(price_data.get('sell_price', 0)),
                'price_impact_10': self.safe_float(price_data.get('price_impact_10', 0)),
                'price_impact_100': self.safe_float(price_data.get('price_impact_100', 0)),
                'explorer': f"https://solscan.io/token/{token['address']}",
            }
            
            analysis['score'] = self.calculate_score(analysis)
            return analysis
        except Exception as e:
            st.error(f"Analysis failed for {token.get('symbol')}: {str(e)}")
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
        except Exception as e:
            print(f"Liquidity calculation error: {str(e)}")
            return 0.0

    def calculate_score(self, analysis):
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
            
            final_score = (raw_score - penalties) * 100
            return final_score
            
        except KeyError as e:
            print(f"Missing key in score calculation: {str(e)}")
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
                'type': price_data.get('type', 'N/A'),
                'confidence': price_data.get('extraInfo', {}).get('confidenceLevel', 'medium'),
                'buy_price': self.safe_float(price_data.get('extraInfo', {}).get('quotedPrice', {}).get('buyPrice', 0)),
                'sell_price': self.safe_float(price_data.get('extraInfo', {}).get('quotedPrice', {}).get('sellPrice', 0)),
                'price_impact_10': self.safe_float(price_data.get('extraInfo', {}).get('depth', {}).get('sellPriceImpactRatio', {}).get('depth', {}).get('10', 0)),
                'price_impact_100': self.safe_float(price_data.get('extraInfo', {}).get('depth', {}).get('sellPriceImpactRatio', {}).get('depth', {}).get('100', 0)),
            }
        except Exception as e:
            st.error(f"Price data error: {str(e)}")
            return {}

class AnalysisManager:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.init_session()
        self.lock = threading.Lock()
        
    def init_session(self):
        defaults = {
            'analysis': {
                'running': False,
                'progress': 0,
                'params': None,
                'metrics': {'start_time': None, 'speed': 0},
                'current_index': 0,
                'current_token': None,
                'tokens': []
            },
            'live_results': pd.DataFrame()
        }
        for key, val in defaults.items():
            st.session_state.setdefault(key, val)

    def start_analysis(self, tokens, params):
        with self.lock:
            st.session_state.analysis = {
                'running': True,
                'progress': 0,
                'params': params,
                'metrics': {
                    'start_time': time.time(),
                    'speed': 0,
                    'total_tokens': len(tokens)
                },
                'current_index': 0,
                'tokens': tokens,
                'current_token': None
            }
            st.session_state.live_results = pd.DataFrame()
        
        # Start background thread
        threading.Thread(target=self.process_tokens, daemon=True).start()

    def process_tokens(self):
        while True:
            with self.lock:
                if not st.session_state.analysis['running']:
                    break

                idx = st.session_state.analysis['current_index']
                tokens = st.session_state.analysis['tokens']
                
                if idx >= len(tokens):
                    self.finalize_analysis()
                    break

                token = tokens[idx]
                analysis = self.analyzer.analyze_token(token)
                
                if analysis is not None:
                    st.session_state.analysis['current_token'] = token
                    new_row = pd.DataFrame([analysis])
                    st.session_state.live_results = pd.concat(
                        [st.session_state.live_results, new_row],
                        ignore_index=True
                    )

                st.session_state.analysis['progress'] = idx + 1
                st.session_state.analysis['current_index'] += 1
                st.session_state.analysis['metrics']['speed'] = (idx + 1) / (
                    time.time() - st.session_state.analysis['metrics']['start_time']
                )

                time.sleep(UI_REFRESH_INTERVAL)
                
            # Update UI through rerun
            st.experimental_rerun()

    def finalize_analysis(self):
        with self.lock:
            st.session_state.analysis['running'] = False
            if st.session_state.live_results.empty:
                st.warning("No qualified tokens found during analysis")

class UIManager:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.manager = AnalysisManager(analyzer)
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

            .main .block-container {
                padding-top: 2rem;
                position: relative;
                z-index: 1;
            }

            @media (max-width: 768px) {
                div.stButton > button {
                    width: 100% !important;
                    margin: 8px 0 !important;
                }
                
                div[data-testid="column"] {
                    flex: 0 0 100% !important;
                    width: 100% !important;
                }
                
                div[data-testid="stDataFrame"] {
                    font-size: 14px;
                }
            }

            @media (min-width: 769px) {
                div[data-testid="stHorizontalBlock"] {
                    gap: 1rem;
                }
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

            with st.container():
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    if st.button("🚀 Start", use_container_width=True):
                        self.start_analysis(params)
                with cols[1]:
                    if st.button("⏹ Stop", use_container_width=True):
                        self.manager.finalize_analysis()
                with cols[2]:
                    if st.button("🧹 Clear", use_container_width=True):
                        st.session_state.live_results = pd.DataFrame()
                        st.experimental_rerun()

            if st.session_state.analysis.get('running', False):
                self.render_progress()
                self.render_current_token()

            if not st.session_state.live_results.empty:
                st.divider()
                self.render_sidebar_results()

    def render_progress(self):
        analysis = st.session_state.analysis
        metrics = analysis['metrics']
        progress = analysis['progress'] / metrics['total_tokens']
        
        st.progress(min(progress, 1.0), text="Analysis Progress")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Processed", f"{analysis['progress']}/{metrics['total_tokens']}")
        with cols[1]:
            st.metric("Speed", f"{(analysis['metrics']['speed']):.1f} tkn/s")
        with cols[2]:
            st.metric("Found", f"{len(st.session_state.live_results)}")

    def render_current_token(self):
        current_token = st.session_state.analysis.get('current_token')
        if not current_token:
            return

        with st.expander("🔍 Current Token", expanded=True):
            st.subheader(current_token.get('symbol', 'Unknown'))
            st.caption(f"`{current_token.get('address', '')[:12]}...`")
            st.write(f"**Chain:** Solana Mainnet")
            st.write(f"**Decimals:** {current_token.get('decimals', 'N/A')}")

    def start_analysis(self, params):
        tokens = self.analyzer.get_all_tokens(params['strict_mode'])
        if tokens:
            self.manager.start_analysis(tokens, params)
        else:
            st.error("No tokens found matching criteria")

    def calculate_table_height(self, df):
        num_rows = len(df)
        base_height = HEADER_HEIGHT + num_rows * ROW_HEIGHT
        return min(base_height, 600)

    def render_sidebar_results(self):
        st.subheader("📊 Live Results")
        df = st.session_state.live_results
        
        sorted_df = df.sort_values(by='score', ascending=False)
        sorted_df = sorted_df[['score', 'symbol', 'price', 'liquidity', 'confidence', 'explorer']]

        table_height = self.calculate_table_height(sorted_df)

        st.data_editor(
            sorted_df,
            column_config={
                'score': st.column_config.NumberColumn('Score', format="%.1f"),
                'symbol': st.column_config.TextColumn('Token'),
                'price': st.column_config.NumberColumn('Price', format="$%.4f"),
                'liquidity': st.column_config.ProgressColumn('Liquidity', format="%.1f"),
                'confidence': st.column_config.SelectboxColumn('Confidence'),
                'explorer': st.column_config.LinkColumn('Explorer')
            },
            height=table_height,
            use_container_width=True,
            hide_index=True,
            key="live_results_table"
        )
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Avg Score", f"{df['score'].mean():.1f}")
        with cols[1]:
            st.metric("Top Score", f"{df['score'].max():.1f}")
        with cols[2]:
            st.metric("High Conf", df[df['confidence'] == 'high'].shape[0])

    def render_main(self):
        pass  # Background thread handles processing

if __name__ == "__main__":
    analyzer = TokenAnalyzer()
    ui = UIManager(analyzer)
    ui.render_sidebar()
    ui.render_main()
