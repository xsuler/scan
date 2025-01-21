import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
from solders.pubkey import Pubkey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
UI_REFRESH_INTERVAL = 0.01

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

    def analyze_token(self, token):
        try:
            price_data = self.get_token_price_data(token)
            if not price_data:
                return None
                
            analysis = {
                'symbol': token.get('symbol', 'Unknown'),
                'address': token['address'],
                'price': float(price_data.get('price', 0)),
                'price_type': price_data.get('type', 'N/A'),
                'liquidity': self.calculate_liquidity_score(token),
                'confidence': price_data.get('confidence', 'medium'),
                'buy_price': float(price_data.get('buy_price', 0)),
                'sell_price': float(price_data.get('sell_price', 0)),
                'price_impact_10': float(price_data.get('price_impact_10', 0)),
                'price_impact_100': float(price_data.get('price_impact_100', 0)),
                'last_buy_price': float(price_data.get('last_buy_price', 0)),
                'last_sell_price': float(price_data.get('last_sell_price', 0)),
                'explorer': f"https://solscan.io/token/{token['address']}",
            }
            
            analysis['rating'] = self.calculate_rating(analysis)
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
            return max(0.0, 100.0 - (float(quote.get('priceImpactPct', 1)) * 10000))
        except Exception as e:
            print(f"Liquidity calculation error: {str(e)}")
            return 0.0

    def calculate_rating(self, analysis):
        try:
            liquidity_score = analysis['liquidity'] * 0.4
            price_stability = (1 - abs(analysis['buy_price'] - analysis['sell_price'])/analysis['price']) * 0.3 if analysis['price'] != 0 else 0
            market_depth = (1 - (analysis['price_impact_10'] + analysis['price_impact_100'])/2) * 0.3
            
            return min(100.0, (liquidity_score + price_stability + market_depth) * 100)
        except KeyError as e:
            print(f"Missing key in rating calculation: {str(e)}")
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
            if not price_data:
                return None
                
            extra_info = price_data.get('extraInfo', {})
            return {
                'price': price_data.get('price', 0),
                'type': price_data.get('type', 'N/A'),
                'confidence': extra_info.get('confidenceLevel', 'medium'),
                'buy_price': extra_info.get('quotedPrice', {}).get('buyPrice', 0),
                'sell_price': extra_info.get('quotedPrice', {}).get('sellPrice', 0),
                'price_impact_10': extra_info.get('depth', {})
                                   .get('sellPriceImpactRatio', {})
                                   .get('depth', {})
                                   .get('10', 0),
                'price_impact_100': extra_info.get('depth', {})
                                    .get('sellPriceImpactRatio', {})
                                    .get('depth', {})
                                    .get('100', 0),
                'last_buy_price': extra_info.get('lastSwappedPrice', {})
                                   .get('lastJupiterBuyPrice', 0),
                'last_sell_price': extra_info.get('lastSwappedPrice', {})
                                    .get('lastJupiterSellPrice', 0)
            }
        except Exception as e:
            st.error(f"Price data error: {str(e)}")
            return None

class AnalysisManager:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.init_session()
        
    def init_session(self):
        defaults = {
            'analysis': {
                'running': False,
                'progress': 0,
                'results': [],
                'params': None,
                'metrics': {'start_time': None, 'speed': 0},
                'current_index': 0,
                'current_token': None,
                'current_analysis': None
            }
        }
        for key, val in defaults.items():
            st.session_state.setdefault(key, val)

    def start_analysis(self, tokens, params):
        st.session_state.analysis = {
            'running': True,
            'progress': 0,
            'results': [],
            'params': params,
            'metrics': {
                'start_time': time.time(),
                'speed': 0,
                'total_tokens': len(tokens)
            },
            'current_index': 0,
            'tokens': tokens,
            'current_token': None,
            'current_analysis': None
        }

    def process_tokens(self):
        if not st.session_state.analysis['running']:
            return

        idx = st.session_state.analysis['current_index']
        tokens = st.session_state.analysis['tokens']
        
        if idx < len(tokens):
            token = tokens[idx]
            analysis = self.analyzer.analyze_token(token)
            
            # Only update current analysis if successful
            if analysis:
                st.session_state.analysis['current_token'] = token
                st.session_state.analysis['current_analysis'] = analysis
                
                if analysis['rating'] >= st.session_state.analysis['params']['min_rating']:
                    st.session_state.analysis['results'].append(analysis)

            # Always update progress
            st.session_state.analysis['progress'] = idx + 1
            st.session_state.analysis['current_index'] += 1
            st.session_state.analysis['metrics']['speed'] = (idx + 1) / (
                time.time() - st.session_state.analysis['metrics']['start_time']
            )
            
            time.sleep(UI_REFRESH_INTERVAL)
            st.rerun()
        else:
            self.finalize_analysis()

    def finalize_analysis(self):
        st.session_state.analysis['running'] = False
        if st.session_state.analysis['results']:
            st.session_state.results = pd.DataFrame(st.session_state.analysis['results'])
        else:
            st.session_state.results = pd.DataFrame()

class UIManager:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.manager = AnalysisManager(analyzer)

    def render_sidebar(self):
        with st.sidebar:
            st.image("https://jup.ag/svg/jupiter-logo.svg", width=200)
            st.header("Analysis Controls")
            
            with st.expander("⚙️ Core Settings", expanded=True):
                params = {
                    'min_rating': st.slider("Minimum Rating", 0, 100, 65),
                    'strict_mode': st.checkbox("Strict Validation", True),
                    'min_liquidity': st.slider("Min Liquidity Score", 0, 100, 50)
                }

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Start Analysis", use_container_width=True):
                    self.start_analysis(params)
            with col2:
                if st.button("⏹ Stop Analysis", use_container_width=True):
                    self.stop_analysis()

            st.divider()
            if st.button("🧹 Clear Data"):
                st.session_state.clear()
                st.rerun()

            if st.session_state.analysis.get('running', False):
                self.render_progress()
                self.render_current_token()

    def render_progress(self):
        analysis = st.session_state.analysis
        metrics = analysis['metrics']
        progress = analysis['progress'] / metrics['total_tokens']
        elapsed = time.time() - metrics['start_time']
        
        st.progress(min(progress, 1.0))
        st.metric("Processed Tokens", f"{analysis['progress']}/{metrics['total_tokens']}")
        st.metric("Analysis Speed", f"{(analysis['progress']/elapsed):.1f} tkn/s" if elapsed > 0 else "N/A")
        st.metric("Elapsed Time", f"{elapsed:.1f}s")

    def render_current_token(self):
        analysis_state = st.session_state.analysis
        current_token = analysis_state.get('current_token')
        current_analysis = analysis_state.get('current_analysis')

        if not current_token or not current_analysis:
            return

        with st.expander("Current Token Details", expanded=True):
            symbol = current_token.get('symbol', 'Unknown')
            st.subheader(f"{symbol}")
            address = current_token.get('address', '')
            shortened_address = f"{address[:6]}...{address[-4:]}" if address else ''
            st.caption(f"`{shortened_address}`")
            
            cols = st.columns(2)
            with cols[0]:
                st.metric("Current Price", f"${current_analysis.get('price', 0):.4f}")
                st.metric("Buy Price", f"${current_analysis.get('buy_price', 0):.4f}")
                st.metric("Price Impact (10)", 
                         f"{current_analysis.get('price_impact_10', 0)*100:.2f}%")
                
            with cols[1]:
                st.metric("Confidence Level", current_analysis.get('confidence', 'N/A'))
                st.metric("Sell Price", f"${current_analysis.get('sell_price', 0):.4f}")
                st.metric("Price Impact (100)", 
                         f"{current_analysis.get('price_impact_100', 0)*100:.2f}%")

    def start_analysis(self, params):
        tokens = self.analyzer.get_all_tokens(params['strict_mode'])
        if tokens:
            self.manager.start_analysis(tokens, params)
            self.manager.process_tokens()
        else:
            st.error("No tokens found matching criteria")

    def stop_analysis(self):
        st.session_state.analysis['running'] = False
        st.rerun()

    def render_main(self):
        st.title("Advanced Solana Token Analyzer")
        
        if st.session_state.analysis.get('running', False):
            self.manager.process_tokens()
        
        if st.session_state.get('results') is not None:
            tab1, tab2, tab3 = st.tabs(["Analysis Results", "Market Depth", "Price Dynamics"])
            
            with tab1:
                self.render_results_table()
            
            with tab2:
                self.render_market_depth_charts()
            
            with tab3:
                self.render_price_analysis()

    def render_results_table(self):
        if st.session_state.results.empty:
            st.warning("No tokens matching the criteria found")
            return
            
        df = st.session_state.results.sort_values('rating', ascending=False)
        st.dataframe(
            df,
            column_config={
                'symbol': 'Symbol',
                'price': st.column_config.NumberColumn('Price', format="$%.4f"),
                'rating': st.column_config.ProgressColumn('Rating', format="%.1f"),
                'liquidity': 'Liquidity Score',
                'confidence': 'Confidence',
                'buy_price': st.column_config.NumberColumn('Buy Price', format="$%.4f"),
                'sell_price': st.column_config.NumberColumn('Sell Price', format="$%.4f"),
            },
            height=600,
            use_container_width=True
        )

    def render_market_depth_charts(self):
        if st.session_state.results.empty:
            return
            
        df = st.session_state.results
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='price_impact_10', y='price_impact_100',
                           color='confidence', hover_data=['symbol'],
                           title="Price Impact Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.density_heatmap(df, x='liquidity', y='rating',
                                   marginal_x="histogram", marginal_y="histogram",
                                   title="Liquidity vs Rating Distribution")
            st.plotly_chart(fig, use_container_width=True)

    def render_price_analysis(self):
        if st.session_state.results.empty:
            return
            
        df = st.session_state.results
        fig = px.scatter_3d(df, x='price', y='buy_price', z='sell_price',
                          color='rating', hover_name='symbol',
                          title="3D Price Relationship Analysis")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    analyzer = TokenAnalyzer()
    ui = UIManager(analyzer)
    ui.render_sidebar()
    ui.render_main()
