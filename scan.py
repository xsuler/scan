import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
from solders.pubkey import Pubkey
from solana.rpc.api import Client
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
SOLANA_RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"
COINGECKO_API = "https://api.coingecko.com/api/v3"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
UI_REFRESH_INTERVAL = 0.01

class TokenAnalyzer:
    def __init__(self):
        self.client = Client(SOLANA_RPC_ENDPOINT)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.debug_info = []

    def get_all_tokens(self, strict_checks=True):
        self.debug_info = []
        try:
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
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
                ('coingecko', lambda: 'coingeckoId' in token.get('extensions', {})),
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
            analysis = {
                'symbol': token.get('symbol', 'Unknown'),
                'address': token['address'],
                'price': self.get_token_price(token),
                'market_cap': self.calculate_market_cap(token),
                'liquidity': self.calculate_liquidity_score(token),
                'rating': 0,
                'volume': self.get_trading_volume(token),
                'holders': self.get_holder_count(token),
                'explorer': f"https://solscan.io/token/{token['address']}",
                'age': self.get_token_age(token)
            }
            analysis['rating'] = self.calculate_rating(analysis)
            return analysis
        except Exception as e:
            st.error(f"Analysis failed for {token.get('symbol')}: {str(e)}")
            return None

    def calculate_market_cap(self, token):
        try:
            supply = self.get_circulating_supply(token)
            return float(supply * self.get_token_price(token))
        except:
            return 0.0

    def calculate_liquidity_score(self, token):
        try:
            decimals = token.get('decimals', 9)
            amount = int(1000 * (10 ** decimals))
            quote = self.session.get(
                f"https://quote-api.jup.ag/v6/quote?inputMint={token['address']}&outputMint={USDC_MINT}&amount={amount}",
                timeout=15
            ).json()
            return max(0.0, 100.0 - (float(quote.get('priceImpactPct', 1)) * 10000))
        except:
            return 0.0

    def calculate_rating(self, analysis):
        try:
            return min(100.0, (analysis['liquidity'] * 0.4) + 
                      (analysis['market_cap'] / 1e6 * 0.3) + 
                      (analysis['volume'] / 1e4 * 0.3))
        except:
            return 0.0

    def get_circulating_supply(self, token):
        try:
            account = self.client.get_account_info_json_parsed(Pubkey.from_string(token['address'])).value
            return float(account.data.parsed['info']['supply'] / 10 ** token.get('decimals', 9))
        except:
            return 0.0

    def get_token_price(self, token):
        try:
            return float(token.get('price', 0))
        except:
            return 0.0

    def get_holder_count(self, token):
        try:
            return int(self.client.get_token_supply(Pubkey.from_string(token['address'])).value.amount)
        except:
            return 0

    def get_trading_volume(self, token):
        try:
            response = requests.get(
                f"{COINGECKO_API}/coins/{token['extensions']['coingeckoId']}/market_chart",
                params={'vs_currency': 'usd', 'days': '1'},
                timeout=15
            )
            volumes = response.json().get('total_volumes', [])
            if volumes:
                return float(volumes[-1][1])
            return 0.0
        except Exception as e:
            print(f"Volume error: {str(e)}")
            return 0.0

    def get_token_age(self, token):
        try:
            return float((time.time() - token.get('timestamp', time.time())) / 86400)
        except:
            return 0.0

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
                'metrics': {'start_time': None, 'speed': 0}
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
            }
        }
        self.process_tokens(tokens)

    def process_tokens(self, tokens):
        for idx, token in enumerate(tokens):
            if not st.session_state.analysis['running']:
                break

            analysis = self.analyzer.analyze_token(token)
            if analysis and analysis['rating'] >= st.session_state.analysis['params']['min_rating']:
                st.session_state.analysis['results'].append(analysis)

            st.session_state.analysis['progress'] = idx + 1
            time.sleep(UI_REFRESH_INTERVAL)
            st.rerun()

        self.finalize_analysis()

    def finalize_analysis(self):
        st.session_state.analysis['running'] = False
        st.session_state.results = pd.DataFrame(st.session_state.analysis['results'])
        st.rerun()

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
                    'min_mcap': st.number_input("Market Cap Floor (USD)", 1000, 10000000, 10000),
                    'strict_mode': st.checkbox("Strict Validation", True)
                }

            with st.expander("📊 Advanced Filters"):
                params.update({
                    'max_age': st.number_input("Max Token Age (days)", 1, 365, 30),
                    'min_liquidity': st.slider("Min Liquidity Score", 0, 100, 50),
                    'min_volume': st.number_input("Min Daily Volume (USD)", 0, 1000000, 1000)
                })

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
        
        st.progress(progress)
        st.metric("Processed Tokens", f"{analysis['progress']}/{metrics['total_tokens']}")
        st.metric("Analysis Speed", f"{(analysis['progress']/elapsed):.1f} tkn/s" if elapsed > 0 else "N/A")
        st.metric("Elapsed Time", f"{elapsed:.1f}s")

    def render_current_token(self):
        if st.session_state.analysis['results']:
            token = st.session_state.analysis['results'][-1]
            with st.expander("Current Token Details", expanded=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(token.get('logoURI', 'https://via.placeholder.com/80'), width=60)
                with cols[1]:
                    st.subheader(f"{token.get('symbol', 'Unknown')}")
                    st.caption(f"`{token.get('address', '')[6:]}...`")
                
                st.markdown(f"""
                **Price:** ${token.get('price', 0.0):.4f}  
                **Market Cap:** ${token.get('market_cap', 0.0):,.0f}  
                **Rating:** {token.get('rating', 0.0):.1f}/100  
                **Volume (24h):** ${token.get('volume', 0.0):,.0f}
                """)

    def start_analysis(self, params):
        tokens = self.analyzer.get_all_tokens(params['strict_mode'])
        if tokens:
            self.manager.start_analysis(tokens, params)
        else:
            st.error("No tokens found matching criteria")

    def stop_analysis(self):
        st.session_state.analysis['running'] = False
        st.rerun()

    def render_main(self):
        st.title("Real-time Solana Token Analysis")
        
        if st.session_state.get('results') is not None:
            self.render_results()

    def render_results(self):
        tab1, tab2 = st.tabs(["Analysis Results", "Market Overview"])
        
        with tab1:
            self.render_results_table()
        
        with tab2:
            self.render_market_charts()

    def render_results_table(self):
        df = st.session_state.results.sort_values('rating', ascending=False)
        st.dataframe(
            df,
            column_config={
                'symbol': 'Symbol',
                'price': st.column_config.NumberColumn('Price', format="$%.4f"),
                'market_cap': st.column_config.NumberColumn('Market Cap', format="$%.2f"),
                'rating': st.column_config.ProgressColumn('Rating', format="%.1f"),
                'liquidity': 'Liquidity Score',
                'volume': st.column_config.NumberColumn('Volume', format="$%.0f"),
                'age': 'Token Age'
            },
            height=600,
            use_container_width=True
        )

    def render_market_charts(self):
        df = st.session_state.results
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='market_cap', y='rating', 
                           hover_data=['symbol'], log_x=True,
                           title="Market Cap vs Rating")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='liquidity', nbins=20,
                             title="Liquidity Distribution")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    analyzer = TokenAnalyzer()
    ui = UIManager(analyzer)
    ui.render_sidebar()
    ui.render_main()
