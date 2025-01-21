import streamlit as st
import pandas as pd
import requests
import json
import os
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
RESULTS_FILE = "token_analysis.csv"
CHECKPOINT_FILE = "analysis_checkpoint.json"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
BATCH_SIZE = 5
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
            return supply * self.get_token_price(token)
        except:
            return 0

    def calculate_liquidity_score(self, token):
        try:
            decimals = token.get('decimals', 9)
            amount = int(1000 * (10 ** decimals))
            quote = self.session.get(
                f"https://quote-api.jup.ag/v6/quote?inputMint={token['address']}&outputMint={USDC_MINT}&amount={amount}",
                timeout=15
            ).json()
            return max(0, 100 - (float(quote.get('priceImpactPct', 1)) * 10000))
        except:
            return 0

    def calculate_rating(self, analysis):
        try:
            return min(100, (analysis['liquidity'] * 0.4) + 
                       (analysis['market_cap'] / 1e6 * 0.3) + 
                       (analysis['volume'] / 1e4 * 0.3))
        except:
            return 0

    def get_circulating_supply(self, token):
        try:
            account = self.client.get_account_info_json_parsed(Pubkey.from_string(token['address'])).value
            return account.data.parsed['info']['supply'] / 10 ** token.get('decimals', 9)
        except:
            return 0

    def get_token_price(self, token):
        try:
            return float(token.get('price', 0))
        except:
            return 0

    def get_holder_count(self, token):
        try:
            return self.client.get_token_supply(Pubkey.from_string(token['address'])).value.amount
        except:
            return 0

    def get_trading_volume(self, token):
        try:
            return requests.get(
                f"{COINGECKO_API}/coins/{token['extensions']['coingeckoId']}/market_chart",
                params={'vs_currency': 'usd', 'days': '1'}
            ).json()['total_volumes'][-1][1]
        except:
            return 0

    def get_token_age(self, token):
        try:
            return (time.time() - token.get('timestamp', time.time())) / 86400
        except:
            return 0

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
            },
            'ui': {'container': None, 'progress': None, 'metrics': None}
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
        self.create_ui()
        self.process_tokens(tokens)

    def create_ui(self):
        # Create container and columns in separate contexts
        st.session_state.ui['container'] = st.empty()
        with st.session_state.ui['container']:
            cols = st.columns([3, 1])
            with cols[0]:
                st.session_state.ui['progress'] = st.progress(0)
                st.session_state.ui['status'] = st.empty()
            with cols[1]:
                st.session_state.ui['metrics'] = st.empty()

    def update_ui(self, token):
        analysis = st.session_state.analysis
        metrics = analysis['metrics']
        elapsed = time.time() - metrics['start_time']
        speed = analysis['progress'] / elapsed if elapsed > 0 else 0

        with st.session_state.ui['container']:
            cols = st.columns([3, 1])
            with cols[0]:
                st.session_state.ui['progress'].progress(
                    analysis['progress'] / metrics['total_tokens']
                )
                status_text = f"""
                **Current Token:** {token.get('symbol', 'Unknown')}  
                **Address:** `{token['address'][:6]}...{token['address'][-4:]}`  
                **Processed:** {analysis['progress']}/{metrics['total_tokens']}
                """
                st.session_state.ui['status'].markdown(status_text)
            
            with cols[1]:
                metrics_text = f"""
                ⚡ **Speed:** {speed:.1f}/sec  
                ⏱️ **Elapsed:** {elapsed:.1f}s  
                ✅ **Valid:** {len(analysis['results'])}  
                📈 **Price:** ${self.analyzer.get_token_price(token):.4f}
                """
                st.session_state.ui['metrics'].markdown(metrics_text)

        time.sleep(UI_REFRESH_INTERVAL)

    def process_tokens(self, tokens):
        for idx, token in enumerate(tokens):
            if not st.session_state.analysis['running']:
                break

            analysis = self.analyzer.analyze_token(token)
            if analysis and analysis['rating'] >= st.session_state.analysis['params']['min_rating']:
                st.session_state.analysis['results'].append(analysis)

            st.session_state.analysis['progress'] = idx + 1
            self.update_ui(token)
            save_checkpoint()

        self.finalize_analysis()

    def finalize_analysis(self):
        st.session_state.analysis['running'] = False
        st.session_state.results = pd.DataFrame(st.session_state.analysis['results'])
        clear_checkpoint()
        self.cleanup_ui()
        st.rerun()

    def cleanup_ui(self):
        if st.session_state.ui['container']:
            st.session_state.ui['container'].empty()
            st.session_state.ui = {'container': None, 'progress': None}

class UIManager:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.manager = AnalysisManager(analyzer)

    def render_sidebar(self):
        with st.sidebar:
            st.image("https://jup.ag/svg/jupiter-logo.svg", width=200)
            st.header("Analysis Parameters")
            
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
            if st.button("🧹 Clear Cache"):
                clear_checkpoint()
                st.session_state.clear()
                st.rerun()

    def start_analysis(self, params):
        if checkpoint := load_checkpoint():
            self.manager.start_analysis(checkpoint['params']['tokens'], params)
        else:
            tokens = self.analyzer.get_all_tokens(params['strict_mode'])
            if tokens:
                self.manager.start_analysis(tokens, params)
            else:
                st.error("No tokens found matching criteria")

    def stop_analysis(self):
        st.session_state.analysis['running'] = False
        clear_checkpoint()
        st.rerun()

    def render_main(self):
        st.title("🪙 Solana Token Analysis Suite")
        
        if st.session_state.analysis.get('running', False):
            self.render_live_dashboard()
        else:
            self.render_results()
            self.render_debug_info()

    def render_live_dashboard(self):
        with st.expander("📈 Real-time Metrics", expanded=True):
            cols = st.columns(4)
            metrics = st.session_state.analysis['metrics']
            cols[0].metric("Tokens Analyzed", st.session_state.analysis['progress'])
            cols[1].metric("Valid Tokens", len(st.session_state.analysis['results']))
            cols[2].metric("Avg Speed", f"{metrics['speed']:.1f} tkn/s")
            cols[3].metric("Elapsed Time", f"{time.time() - metrics['start_time']:.1f}s")

        with st.expander("🔍 Live Token Preview"):
            if st.session_state.analysis['results']:
                self.render_token_card(st.session_state.analysis['results'][-1])

    def render_results(self):
        if hasattr(st.session_state, 'results') and not st.session_state.results.empty:
            tab1, tab2, tab3 = st.tabs(["📋 Results Table", "📈 Market Overview", "📤 Export Data"])
            
            with tab1:
                self.render_results_table()
            
            with tab2:
                self.render_market_charts()
            
            with tab3:
                self.render_export_section()

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

    def render_export_section(self):
        st.download_button(
            "💾 Download CSV Report",
            data=st.session_state.results.to_csv(index=False),
            file_name=RESULTS_FILE,
            mime="text/csv"
        )
        
        json_data = st.session_state.results.to_json(orient='records')
        st.download_button(
            "📥 Download JSON Data",
            data=json_data,
            file_name="token_analysis.json",
            mime="application/json"
        )

    def render_token_card(self, token):
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(token.get('logoURI', 'https://via.placeholder.com/80'), width=80)
        
        with cols[1]:
            st.subheader(f"{token['symbol']}")
            st.markdown(f"**Price:** ${token['price']:.4f} | **MCap:** ${token['market_cap']:,.0f}")
            st.progress(token['rating']/100, f"Rating: {token['rating']:.1f}/100")
            
        st.markdown(f"""
        - **Liquidity Score:** {token['liquidity']:.1f}
        - **24h Volume:** ${token['volume']:,.0f}
        - **Token Age:** {token['age']:.1f} days
        - [Explorer Link]({token['explorer']})
        """)

    def render_debug_info(self):
        if st.session_state.get('show_debug', False):
            with st.expander("🔧 Debug Information"):
                debug_df = pd.DataFrame(self.analyzer.debug_info)
                st.dataframe(
                    debug_df[['symbol', 'address', 'valid', 'reasons']],
                    column_config={"reasons": "Validation Issues"},
                    height=300
                )

def save_checkpoint():
    data = {
        'params': st.session_state.analysis['params'],
        'results': st.session_state.analysis['results'],
        'progress': st.session_state.analysis['progress']
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                return json.load(f)
        except:
            pass
    return None

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

if __name__ == "__main__":
    analyzer = TokenAnalyzer()
    ui = UIManager(analyzer)
    ui.render_sidebar()
    ui.render_main()
    st.session_state.setdefault('show_debug', False)
