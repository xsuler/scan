import streamlit as st
import pandas as pd
import requests
import json
import os
import time
from solders.pubkey import Pubkey
from solana.rpc.api import Client
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
SOLANA_RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"
RESULTS_FILE = "token_analysis.csv"
CHECKPOINT_FILE = "analysis_checkpoint.json"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
BATCH_SIZE = 5
UI_REFRESH_INTERVAL = 0.01  # Seconds

class TokenAnalyzer:
    def __init__(self):
        self.client = Client(SOLANA_RPC_ENDPOINT)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.debug_info = []

    # Keep existing validation and analysis methods from previous implementation
    # [Include all the TokenAnalyzer methods from previous implementation here]
    
class AnalysisManager:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.init_session_state()
        
    def init_session_state(self):
        defaults = {
            'analysis': {
                'running': False,
                'results': [],
                'params': None,
                'processed': 0,
                'total': 0
            },
            'analysis_results': pd.DataFrame(),
            'ui_elements': {
                'progress_bar': None,
                'status_text': None,
                'container': None
            }
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    def start_analysis(self, tokens, min_mcap, strict_mode):
        st.session_state.analysis = {
            'running': True,
            'results': [],
            'params': {
                'min_mcap': min_mcap,
                'strict_mode': strict_mode,
                'tokens': tokens
            },
            'processed': 0,
            'total': len(tokens)
        }
        self.create_progress_ui()
        self.process_tokens()

    def create_progress_ui(self):
        if st.session_state.ui_elements['container'] is None:
            st.session_state.ui_elements['container'] = st.empty()
            
        with st.session_state.ui_elements['container'].container():
            st.subheader("Analysis Progress")
            st.session_state.ui_elements['progress_bar'] = st.progress(0)
            st.session_state.ui_elements['status_text'] = st.empty()

    def update_progress_ui(self, current_token):
        progress = st.session_state.analysis['processed'] / st.session_state.analysis['total']
        st.session_state.ui_elements['progress_bar'].progress(progress)
        
        status_content = f"""
        **Progress:** {st.session_state.analysis['processed']}/{st.session_state.analysis['total']}
        **Valid Tokens:** {len(st.session_state.analysis['results'])}
        **Current Token:** {current_token.get('symbol', 'Unknown')} ({current_token['address'][:6]}...)
        """
        st.session_state.ui_elements['status_text'].markdown(status_content)
        time.sleep(UI_REFRESH_INTERVAL)

    def process_tokens(self):
        tokens = st.session_state.analysis['params']['tokens']
        min_mcap = st.session_state.analysis['params']['min_mcap']
        
        for idx, token in enumerate(tokens[st.session_state.analysis['processed']:]):
            if not st.session_state.analysis['running']:
                break

            try:
                analysis = self.analyzer.deep_analyze(token)
                if analysis and analysis['market_cap'] >= min_mcap:
                    st.session_state.analysis['results'].append(analysis)
            except Exception as e:
                st.error(f"Skipping {token.get('symbol')}: {str(e)}")

            st.session_state.analysis['processed'] += 1
            
            if idx % BATCH_SIZE == 0 or idx == len(tokens)-1:
                self.update_progress_ui(token)
                save_checkpoint(st.session_state.analysis)

        self.finalize_analysis()

    def finalize_analysis(self):
        st.session_state.analysis['running'] = False
        st.session_state.analysis_results = pd.DataFrame(st.session_state.analysis['results'])
        clear_checkpoint()
        self.cleanup_ui()
        st.rerun()

    def cleanup_ui(self):
        if st.session_state.ui_elements['container']:
            st.session_state.ui_elements['container'].empty()
            st.session_state.ui_elements = {
                'progress_bar': None,
                'status_text': None,
                'container': None
            }

def save_checkpoint(data):
    checkpoint = {
        'running': data['running'],
        'results': data['results'],
        'params': data['params'],
        'processed': data['processed'],
        'total': data['total']
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, default=str)

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            if validate_checkpoint(data):
                return data
            clear_checkpoint()
            return None
    except Exception as e:
        st.error(f"Checkpoint error: {str(e)}")
        clear_checkpoint()
        return None

def validate_checkpoint(data):
    required = ['params', 'processed', 'total', 'results', 'running']
    if not all(key in data for key in required):
        return False
    if 'tokens' not in data['params'] or 'min_mcap' not in data['params']:
        return False
    return True

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

def main_ui(manager):
    st.set_page_config(page_title="Token Analyst Pro", layout="wide")
    st.title("🔍 Pure On-Chain Token Analysis")

    with st.sidebar:
        st.header("Parameters")
        min_rating = st.slider("Minimum Rating", 0, 100, 65)
        min_mcap = st.number_input("Minimum Market Cap (USD)", 1000, 10000000, 10000)
        strict_mode = st.checkbox("Strict Validation", value=True)
        show_debug = st.checkbox("Show Debug Info", value=False)

        if st.button("🔍 Start Analysis"):
            handle_analysis_start(manager, min_mcap, strict_mode)

        if st.button("⏹ Stop Analysis"):
            handle_analysis_stop(manager)

    display_results(manager, min_rating)
    display_debug_info(manager.analyzer, show_debug)

def handle_analysis_start(manager, min_mcap, strict_mode):
    checkpoint = load_checkpoint()
    if checkpoint:
        try:
            st.session_state.analysis = checkpoint
            manager.create_progress_ui()
            manager.process_tokens()
        except Exception as e:
            st.error(f"Checkpoint error: {str(e)}")
            clear_checkpoint()
    else:
        tokens = manager.analyzer.get_all_tokens(strict_checks=strict_mode)
        if tokens:
            manager.start_analysis(tokens, min_mcap, strict_mode)
        else:
            st.error("No valid tokens found")

def handle_analysis_stop(manager):
    st.session_state.analysis['running'] = False
    clear_checkpoint()
    manager.cleanup_ui()
    st.rerun()

def display_results(manager, min_rating):
    if not st.session_state.analysis['running'] and not st.session_state.analysis_results.empty:
        filtered = st.session_state.analysis_results[
            st.session_state.analysis_results['rating'] >= min_rating
        ].sort_values('rating', ascending=False)
        
        st.subheader("📊 Analysis Results")
        st.dataframe(
            filtered,
            column_config={
                'symbol': 'Symbol',
                'address': 'Contract',
                'price': st.column_config.NumberColumn('Price', format="$%.6f"),
                'market_cap': st.column_config.NumberColumn('Market Cap', format="$%.2f"),
                'rating': st.column_config.ProgressColumn('Rating', min_value=0, max_value=100),
                'liquidity_score': 'Liquidity',
                'explorer': st.column_config.LinkColumn('Explorer'),
                'supply': 'Circulating Supply'
            },
            hide_index=True,
            height=600,
            use_container_width=True
        )
        
        st.download_button(
            "📥 Export Report",
            data=filtered.to_csv(index=False),
            file_name=RESULTS_FILE,
            mime="text/csv"
        )

def display_debug_info(analyzer, show_debug):
    if show_debug and hasattr(analyzer, 'debug_info'):
        st.subheader("🔧 Validation Debug Info")
        debug_df = pd.DataFrame(analyzer.debug_info)
        debug_df = debug_df[['symbol', 'address', 'checks_passed', 'reasons']]
        debug_df['reasons'] = debug_df['reasons'].apply(lambda x: '\n'.join(x))
        
        st.dataframe(
            debug_df,
            column_config={
                'symbol': 'Symbol',
                'address': 'Contract',
                'checks_passed': 'Checks Passed',
                'reasons': 'Validation Reasons'
            },
            height=400,
            use_container_width=True
        )

if __name__ == "__main__":
    analyzer = TokenAnalyzer()
    manager = AnalysisManager(analyzer)
    main_ui(manager)
