import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from solders.pubkey import Pubkey
from solana.rpc.api import Client
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
SOLANA_RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"
RESULTS_FILE = "token_analysis.csv"
REQUEST_INTERVAL = 0.5

class TokenAnalyzer:
    def __init__(self):
        self.client = Client(SOLANA_RPC_ENDPOINT)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, 
                      status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.debug_info = []
        
    def get_recent_tokens(self, days=3, strict_checks=True):
        """Improved token filtering with detailed debug info"""
        self.debug_info = []
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
            tokens = response.json()
            st.info(tokens[0])
            
            recent_tokens = []
            for t in tokens:
                debug_entry = {
                    'symbol': t.get('symbol', 'Unknown'),
                    'address': t.get('address', ''),
                    'raw_time': t.get('timeAdded', ''),
                    'valid': False,
                    'reasons': [],
                    'parsed_time': None,
                    'time_valid': False,
                    'checks_passed': 0
                }
                
                try:
                    # Parse timestamp with timezone handling
                    time_str = t.get('timeAdded', '')
                    if 'Z' in time_str:
                        time_str = time_str.replace('Z', '+00:00')
                    added_date = datetime.fromisoformat(time_str)
                    debug_entry['parsed_time'] = added_date.isoformat()
                    
                    # Check time validity
                    time_valid = added_date > cutoff_date
                    debug_entry['time_valid'] = time_valid
                    if not time_valid:
                        debug_entry['reasons'].append('Too old')
                        self.debug_info.append(debug_entry)
                        continue

                    # Validate token
                    validation_result, reasons = self._valid_token(t, strict_checks)
                    debug_entry['reasons'] = reasons
                    debug_entry['checks_passed'] = len([r for r in reasons if not r.startswith('Failed')])
                    
                    if validation_result:
                        debug_entry['valid'] = True
                        recent_tokens.append(t)
                    else:
                        debug_entry['valid'] = False
                        
                except Exception as e:
                    debug_entry['reasons'].append(f'Parse error: {str(e)}')
                    
                self.debug_info.append(debug_entry)
                
            return recent_tokens
        except Exception as e:
            st.error(f"Token fetch error: {str(e)}")
            return []

    def _valid_token(self, token, strict_checks):
        """Validation with detailed rejection reasons"""
        reasons = []
        try:
            # Basic checks
            address = token.get('address', '')
            if not address:
                reasons.append('No address')
                return False, reasons
                
            try:
                Pubkey.from_string(address)
            except:
                reasons.append('Invalid SPL address')
                return False, reasons

            # Strict mode checks
            if strict_checks:
                checks = [
                    ('symbol', lambda: token.get('symbol') not in ['SOL', 'USDC', 'USDT'], 'Excluded symbol'),
                    ('price', lambda: float(token.get('price', 0)) > 0.000001, 'Price too low'),
                    ('website', lambda: bool(token.get('extensions', {}).get('website')), 'Missing website'),
                    ('twitter', lambda: bool(token.get('extensions', {}).get('twitter')), 'Missing twitter')
                ]
            else:
                checks = [
                    ('symbol', lambda: bool(token.get('symbol')), 'Missing symbol'),
                    ('name', lambda: bool(token.get('name')), 'Missing name'),
                    ('price', lambda: float(token.get('price', 0)) > 0, 'Invalid price')
                ]

            passed = True
            for check_name, check_func, msg in checks:
                if not check_func():
                    reasons.append(f'Failed {check_name}: {msg}')
                    passed = False
                else:
                    reasons.append(f'Passed {check_name}')

            return passed, reasons
            
        except Exception as e:
            reasons.append(f'Validation error: {str(e)}')
            return False, reasons

    # [Keep rest of the analysis methods unchanged from previous version]

def main():
    st.set_page_config(page_title="Token Analyst Pro+", layout="wide")
    st.title("🔍 Token Discovery with Debug Info")
    
    analyzer = TokenAnalyzer()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = pd.DataFrame()

    with st.sidebar:
        st.header("Parameters")
        days = st.slider("Lookback Days", 1, 7, 3)
        min_rating = st.slider("Minimum Rating", 0, 100, 65)
        min_mcap = st.number_input("Minimum Market Cap (USD)", 1000, 10000000, 10000)
        strict_mode = st.checkbox("Strict Validation", value=False)
        show_debug = st.checkbox("Show Debug Info", value=False)
        
        if st.button("🔍 Find Promising Tokens"):
            with st.spinner("Scanning new listings..."):
                tokens = analyzer.get_recent_tokens(days=days, strict_checks=strict_mode)
                if not tokens:
                    st.error("No tokens found. Try adjusting filters.")
                    return
                    
                results = []
                progress_bar = st.progress(0)
                
                for idx, token in enumerate(tokens):
                    analysis = analyzer.deep_analyze(token)
                    if analysis and analysis['market_cap'] >= min_mcap:
                        results.append(analysis)
                    progress_bar.progress((idx + 1) / len(tokens))
                
                st.session_state.analysis_results = pd.DataFrame(results)
                st.success(f"Found {len(results)} qualifying tokens")
                progress_bar.empty()

        if not st.session_state.analysis_results.empty:
            st.download_button(
                "📥 Export Report",
                data=st.session_state.analysis_results.to_csv(index=False),
                file_name=RESULTS_FILE,
                mime="text/csv"
            )

    if show_debug and hasattr(analyzer, 'debug_info'):
        st.subheader("🚨 Debug Information - Token Listing")
        debug_df = pd.DataFrame(analyzer.debug_info)
        
        # Filter and format debug info
        debug_df = debug_df[['symbol', 'address', 'raw_time', 'parsed_time', 
                           'time_valid', 'checks_passed', 'reasons']]
        debug_df['reasons'] = debug_df['reasons'].apply(lambda x: '\n'.join(x))
        
        st.dataframe(
            debug_df,
            column_config={
                'symbol': 'Symbol',
                'address': 'Contract',
                'raw_time': 'Raw TimeAdded',
                'parsed_time': 'Parsed Time',
                'time_valid': 'Time Valid',
                'checks_passed': 'Checks Passed',
                'reasons': 'Validation Reasons'
            },
            hide_index=True,
            height=600,
            use_container_width=True
        )

    if not st.session_state.analysis_results.empty:
        filtered = st.session_state.analysis_results[
            st.session_state.analysis_results['rating'] >= min_rating
        ].sort_values('rating', ascending=False)
        
        st.subheader("Top Candidate Tokens")
        st.dataframe(
            filtered,
            column_config={
                'symbol': 'Symbol',
                'address': 'Contract',
                'price': st.column_config.NumberColumn('Price', format="$%.6f"),
                'market_cap': st.column_config.NumberColumn('Market Cap', format="$%.2f"),
                'rating': st.column_config.ProgressColumn('Rating', min_value=0, max_value=100),
                'liquidity_score': 'Liquidity',
                'volatility': 'Volatility',
                'depth_quality': 'Depth',
                'confidence': 'Confidence',
                'explorer': st.column_config.LinkColumn('Explorer'),
                'supply': 'Circulating Supply'
            },
            hide_index=True,
            height=600,
            use_container_width=True
        )

if __name__ == "__main__":
    main()
