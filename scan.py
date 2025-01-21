import streamlit as st
import pandas as pd
import requests
import time
from solders.pubkey import Pubkey
from solana.rpc.api import Client
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
SOLANA_RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"
RESULTS_FILE = "token_analysis.csv"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

class TokenAnalyzer:
    def __init__(self):
        self.client = Client(SOLANA_RPC_ENDPOINT)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, 
                      status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.debug_info = []
        
    def get_all_tokens(self, strict_checks=True):
        """Get all tokens with validation"""
        self.debug_info = []
        try:
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
            tokens = response.json()
            
            valid_tokens = []
            for t in tokens:
                debug_entry = {
                    'symbol': t.get('symbol', 'Unknown'),
                    'address': t.get('address', ''),
                    'valid': False,
                    'reasons': [],
                    'checks_passed': 0
                }
                
                try:
                    validation_result, reasons = self._valid_token(t, strict_checks)
                    debug_entry['reasons'] = reasons
                    debug_entry['checks_passed'] = len([r for r in reasons if not r.startswith('Failed')])
                    
                    if validation_result:
                        debug_entry['valid'] = True
                        valid_tokens.append(t)
                        
                except Exception as e:
                    debug_entry['reasons'].append(f'Validation error: {str(e)}')
                    
                self.debug_info.append(debug_entry)
            st.error(len(valid_tokens)) 
            return valid_tokens
            
        except Exception as e:
            st.error(f"Token fetch error: {str(e)}")
            return []

    def _valid_token(self, token, strict_checks):
        """Token validation with birdeye-trending requirement"""
        reasons = [] 
        try:
            address = token.get('address', '')
            if not address:
                reasons.append('No address')
                return False, reasons
                
            try:
                Pubkey.from_string(address)
            except:
                reasons.append('Invalid SPL address')
                return False, reasons

            st.info(token.get("tags"))
            checks = [
                    ('tag', lambda: "verified" in token.get('tags'), 'Missing symbol'),
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

    def deep_analyze(self, token):
        """Perform comprehensive token analysis"""
        try:
            analysis = {
                'symbol': token.get('symbol', 'Unknown'),
                'address': token['address'],
                'price': float(token.get('price', 0)),
                'market_cap': self._estimate_market_cap(token),
                'liquidity_score': self._calculate_liquidity(token),
                'rating': self._calculate_rating(token),
                'explorer': f"https://solscan.io/token/{token['address']}",
                'supply': self._get_circulating_supply(token)
            }
            return analysis
        except Exception as e:
            st.error(f"Analysis failed for {token.get('symbol')}: {str(e)}")
            return None

    def _estimate_market_cap(self, token):
        """Market cap estimation using circulating supply"""
        try:
            supply = self._get_circulating_supply(token)
            return supply * float(token.get('price', 0))
        except:
            return 0

    def _calculate_liquidity(self, token):
        """Liquidity scoring using Jupiter swap simulation"""
        try:
            input_mint = token['address']
            decimals = token.get('decimals', 9)
            amount = int(1000 * (10 ** decimals))  # Simulate 1000 token swap
            
            quote_url = f"https://quote-api.jup.ag/v6/quote?inputMint={input_mint}&outputMint={USDC_MINT}&amount={amount}"
            response = self.session.get(quote_url, timeout=15)
            quote = response.json()
            
            price_impact = float(quote.get('priceImpactPct', 1))
            return max(0, 100 - (price_impact * 10000))  # Convert impact to liquidity score
        except:
            return 0

    def _calculate_rating(self, token):
        """Composite rating system"""
        try:
            score = 0
            liquidity = self._calculate_liquidity(token)
            mcap = self._estimate_market_cap(token)
            
            # Liquidity component (0-60)
            score += liquidity * 0.6
            
            # Market cap component (0-40)
            score += 40 * (1 / (1 + (mcap / 1e6)))  # Inverse scaling
            
            return min(100, max(0, round(score, 2)))
        except:
            return 0

    def _get_circulating_supply(self, token):
        """Get circulating supply from chain"""
        try:
            mint_address = Pubkey.from_string(token['address'])
            account_info = self.client.get_account_info_json_parsed(mint_address).value
            return account_info.data.parsed['info']['supply'] / 10 ** token.get('decimals', 9)
        except:
            return 0

def main():
    st.set_page_config(page_title="Token Analyst Pro", layout="wide")
    st.title("🔍 Pure On-Chain Token Analysis")
    
    analyzer = TokenAnalyzer()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = pd.DataFrame()

    with st.sidebar:
        st.header("Parameters")
        min_rating = st.slider("Minimum Rating", 0, 100, 65)
        min_mcap = st.number_input("Minimum Market Cap (USD)", 1000, 10000000, 10000)
        strict_mode = st.checkbox("Strict Validation", value=True)
        show_debug = st.checkbox("Show Debug Info", value=False)
        
        if st.button("🔍 Analyze Tokens"):
            with st.spinner("Scanning blockchain..."):
                tokens = analyzer.get_all_tokens(strict_checks=strict_mode)
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
                st.success(f"Analyzed {len(results)} qualifying tokens")
                progress_bar.empty()

        if not st.session_state.analysis_results.empty:
            st.download_button(
                "📥 Export Report",
                data=st.session_state.analysis_results.to_csv(index=False),
                file_name=RESULTS_FILE,
                mime="text/csv"
            )

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

    if not st.session_state.analysis_results.empty:
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

if __name__ == "__main__":
    main()
