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
BATCH_SIZE = 5  # Tokens per batch

class TokenAnalyzer:
    def __init__(self):
        self.client = Client(SOLANA_RPC_ENDPOINT)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, 
                      status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.debug_info = []
        
    def get_all_tokens(self, strict_checks=True):
        """Get validated tokens with debug info"""
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
            return valid_tokens
            
        except Exception as e:
            st.error(f"Token fetch error: {str(e)}")
            return []

    def _valid_token(self, token, strict_checks):
        """Validate token against criteria"""
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

            required_fields = [
                ('symbol', 'Missing symbol'),
                ('name', 'Missing name'),
                ('decimals', 'Missing decimals'),
                ('logoURI', 'Missing logo')
            ]
            
            checks = [
                ('tag', lambda: "community" in token.get('tags') or "old-registry" in token.get("tags"), 'Missing community tag'),
                ('extensions', lambda: 'coingeckoId' in token.get('extensions', {}), 'Missing Coingecko ID'),
                ('strict', lambda: not strict_checks or (token.get('chainId') == 101 and token.get('address')), 'Strict check failed')
            ]

            for field, message in required_fields:
                if field not in token or not token[field]:
                    reasons.append(f'Missing {field}: {message}')
                    return False, reasons

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

    def analyze_generator(self, tokens, min_mcap):
        """Generator function for batch processing"""
        results = []
        processed = 0
        total = len(tokens)
        
        for idx, token in enumerate(tokens):
            try:
                analysis = self.deep_analyze(token)
                if analysis and analysis['market_cap'] >= min_mcap:
                    results.append(analysis)
            except Exception as e:
                st.error(f"Skipping {token.get('symbol')}: {str(e)}")
            
            processed = idx + 1
            if (idx + 1) % BATCH_SIZE == 0 or processed == total:
                yield {
                    'processed': processed,
                    'total': total,
                    'results': results,
                    'current_token': token
                }
                results = []  # Reset batch

    def deep_analyze(self, token):
        """Comprehensive token analysis"""
        try:
            return {
                'symbol': token.get('symbol', 'Unknown'),
                'address': token['address'],
                'price': float(token.get('price', 0)),
                'market_cap': self._estimate_market_cap(token),
                'liquidity_score': self._calculate_liquidity(token),
                'rating': self._calculate_rating(token),
                'explorer': f"https://solscan.io/token/{token['address']}",
                'supply': self._get_circulating_supply(token)
            }
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {str(e)}") from e

    def _estimate_market_cap(self, token):
        """Calculate market cap"""
        try:
            supply = self._get_circulating_supply(token)
            return supply * float(token.get('price', 0))
        except:
            return 0

    def _calculate_liquidity(self, token):
        """Calculate liquidity score"""
        try:
            input_mint = token['address']
            decimals = token.get('decimals', 9)
            amount = int(1000 * (10 ** decimals))
            
            quote_url = f"https://quote-api.jup.ag/v6/quote?inputMint={input_mint}&outputMint={USDC_MINT}&amount={amount}"
            response = self.session.get(quote_url, timeout=15)
            quote = response.json()
            
            price_impact = float(quote.get('priceImpactPct', 1))
            return max(0, 100 - (price_impact * 10000))
        except:
            return 0

    def _calculate_rating(self, token):
        """Calculate composite rating"""
        try:
            liquidity = self._calculate_liquidity(token)
            mcap = self._estimate_market_cap(token)
            return min(100, max(0, round((liquidity * 0.6) + (40 * (1 / (1 + (mcap / 1e6)))), 2)))
        except:
            return 0

    def _get_circulating_supply(self, token):
        """Get supply from chain"""
        try:
            mint_address = Pubkey.from_string(token['address'])
            account_info = self.client.get_account_info_json_parsed(mint_address).value
            return account_info.data.parsed['info']['supply'] / 10 ** token.get('decimals', 9)
        except:
            return 0

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

def main():
    st.set_page_config(page_title="Token Analyst Pro", layout="wide")
    st.title("🔍 Pure On-Chain Token Analysis")
    
    analyzer = TokenAnalyzer()

    # Initialize session state
    if 'analysis' not in st.session_state:
        st.session_state.analysis = {
            'running': False,
            'generator': None,
            'results': [],
            'params': None,
            'processed': 0,
            'total': 0
        }

    with st.sidebar:
        st.header("Parameters")
        min_rating = st.slider("Minimum Rating", 0, 100, 65)
        min_mcap = st.number_input("Minimum Market Cap (USD)", 1000, 10000000, 10000)
        strict_mode = st.checkbox("Strict Validation", value=True)
        show_debug = st.checkbox("Show Debug Info", value=False)

        def start_analysis():
            checkpoint = load_checkpoint()
            if checkpoint:
                st.session_state.analysis.update({
                    'running': True,
                    'processed': checkpoint['processed'],
                    'total': checkpoint['total'],
                    'results': checkpoint['results'],
                    'params': checkpoint['params'],
                    'generator': analyzer.analyze_generator(
                        checkpoint['params']['tokens'],
                        checkpoint['params']['min_mcap']
                    )
                })
            else:
                tokens = analyzer.get_all_tokens(strict_checks=strict_mode)
                if not tokens:
                    st.error("No tokens found")
                    return
                
                st.session_state.analysis = {
                    'running': True,
                    'generator': analyzer.analyze_generator(tokens, min_mcap),
                    'results': [],
                    'params': {
                        'min_mcap': min_mcap,
                        'strict_mode': strict_mode,
                        'tokens': tokens
                    },
                    'processed': 0,
                    'total': len(tokens)
                }
            save_checkpoint(st.session_state.analysis)

        def stop_analysis():
            st.session_state.analysis['running'] = False
            clear_checkpoint()
            st.rerun()

        st.button("🔍 Start Analysis", on_click=start_analysis)
        st.button("⏹ Stop Analysis", on_click=stop_analysis)

    # Analysis progress handling
    if st.session_state.analysis['running']:
        st.subheader("Analysis Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Process next batch
            batch_result = next(st.session_state.analysis['generator'])
            st.session_state.analysis['processed'] = batch_result['processed']
            st.session_state.analysis['results'].extend(batch_result['results'])
            
            # Update progress
            progress = st.session_state.analysis['processed'] / st.session_state.analysis['total']
            progress_bar.progress(progress)
            
            # Update status
            status_text.markdown(f"""
            **Progress:** {st.session_state.analysis['processed']}/{st.session_state.analysis['total']}  
            **Valid Tokens:** {len(st.session_state.analysis['results'])}  
            **Current Token:** {batch_result['current_token'].get('symbol', 'Unknown')} 
            ({batch_result['current_token']['address'][:6]}...)
            """)
            
            # Save checkpoint and rerun
            save_checkpoint({
                'processed': st.session_state.analysis['processed'],
                'total': st.session_state.analysis['total'],
                'results': st.session_state.analysis['results'],
                'params': st.session_state.analysis['params']
            })
            st.rerun()
            
        except StopIteration:
            # Analysis complete
            st.session_state.analysis['running'] = False
            st.session_state.analysis_results = pd.DataFrame(st.session_state.analysis['results'])
            clear_checkpoint()
            st.success("✅ Analysis completed!")
            st.rerun()

    # Display results
    if not st.session_state.analysis['running'] and 'analysis_results' in st.session_state:
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

    # Debug information
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
    main()
