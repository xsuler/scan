import streamlit as st
import pandas as pd
import requests
import numpy as np
import time
from datetime import datetime
from solders.pubkey import Pubkey
from solana.rpc.api import Client
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
SOLANA_RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"
RESULTS_FILE = "token_analysis.csv"
REQUEST_INTERVAL = 0.5  # Seconds between requests

# Initialize Solana client
solana_client = Client(SOLANA_RPC_ENDPOINT)

# Configure requests session
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

class TokenAnalyzer:
    """Robust token analysis system with error protection"""
    
    RATING_WEIGHTS = {
        'liquidity': 0.4,
        'volatility': 0.3,
        'depth_quality': 0.2,
        'confidence': 0.1
    }
    
    def __init__(self):
        self.token_list = []
        self.analysis_results = pd.DataFrame()
        
    def fetch_token_list(self):
        """Get all tokens from Jupiter with validation"""
        try:
            response = session.get(JUPITER_TOKEN_LIST, timeout=15)
            response.raise_for_status()
            tokens = response.json()
            self.token_list = [
                t for t in tokens 
                if self._valid_mint(t.get('address')) 
                and t.get('symbol') != 'SOL'
            ]
            return True
        except Exception as e:
            st.error(f"Token list error: {str(e)}")
            return False
            
    def _valid_mint(self, mint_address):
        """Validate SPL token address"""
        try:
            Pubkey.from_string(mint_address)
            return mint_address != "So11111111111111111111111111111111111111112"
        except:
            return False

    def _safe_float(self, value, default=0.0):
        """Type-safe float conversion"""
        try:
            return float(value) if value not in [None, 'null', ''] else default
        except:
            return default

    def analyze_tokens(self, progress_callback=None):
        """Analyze all tokens with rate limiting"""
        results = []
        
        for idx, token in enumerate(self.token_list):
            try:
                mint = token['address']
                # Rate limit protection
                if idx > 0:
                    time.sleep(REQUEST_INTERVAL)
                    
                # Fetch price data
                price_data = session.get(
                    f"{JUPITER_PRICE_API}?ids={mint}&showExtraInfo=true",
                    timeout=10
                ).json().get('data', {}).get(mint, {})
                
                if not price_data:
                    continue
                    
                # Get supply info
                supply = 0
                try:
                    supply_info = solana_client.get_token_supply(Pubkey.from_string(mint))
                    if supply_info.value:
                        supply = int(supply_info.value.amount) / 10 ** supply_info.value.decimals
                except:
                    pass
                
                # Extract metrics with null safety
                extra_info = price_data.get('extraInfo', {})
                last_swap = extra_info.get('lastSwappedPrice', {})
                
                buy_price = self._safe_float(last_swap.get('lastJupiterBuyPrice'))
                sell_price = self._safe_float(last_swap.get('lastJupiterSellPrice'))
                current_price = self._safe_float(price_data.get('price'))
                market_cap = current_price * supply
                
                # Calculate volatility
                volatility = abs(buy_price - sell_price) / current_price * 100 if current_price else 0
                
                # Depth analysis
                depth = extra_info.get('depth', {})
                buy_depth = depth.get('buyPriceImpactRatio', {}).get('depth', {})
                sell_depth = depth.get('sellPriceImpactRatio', {}).get('depth', {})
                
                # Liquidity score calculation
                depth_metrics = [
                    self._safe_float(buy_depth.get('10')),
                    self._safe_float(buy_depth.get('100')),
                    self._safe_float(buy_depth.get('1000')),
                    self._safe_float(sell_depth.get('10')),
                    self._safe_float(sell_depth.get('100')),
                    self._safe_float(sell_depth.get('1000'))
                ]
                liquidity_score = max(0, 100 - (np.mean(depth_metrics) * 1000))
                
                # Confidence scoring
                confidence_map = {'high': 90, 'medium': 70, 'low': 50}
                confidence = confidence_map.get(
                    extra_info.get('confidenceLevel', 'unknown').lower(), 30
                )
                
                # Depth quality rating
                impact_values = [
                    self._safe_float(buy_depth.get('1000')),
                    self._safe_float(sell_depth.get('1000'))
                ]
                depth_quality = (
                    90 if all(x < 0.1 for x in impact_values) else
                    70 if all(x < 0.3 for x in impact_values) else 40
                )
                
                # Calculate overall rating
                overall_rating = (
                    liquidity_score * self.RATING_WEIGHTS['liquidity'] +
                    (100 - volatility) * self.RATING_WEIGHTS['volatility'] +
                    depth_quality * self.RATING_WEIGHTS['depth_quality'] +
                    confidence * self.RATING_WEIGHTS['confidence']
                )
                
                results.append({
                    'symbol': token.get('symbol', 'Unknown').upper(),
                    'mint': mint,
                    'price': current_price,
                    'market_cap': market_cap,
                    'liquidity_score': liquidity_score,
                    'volatility': volatility,
                    'depth_quality': depth_quality,
                    'confidence': confidence,
                    'overall_rating': overall_rating,
                    'timestamp': datetime.now().isoformat()
                })
                
                if progress_callback:
                    progress_callback((idx + 1) / len(self.token_list))
                    
            except Exception as e:
                st.error(f"Error analyzing {token.get('symbol', 'unknown')}: {str(e)}")
                continue
                
        self.analysis_results = pd.DataFrame(results)
        return self.analysis_results

def main():
    st.set_page_config(page_title="Professional Token Analyzer", layout="wide")
    st.title("📊 Institutional Token Analysis Platform")
    
    analyzer = TokenAnalyzer()
    
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = pd.DataFrame()
    
    with st.sidebar:
        st.header("Analysis Controls")
        min_rating = st.slider("Minimum Rating Score", 0, 100, 70)
        run_analysis = st.button("🚀 Run Full Market Analysis")
        
        if st.session_state.analysis_data.empty:
            st.warning("No analysis data available")
        else:
            st.download_button(
                "💾 Export Full Report",
                data=st.session_state.analysis_data.to_csv(index=False),
                file_name=RESULTS_FILE,
                mime="text/csv"
            )

    if run_analysis:
        with st.spinner("Loading token list..."):
            if not analyzer.fetch_token_list():
                st.error("Failed to load token list")
                return
                
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Analyzing market: {progress*100:.1f}% complete")
        
        with st.spinner("Analyzing entire market..."):
            start_time = time.time()
            results = analyzer.analyze_tokens(progress_callback=update_progress)
            st.session_state.analysis_data = results
            duration = time.time() - start_time
            
        st.success(f"Analyzed {len(results)} tokens in {duration:.2f} seconds")
        progress_bar.empty()
        status_text.empty()
    
    if not st.session_state.analysis_data.empty:
        filtered = st.session_state.analysis_data[
            st.session_state.analysis_data['overall_rating'] >= min_rating
        ].sort_values('overall_rating', ascending=False)
        
        st.subheader(f"Top Rated Tokens (Score ≥ {min_rating})")
        
        # Display metrics
        cols = st.columns(4)
        cols[0].metric("Total Analyzed", len(st.session_state.analysis_data))
        cols[1].metric("Average Rating", 
                      f"{st.session_state.analysis_data['overall_rating'].mean():.1f}/100")
        cols[2].metric("High Quality Tokens", len(filtered))
        cols[3].metric("Market Coverage", 
                      f"{len(filtered)/len(st.session_state.analysis_data):.1%}")
        
        # Display results
        st.dataframe(
            filtered,
            column_config={
                'symbol': 'Symbol',
                'mint': 'Contract Address',
                'price': st.column_config.NumberColumn(
                    'Price', format="$%.4f"
                ),
                'market_cap': st.column_config.NumberColumn(
                    'Market Cap', format="$%.2f"
                ),
                'overall_rating': st.column_config.ProgressColumn(
                    'Rating', format="%.1f", min_value=0, max_value=100
                ),
                'liquidity_score': 'Liquidity',
                'volatility': 'Volatility %',
                'depth_quality': 'Depth Quality',
                'confidence': 'Confidence'
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )

if __name__ == "__main__":
    main()
