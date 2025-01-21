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
REQUEST_INTERVAL = 0.5  # 500ms between requests

class TokenAnalyzer:
    def __init__(self):
        self.client = Client(SOLANA_RPC_ENDPOINT)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, 
                      status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
    def get_recent_tokens(self, days=3, strict_checks=True):
        """Get tokens added in specified days with optional validation"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
            tokens = response.json()
            
            recent_tokens = []
            for t in tokens:
                try:
                    added_date = datetime.fromisoformat(t['timeAdded'].rstrip('Z'))
                    if added_date > cutoff_date and self._valid_token(t, strict_checks):
                        recent_tokens.append(t)
                except:
                    continue
            return recent_tokens
        except Exception as e:
            st.error(f"Token fetch error: {str(e)}")
            return []

    def _valid_token(self, token, strict_checks):
        """Configurable token validation"""
        try:
            # Basic checks that apply to all tokens
            if not Pubkey.from_string(token['address']):
                return False
            
            # Strict mode checks
            if strict_checks:
                return (
                    token['symbol'] != 'SOL' and
                    float(token.get('price', 0)) > 0 and
                    token.get('extensions', {}).get('coingeckoId') and
                    token.get('extensions', {}).get('website')
                )
            return True  # Only check valid address in non-strict mode
        except:
            return False

    def deep_analyze(self, token):
        """Comprehensive token analysis"""
        time.sleep(REQUEST_INTERVAL)
        mint = token['address']
        
        try:
            # Get detailed price data
            price_data = self.session.get(
                f"{JUPITER_PRICE_API}?ids={mint}&showExtraInfo=true",
                timeout=10
            ).json().get('data', {}).get(mint, {})
            
            if not price_data:
                return None

            # Get supply information
            supply = self._get_token_supply(mint)
            market_cap = float(price_data.get('price', 0)) * supply
            
            # Extract market depth metrics
            extra_info = price_data.get('extraInfo', {})
            depth = extra_info.get('depth', {})
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(depth)
            
            # Calculate volatility index
            volatility = self._calculate_volatility(extra_info.get('lastSwappedPrice', {}))
            
            # Confidence scoring
            confidence_map = {'high': 9, 'medium': 7, 'low': 5}
            confidence = confidence_map.get(extra_info.get('confidenceLevel', 'low'), 5)
            
            # Depth quality analysis
            depth_quality = self._assess_depth_quality(depth)
            
            # Final rating calculation
            rating = self._calculate_rating(liquidity_score, volatility, depth_quality, confidence)
            
            return {
                'symbol': token['symbol'].upper(),
                'address': mint,
                'price': float(price_data.get('price', 0)),
                'market_cap': market_cap,
                'liquidity_score': liquidity_score,
                'volatility_index': volatility,
                'depth_quality': depth_quality,
                'confidence': confidence,
                'rating': rating,
                'added_date': token['timeAdded'],
                'explorer': f"https://solscan.io/token/{mint}",
                'validated': self._valid_token(token, strict_checks=True)
            }
        except Exception as e:
            st.error(f"Analysis failed for {token['symbol']}: {str(e)}")
            return None

    def _get_token_supply(self, mint):
        """Get verified token supply"""
        try:
            supply_info = self.client.get_token_supply(Pubkey.from_string(mint))
            return int(supply_info.value.amount) / 10 ** supply_info.value.decimals
        except:
            return 0

    def _calculate_liquidity_score(self, depth_data):
        """Calculate liquidity score (0-100) based on market depth"""
        try:
            buy_impact = depth_data.get('buyPriceImpactRatio', {}).get('depth', {})
            sell_impact = depth_data.get('sellPriceImpactRatio', {}).get('depth', {})
            
            impacts = [
                buy_impact.get('10', 0),
                buy_impact.get('100', 0),
                buy_impact.get('1000', 0),
                sell_impact.get('10', 0),
                sell_impact.get('100', 0),
                sell_impact.get('1000', 0)
            ]
            avg_impact = sum(impacts) / len(impacts)
            return max(0, 100 - (avg_impact * 1000))
        except:
            return 0

    def _calculate_volatility(self, last_swap):
        """Calculate volatility percentage"""
        try:
            buy_price = float(last_swap.get('lastJupiterBuyPrice', 0))
            sell_price = float(last_swap.get('lastJupiterSellPrice', 0))
            current_price = (buy_price + sell_price) / 2
            return abs(buy_price - sell_price) / current_price * 100
        except:
            return 0

    def _assess_depth_quality(self, depth_data):
        """Quality assessment for large trades"""
        try:
            buy_1k = depth_data.get('buyPriceImpactRatio', {}).get('depth', {}).get('1000', 0)
            sell_1k = depth_data.get('sellPriceImpactRatio', {}).get('depth', {}).get('1000', 0)
            avg_impact = (buy_1k + sell_1k) / 2
            
            if avg_impact < 0.05: return 'Excellent'
            if avg_impact < 0.15: return 'Good'
            if avg_impact < 0.3: return 'Fair'
            return 'Poor'
        except:
            return 'Unknown'

    def _calculate_rating(self, liquidity, volatility, depth_quality, confidence):
        """Calculate composite rating (0-100)"""
        quality_scores = {'Excellent': 95, 'Good': 80, 'Fair': 65, 'Poor': 40}
        return (
            (liquidity * 0.4) +
            ((100 - volatility) * 0.3) +
            (quality_scores.get(depth_quality, 50) * 0.2) +
            (confidence * 10 * 0.1)
        )

def main():
    st.set_page_config(page_title="Smart Token Analyzer Pro", layout="wide")
    st.title("🔭 Advanced Token Discovery & Analysis")
    
    analyzer = TokenAnalyzer()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = pd.DataFrame()

    with st.sidebar:
        st.header("Analysis Parameters")
        
        # Configurable parameters
        analysis_days = st.number_input("Analysis Days", 1, 30, 3)
        min_rating = st.slider("Minimum Rating", 0, 100, 75)
        min_mcap = st.number_input("Minimum Market Cap (USD)", 0, 1000000000, 100000)
        strict_checks = st.checkbox("Strict Validation", value=True,
                                  help="Enable strict token validation checks")
        
        run_analysis = st.button("🚀 Find Promising Tokens")
        
        if st.session_state.analysis_results.empty:
            st.info("No analysis data yet")
        else:
            st.download_button(
                "📥 Download Full Report",
                data=st.session_state.analysis_results.to_csv(index=False),
                file_name=RESULTS_FILE,
                mime="text/csv"
            )

    if run_analysis:
        with st.spinner(f"Finding tokens added in last {analysis_days} days..."):
            recent_tokens = analyzer.get_recent_tokens(days=analysis_days, strict_checks=strict_checks)
            if not recent_tokens:
                st.error("No new tokens found meeting criteria")
                return
                
            st.info(f"Found {len(recent_tokens)} new tokens, starting deep analysis...")
            
            results = []
            progress_bar = st.progress(0)
            total_tokens = len(recent_tokens)
            
            for idx, token in enumerate(recent_tokens):
                analysis = analyzer.deep_analyze(token)
                if analysis and analysis['market_cap'] >= min_mcap:
                    results.append(analysis)
                progress_bar.progress((idx + 1) / total_tokens)
            
            st.session_state.analysis_results = pd.DataFrame(results)
            progress_bar.empty()
            st.success(f"Analysis complete! Found {len(results)} promising tokens")

    if not st.session_state.analysis_results.empty:
        filtered = st.session_state.analysis_results[
            (st.session_state.analysis_results['rating'] >= min_rating)
        ].sort_values('rating', ascending=False)
        
        st.subheader("Top Potential Tokens")
        st.dataframe(
            filtered,
            column_config={
                'symbol': 'Symbol',
                'address': 'Contract Address',
                'price': st.column_config.NumberColumn('Price', format="$%.4f"),
                'market_cap': st.column_config.NumberColumn('Market Cap', format="$%.2f"),
                'rating': st.column_config.ProgressColumn('Rating', format="%.1f", min_value=0, max_value=100),
                'liquidity_score': 'Liquidity',
                'volatility_index': 'Volatility %',
                'depth_quality': 'Depth Quality',
                'confidence': 'Confidence',
                'validated': 'Validated',
                'explorer': st.column_config.LinkColumn('Explorer')
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )

if __name__ == "__main__":
    main()
