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
        """Improved token filtering with better validation"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
            tokens = response.json()
            
            recent_tokens = []
            for t in tokens:
                try:
                    # Handle timestamp with proper timezone
                    time_str = t.get('timeAdded', '').replace('Z', '+00:00')
                    added_date = datetime.fromisoformat(time_str)
                    
                    if added_date < cutoff_date:
                        continue
                        
                    if not self._valid_token(t, strict_checks):
                        continue
                        
                    recent_tokens.append(t)
                except Exception as e:
                    continue
            return recent_tokens
        except Exception as e:
            st.error(f"Token fetch error: {str(e)}")
            return []

    def _valid_token(self, token, strict_checks):
        """More practical validation criteria"""
        try:
            # Basic checks for all tokens
            address = token.get('address', '')
            Pubkey.from_string(address)
            
            if strict_checks:
                # Reasonable strict criteria
                return (
                    token.get('symbol') not in ['SOL', 'USDC', 'USDT'] and
                    float(token.get('price', 0)) > 0.000001 and
                    token.get('extensions', {}).get('website') and
                    token.get('extensions', {}).get('twitter')
                )
            else:
                # Loose criteria
                return (
                    token.get('symbol') and
                    token.get('name') and
                    float(token.get('price', 0)) > 0
                )
        except:
            return False

    def deep_analyze(self, token):
        """Reliable analysis with fallback values"""
        time.sleep(REQUEST_INTERVAL)
        mint = token['address']
        
        try:
            # Get price data with error handling
            price_data = self.session.get(
                f"{JUPITER_PRICE_API}?ids={mint}&showExtraInfo=true",
                timeout=10
            ).json().get('data', {}).get(mint, {})
            
            if not price_data:
                return None

            # Get supply with error protection
            supply = 0
            try:
                supply_info = self.client.get_token_supply(Pubkey.from_string(mint))
                if supply_info.value:
                    supply = int(supply_info.value.amount) / 10 ** supply_info.value.decimals
            except:
                pass
                
            if supply <= 0:
                return None

            # Extract metrics with fallbacks
            current_price = float(price_data.get('price', 0))
            market_cap = current_price * supply
            
            extra_info = price_data.get('extraInfo', {})
            last_swap = extra_info.get('lastSwappedPrice', {})
            depth = extra_info.get('depth', {})
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(depth)
            
            # Calculate volatility
            try:
                buy_price = float(last_swap.get('lastJupiterBuyPrice', current_price))
                sell_price = float(last_swap.get('lastJupiterSellPrice', current_price))
                volatility = abs(buy_price - sell_price) / current_price * 100
            except:
                volatility = 0

            # Confidence scoring with fallback
            confidence_map = {'high': 90, 'medium': 70, 'low': 50}
            confidence = confidence_map.get(
                extra_info.get('confidenceLevel', 'low').lower(), 50
            )
            
            # Depth quality analysis
            depth_quality = self._assess_depth_quality(depth)
            
            # Final rating calculation
            rating = self._calculate_rating(liquidity_score, volatility, depth_quality, confidence)
            
            return {
                'symbol': token['symbol'].upper(),
                'address': mint,
                'price': current_price,
                'market_cap': market_cap,
                'liquidity_score': liquidity_score,
                'volatility': f"{volatility:.2f}%",
                'depth_quality': depth_quality,
                'confidence': confidence,
                'rating': rating,
                'added_date': token['timeAdded'],
                'explorer': f"https://solscan.io/token/{mint}",
                'supply': supply
            }
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return None

    def _calculate_liquidity_score(self, depth_data):
        """Safe liquidity score calculation"""
        try:
            buy_impact = depth_data.get('buyPriceImpactRatio', {}).get('depth', {})
            sell_impact = depth_data.get('sellPriceImpactRatio', {}).get('depth', {})
            
            impacts = [
                buy_impact.get('10', 0.1),
                buy_impact.get('100', 0.15),
                buy_impact.get('1000', 0.2),
                sell_impact.get('10', 0.1),
                sell_impact.get('100', 0.15),
                sell_impact.get('1000', 0.2)
            ]
            avg_impact = sum(impacts) / len(impacts)
            return max(0, 100 - (avg_impact * 1000))
        except:
            return 50  # Default mid-range score

    def _assess_depth_quality(self, depth_data):
        """Depth analysis with fallbacks"""
        try:
            buy_1k = depth_data.get('buyPriceImpactRatio', {}).get('depth', {}).get('1000', 0.2)
            sell_1k = depth_data.get('sellPriceImpactRatio', {}).get('depth', {}).get('1000', 0.2)
            avg_impact = (buy_1k + sell_1k) / 2
            
            if avg_impact < 0.1: return 'Excellent'
            if avg_impact < 0.25: return 'Good'
            if avg_impact < 0.5: return 'Fair'
            return 'Poor'
        except:
            return 'Unknown'

    def _calculate_rating(self, liquidity, volatility, depth_quality, confidence):
        """Robust rating calculation"""
        quality_scores = {'Excellent': 90, 'Good': 75, 'Fair': 60, 'Poor': 40, 'Unknown': 50}
        try:
            volatility_score = max(0, 100 - float(volatility.replace('%', '')))
        except:
            volatility_score = 80
            
        return (
            (liquidity * 0.4) +
            (volatility_score * 0.3) +
            (quality_scores.get(depth_quality, 50) * 0.2) +
            (confidence * 0.1)
        )

def main():
    st.set_page_config(page_title="Token Analyst Pro", layout="wide")
    st.title("🔍 Professional Token Discovery & Analysis")
    
    analyzer = TokenAnalyzer()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = pd.DataFrame()

    with st.sidebar:
        st.header("Parameters")
        days = st.slider("Lookback Days", 1, 7, 3)
        min_rating = st.slider("Minimum Rating", 0, 100, 65)
        min_mcap = st.number_input("Minimum Market Cap (USD)", 1000, 10000000, 10000)
        strict_mode = st.checkbox("Strict Validation", value=False)
        
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
