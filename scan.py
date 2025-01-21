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
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
SOLANA_RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"
HISTORY_FILE = "token_ratings.csv"

# Initialize Solana client
solana_client = Client(SOLANA_RPC_ENDPOINT)

# Configure requests session
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

class TokenAnalyzer:
    """Professional token rating system with filtering"""
    
    RATING_WEIGHTS = {
        'liquidity': 0.4,
        'volatility': 0.3,
        'depth_quality': 0.2,
        'confidence': 0.1
    }
    
    def __init__(self, mint_address):
        self.mint_address = mint_address
        self.raw_data = None
        self.metrics = {
            'price': 0.0,
            'market_cap': 0.0,
            'liquidity_score': 0.0,
            'volatility_index': 0.0,
            'depth_quality': 0.0,
            'confidence_score': 0.0,
            'overall_rating': 0.0,
            'last_updated': None
        }
        
    def fetch_data(self):
        """Fetch market data from Jupiter API"""
        try:
            response = session.get(
                f"{JUPITER_PRICE_API}?ids={self.mint_address}&showExtraInfo=true",
                timeout=10
            )
            if response.status_code == 200:
                self.raw_data = response.json().get('data', {}).get(self.mint_address)
                self._calculate_metrics()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
    
    def _calculate_metrics(self):
        """Calculate professional rating metrics"""
        if not self.raw_data:
            return
            
        try:
            # Core metrics
            extra_info = self.raw_data.get('extraInfo', {})
            
            # Liquidity score calculation
            buy_depth = extra_info.get('depth', {}).get('buyPriceImpactRatio', {}).get('depth', {})
            sell_depth = extra_info.get('depth', {}).get('sellPriceImpactRatio', {}).get('depth', {})
            depth_impact = [
                buy_depth.get('10', 0), buy_depth.get('100', 0), buy_depth.get('1000', 0),
                sell_depth.get('10', 0), sell_depth.get('100', 0), sell_depth.get('1000', 0)
            ]
            avg_impact = np.mean([x for x in depth_impact if x is not None])
            self.metrics['liquidity_score'] = max(0, 100 - (avg_impact * 1000))
            
            # Volatility calculation
            last_swap = extra_info.get('lastSwappedPrice', {})
            buy_price = float(last_swap.get('lastJupiterBuyPrice', 0))
            sell_price = float(last_swap.get('lastJupiterSellPrice', 0))
            self.metrics['volatility_index'] = abs(buy_price - sell_price) / self.metrics['price'] * 100
            
            # Depth quality score
            impact_values = [buy_depth.get('1000', 0), sell_depth.get('1000', 0)]
            self.metrics['depth_quality'] = 90 if all(x < 0.1 for x in impact_values) else \
                                          70 if all(x < 0.3 for x in impact_values) else 40
            
            # Confidence score
            confidence_map = {'high': 90, 'medium': 70, 'low': 50, 'unknown': 30}
            self.metrics['confidence_score'] = confidence_map.get(
                extra_info.get('confidenceLevel', 'unknown').lower(), 30
            )
            
            # Market cap calculation
            supply_info = solana_client.get_token_supply(Pubkey.from_string(self.mint_address))
            if supply_info.value:
                supply = int(supply_info.value.amount) / 10 ** supply_info.value.decimals
                self.metrics['market_cap'] = float(self.raw_data.get('price', 0)) * supply
            
            # Calculate overall rating
            self.metrics['overall_rating'] = (
                self.metrics['liquidity_score'] * self.RATING_WEIGHTS['liquidity'] +
                (100 - self.metrics['volatility_index']) * self.RATING_WEIGHTS['volatility'] +
                self.metrics['depth_quality'] * self.RATING_WEIGHTS['depth_quality'] +
                self.metrics['confidence_score'] * self.RATING_WEIGHTS['confidence']
            )
            
            self.metrics['last_updated'] = datetime.now().isoformat()
            
        except Exception as e:
            st.error(f"Metric calculation error: {str(e)}")

    def get_rating(self):
        """Get professional rating report"""
        return {
            'Mint Address': self.mint_address,
            'Price': f"${self.metrics['price']:.4f}",
            'Market Cap': self.metrics['market_cap'],
            'Overall Rating': f"{self.metrics['overall_rating']:.1f}/100",
            'Liquidity Score': self.metrics['liquidity_score'],
            'Volatility Index': self.metrics['volatility_index'],
            'Depth Quality': self.metrics['depth_quality'],
            'Confidence Score': self.metrics['confidence_score'],
            'Last Updated': self.metrics['last_updated']
        }

def initialize_session():
    """Initialize session state"""
    if 'ratings_history' not in st.session_state:
        try:
            st.session_state.ratings_history = pd.read_csv(HISTORY_FILE)
        except:
            st.session_state.ratings_history = pd.DataFrame(columns=[
                'timestamp', 'mint_address', 'overall_rating', 'price',
                'market_cap', 'liquidity_score', 'volatility_index',
                'depth_quality', 'confidence_score'
            ])

def save_history():
    """Save ratings history"""
    st.session_state.ratings_history.to_csv(HISTORY_FILE, index=False)

def display_rating(report, min_score):
    """Display professional rating card with filtering"""
    rating = float(report['Overall Rating'].split('/')[0])
    if rating < min_score:
        return False
        
    with st.expander(f"📈 {report['Mint Address'][:6]}...{report['Mint Address'][-6:]} - Rating: {rating:.1f}/100"):
        cols = st.columns(4)
        cols[0].metric("Overall Rating", report['Overall Rating'])
        cols[1].metric("Liquidity Score", f"{report['Liquidity Score']:.1f}")
        cols[2].metric("Volatility", f"{report['Volatility Index']:.2f}%")
        cols[3].metric("Market Cap", f"${report['Market Cap']/1e6:.2f}M" if report['Market Cap'] > 1e6 else f"${report['Market Cap']:,.2f}")
        
        cols = st.columns(2)
        cols[0].progress(report['Depth Quality']/100, f"Depth Quality: {report['Depth Quality']:.1f}")
        cols[1].progress(report['Confidence Score']/100, f"Confidence: {report['Confidence Score']:.1f}")
        
    return True

def main():
    st.set_page_config(page_title="Professional Token Rater", layout="wide")
    st.title("🔍 Institutional Token Rating System")
    
    initialize_session()
    
    # Controls
    with st.sidebar:
        st.header("Rating Parameters")
        min_score = st.slider("Minimum Rating Score", 0, 100, 70)
        min_mcap = st.number_input("Minimum Market Cap (USD)", 0, 1000000000, 1000000)
        max_volatility = st.slider("Maximum Volatility (%)", 0, 100, 30)
        tokens = st.text_area("Token Addresses (comma separated)", 
                            "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN, So11111111111111111111111111111111111111112")
        
        if st.button("💼 Run Professional Analysis", type="primary"):
            with st.spinner("Analyzing tokens..."):
                start_time = time.time()
                valid_tokens = 0
                
                for mint in [m.strip() for m in tokens.split(',') if m.strip()]:
                    analyzer = TokenAnalyzer(mint)
                    analyzer.fetch_data()
                    if analyzer.raw_data:
                        report = analyzer.get_rating()
                        if display_rating(report, min_score):
                            # Add to history
                            new_entry = {
                                'timestamp': datetime.now().isoformat(),
                                'mint_address': mint,
                                'overall_rating': float(report['Overall Rating'].split('/')[0]),
                                'price': float(report['Price'].replace('$', '')),
                                'market_cap': report['Market Cap'],
                                'liquidity_score': report['Liquidity Score'],
                                'volatility_index': report['Volatility Index'],
                                'depth_quality': report['Depth Quality'],
                                'confidence_score': report['Confidence Score']
                            }
                            st.session_state.ratings_history = pd.concat([
                                st.session_state.ratings_history,
                                pd.DataFrame([new_entry])
                            ], ignore_index=True)
                            valid_tokens += 1
                        time.sleep(0.3)  # Rate limiting
                
                save_history()
                st.success(f"Analyzed {valid_tokens} tokens in {time.time() - start_time:.2f}s")
        
        st.download_button(
            "📥 Download Full Report",
            data=st.session_state.ratings_history.to_csv(index=False),
            file_name="token_ratings.csv",
            mime="text/csv"
        )
        
        if st.button("🔄 Clear History"):
            st.session_state.ratings_history = pd.DataFrame(columns=[
                'timestamp', 'mint_address', 'overall_rating', 'price',
                'market_cap', 'liquidity_score', 'volatility_index',
                'depth_quality', 'confidence_score'
            ])
            save_history()
            st.success("History cleared")

    # Main display
    if not st.session_state.ratings_history.empty:
        st.subheader("Rating History")
        
        # Apply filters
        filtered = st.session_state.ratings_history[
            (st.session_state.ratings_history['overall_rating'] >= min_score) &
            (st.session_state.ratings_history['market_cap'] >= min_mcap) &
            (st.session_state.ratings_history['volatility_index'] <= max_volatility)
        ]
        
        # Display dataframe
        st.dataframe(
            filtered.sort_values('overall_rating', ascending=False),
            column_config={
                'timestamp': 'Time',
                'mint_address': 'Token',
                'overall_rating': st.column_config.NumberColumn(
                    'Rating', format="%.1f/100"
                ),
                'price': st.column_config.NumberColumn(
                    'Price', format="$%.4f"
                ),
                'market_cap': st.column_config.NumberColumn(
                    'Market Cap', format="$%.2f"
                ),
                'liquidity_score': 'Liquidity',
                'volatility_index': 'Volatility %',
                'depth_quality': 'Depth',
                'confidence_score': 'Confidence'
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )

if __name__ == "__main__":
    main()
