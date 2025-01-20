import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time

# Configuration
COINCAP_API = "https://api.coincap.io/v2"
RESULTS_FILE = "crypto_scan_results.csv"

# ======================
# Data Fetching & Processing
# ======================
def fetch_assets():
    """Fetch top assets from CoinCap API with error handling"""
    try:
        response = requests.get(f"{COINCAP_API}/assets", params={
            'limit': 100,
            'sort': 'volumeUsd24Hr',
            'order': 'desc'
        })
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return []

def process_asset(asset):
    """Convert API response to cleaned data with proper types"""
    return {
        'id': asset['id'],
        'symbol': asset['symbol'].upper(),
        'name': asset['name'],
        'price': float(asset['priceUsd']),
        'market_cap': float(asset['marketCapUsd']),
        'volume_24h': float(asset['volumeUsd24Hr']),
        'change_24h': float(asset['changePercent24Hr']),
        'supply': float(asset['supply']),
        'max_supply': float(asset['maxSupply']) if asset['maxSupply'] else None,
        'rank': int(asset['rank']),
        'vwap_24h': float(asset['vwap24Hr']),
        'timestamp': datetime.now().isoformat(),
        'explorer': asset.get('explorer', '')
    }

# ======================
# Professional Analysis Engine
# ======================
class AssetAnalyzer:
    def __init__(self, params):
        self.params = params
        
    def analyze(self, asset):
        """Comprehensive asset analysis with multiple metrics"""
        analysis = {
            'symbol': asset['symbol'],
            'price': asset['price'],
            'market_cap': asset['market_cap'],
            'volume_24h': asset['volume_24h'],
            'change_24h': asset['change_24h'],
            'vwap_deviation_pct': self.calculate_vwap_deviation(asset),
            'supply_utilization': self.calculate_supply_utilization(asset),
            'volume_mcap_ratio': self.calculate_volume_mcap_ratio(asset),
            'liquidity_score': self.calculate_liquidity_score(asset),
            'timestamp': asset.get('timestamp', datetime.now().isoformat())
        }
        
        if self.passes_filters(analysis):
            return analysis
        return None

    def calculate_vwap_deviation(self, asset):
        try:
            return ((asset['price'] - asset['vwap_24h']) / asset['vwap_24h']) * 100
        except ZeroDivisionError:
            return 0.0

    def calculate_supply_utilization(self, asset):
        if asset['max_supply'] and asset['max_supply'] > 0:
            return (asset['supply'] / asset['max_supply']) * 100
        return 100.0

    def calculate_volume_mcap_ratio(self, asset):
        if asset['market_cap'] > 0:
            return (asset['volume_24h'] / asset['market_cap']) * 100
        return 0.0

    def calculate_liquidity_score(self, asset):
        return (asset['volume_24h'] * 0.4 + 
                asset['market_cap'] * 0.3 + 
                (100 + asset['change_24h']) * 0.3)

    def passes_filters(self, analysis):
        checks = [
            analysis['market_cap'] >= self.params['min_mcap'],
            analysis['volume_24h'] >= self.params['min_volume'],
            analysis['price'] >= self.params['min_price'],
            analysis['change_24h'] >= self.params['min_change_24h'],
            analysis['volume_mcap_ratio'] >= self.params['min_volume_ratio'],
            analysis['vwap_deviation_pct'] >= self.params['min_vwap_deviation'],
            analysis['liquidity_score'] >= self.params['min_liquidity_score']
        ]
        return all(checks)

# ======================
# Enhanced State Management
# ======================
def load_results():
    """Load results with column validation"""
    required_cols = [
        'timestamp', 'symbol', 'price', 'market_cap',
        'volume_24h', 'change_24h', 'vwap_deviation_pct',
        'supply_utilization', 'volume_mcap_ratio', 'liquidity_score'
    ]
    
    try:
        df = pd.read_csv(RESULTS_FILE)
        # Add missing columns with default values
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NaT if col == 'timestamp' else 0.0
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=required_cols)

def save_results(df):
    """Save results with column validation"""
    required_cols = [
        'timestamp', 'symbol', 'price', 'market_cap',
        'volume_24h', 'change_24h', 'vwap_deviation_pct',
        'supply_utilization', 'volume_mcap_ratio', 'liquidity_score'
    ]
    
    # Create new DataFrame with required columns
    safe_df = pd.DataFrame(columns=required_cols)
    
    # Copy existing data
    for col in required_cols:
        if col in df.columns:
            safe_df[col] = df[col]
        else:
            safe_df[col] = pd.NaT if col == 'timestamp' else 0.0
    
    safe_df.to_csv(RESULTS_FILE, index=False)

# ======================
# Streamlit UI with Fixes
# ======================
def main():
    st.set_page_config(page_title="Professional Crypto Scanner", layout="wide")
    st.title("📊 Institutional-Grade Crypto Asset Scanner")

    # Initialize session state
    if 'params' not in st.session_state:
        st.session_state.params = {
            'min_mcap': 1e9,
            'min_volume': 5e7,
            'min_price': 1.0,
            'min_change_24h': 2.0,
            'min_volume_ratio': 0.5,
            'min_vwap_deviation': -2.0,
            'min_liquidity_score': 50.0,
            'excluded_assets': ['USDT', 'USDC', 'BUSD']
        }
    
    if 'results' not in st.session_state:
        st.session_state.results = load_results()

    # Sidebar Controls
    with st.sidebar:
        st.header("Analysis Parameters")
        
        st.session_state.params['min_mcap'] = st.number_input(
            "Minimum Market Cap (USD)", 
            value=1e9, 
            format="%.0f", 
            step=1e6
        )
        
        st.session_state.params['min_volume'] = st.number_input(
            "Minimum 24h Volume (USD)", 
            value=5e7, 
            format="%.0f",
            step=1e5
        )
        
        st.session_state.params['min_price'] = st.number_input(
            "Minimum Price (USD)", 
            value=1.0, 
            step=0.1,
            format="%.2f"
        )
        
        st.session_state.params['min_change_24h'] = st.number_input(
            "Minimum 24h Change (%)", 
            value=2.0, 
            step=0.1
        )
        
        st.session_state.params['min_volume_ratio'] = st.slider(
            "Volume/MCap Ratio (%)", 
            0.0, 10.0, 0.5, 0.1
        )
        
        st.session_state.params['min_vwap_deviation'] = st.slider(
            "VWAP Deviation (%)", 
            -10.0, 10.0, -2.0, 0.1
        )
        
        st.session_state.params['min_liquidity_score'] = st.slider(
            "Liquidity Score Threshold", 
            0.0, 100.0, 50.0, 1.0
        )

        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Scanning crypto markets..."):
                start_time = time.time()
                raw_assets = fetch_assets()
                processed_assets = [process_asset(a) for a in raw_assets]
                filtered_assets = [
                    a for a in processed_assets 
                    if a['symbol'] not in st.session_state.params['excluded_assets']
                ]
                
                analyzer = AssetAnalyzer(st.session_state.params)
                new_results = []
                
                for asset in filtered_assets:
                    result = analyzer.analyze(asset)
                    if result:
                        new_results.append(result)
                
                if new_results:
                    new_df = pd.DataFrame(new_results)
                    st.session_state.results = pd.concat(
                        [st.session_state.results, new_df], 
                        ignore_index=True
                    )
                    save_results(st.session_state.results)
                    st.success(f"Found {len(new_results)} new opportunities!")
                else:
                    st.warning("No assets matching current criteria")
                
                st.write(f"Analysis completed in {time.time() - start_time:.2f} seconds")

        st.download_button(
            "💾 Download Results",
            data=st.session_state.results.to_csv(index=False),
            file_name="crypto_opportunities.csv",
            mime="text/csv"
        )

        if st.button("🔄 Clear Results"):
            st.session_state.results = pd.DataFrame(columns=load_results().columns)
            save_results(st.session_state.results)
            st.success("Results cleared successfully")

    # Main Display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Opportunities", len(st.session_state.results))
    with col2:
        if not st.session_state.results.empty and 'timestamp' in st.session_state.results:
            recent_count = len(st.session_state.results[
                pd.to_datetime(st.session_state.results['timestamp']) > 
                (datetime.now() - timedelta(hours=24))
            ])
            st.metric("24h New Opportunities", recent_count)
        else:
            st.metric("24h New Opportunities", 0)

    # Results Display
    if not st.session_state.results.empty:
        display_df = st.session_state.results.copy()
        
        # Ensure all columns exist
        for col in ['price', 'market_cap', 'volume_24h', 'change_24h', 
                   'vwap_deviation_pct', 'supply_utilization', 
                   'volume_mcap_ratio', 'liquidity_score']:
            if col not in display_df.columns:
                display_df[col] = 0.0

        # Format display values
        display_df['price'] = display_df['price'].apply(
            lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
        display_df['market_cap'] = display_df['market_cap'].apply(
            lambda x: f"${x/1e9:,.2f}B" if x >= 1e9 else f"${x/1e6:,.2f}M")
        display_df['volume_24h'] = display_df['volume_24h'].apply(
            lambda x: f"${x/1e6:,.2f}M" if pd.notnull(x) else "N/A")
        display_df['change_24h'] = display_df['change_24h'].apply(
            lambda x: f"{x:.2f}")
        display_df['vwap_deviation_pct'] = display_df['vwap_deviation_pct'].apply(
            lambda x: f"{x:.2f}%")
        display_df['supply_utilization'] = display_df['supply_utilization'].apply(
            lambda x: f"{x:.2f}%")
        display_df['volume_mcap_ratio'] = display_df['volume_mcap_ratio'].apply(
            lambda x: f"{x:.2f}%")
        display_df['liquidity_score'] = display_df['liquidity_score'].apply(
            lambda x: f"{x:.2f}/100")

        st.dataframe(
            display_df.sort_values('timestamp', ascending=False),
            column_config={
                'symbol': 'Symbol',
                'price': 'Price',
                'market_cap': 'Market Cap',
                'volume_24h': '24h Volume',
                'change_24h': st.column_config.NumberColumn(
                    '24h Change',
                    format="%.2f%%"
                ),
                'vwap_deviation_pct': 'VWAP Deviation',
                'supply_utilization': st.column_config.ProgressColumn(
                    'Supply Utilization',
                    format="%.2f%%",
                    min_value=0,
                    max_value=100
                ),
                'volume_mcap_ratio': 'Volume/MCap',
                'liquidity_score': 'Liquidity Score'
            },
            height=600,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Run analysis to discover market opportunities")

if __name__ == "__main__":
    main()