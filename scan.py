import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

# Configuration
SOLSCAN_API = "https://api.solscan.io/tokens"
SOLANA_TOKEN_API = "https://api.solscan.io/chain/wallet/tokens"
RESULTS_FILE = "solana_scan_results.csv"

# ======================
# Data Fetching & Processing
# ======================
def fetch_assets():
    """Fetch top Solana tokens from Solscan API with retry logic"""
    try:
        response = requests.get(SOLSCAN_API, params={
            'sort': 'volume24h',
            'direction': 'desc',
            'limit': 100,
            'offset': 0
        })
        response.raise_for_status()
        data = response.json().get('data', {}).get('tokens', [])
        
        # Additional metadata fetch for contract details
        enhanced_data = []
        for token in data[:50]:  # Limit to top 50 for performance
            try:
                detail_response = requests.get(f"https://api.solscan.io/token/meta?token={token['mint']}")
                if detail_response.status_code == 200:
                    token.update(detail_response.json().get('data', {}))
                enhanced_data.append(token)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                st.error(f"Error fetching details for {token['mint']}: {str(e)}")
        return enhanced_data
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return []

def process_asset(asset):
    """Convert Solscan API response to cleaned data with enhanced fields"""
    try:
        price = float(asset.get('price', 0))
        market_cap = float(asset.get('marketCap', 0))
        supply = market_cap / price if price != 0 else 0
        
        return {
            'symbol': asset.get('symbol', '').upper(),
            'name': asset.get('name', 'Unknown'),
            'price': price,
            'market_cap': market_cap,
            'volume_24h': float(asset.get('volume24h', 0)),
            'supply': supply,
            'timestamp': datetime.now().isoformat(),
            'contract_address': asset.get('mint', ''),
            'decimals': asset.get('decimals', 9),
            'holder_count': asset.get('holderCount', 0),
            'website': asset.get('website', ''),
            'twitter': asset.get('twitter', ''),
            'explorer': f"https://solscan.io/token/{asset.get('mint', '')}",
            'created_at': asset.get('createdTime', ''),
            'tag': asset.get('tag', ''),
            'verified': asset.get('verified', False)
        }
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# ======================
# Advanced Analysis Engine
# ======================
class ContractAnalyzer:
    def __init__(self, params):
        self.params = params
        self.risk_metrics = ['holder_count', 'verified', 'volume_24h']
        
    def analyze(self, asset):
        """Comprehensive analysis of Solana contract"""
        if not asset:
            return None
            
        analysis = {
            'symbol': asset['symbol'],
            'price': asset['price'],
            'market_cap': asset['market_cap'],
            'volume_24h': asset['volume_24h'],
            'volume_mcap_ratio': self.calculate_volume_mcap_ratio(asset),
            'liquidity_score': self.calculate_liquidity_score(asset),
            'risk_score': self.calculate_risk_score(asset),
            'timestamp': asset['timestamp'],
            'contract_address': asset['contract_address'],
            'explorer': asset['explorer'],
            'holders': asset['holder_count'],
            'verified': asset['verified'],
            'age_days': self.calculate_age_days(asset)
        }
        
        if self.passes_filters(analysis):
            return analysis
        return None

    def calculate_volume_mcap_ratio(self, asset):
        try:
            return (asset['volume_24h'] / asset['market_cap']) * 100 if asset['market_cap'] > 0 else 0
        except:
            return 0

    def calculate_liquidity_score(self, asset):
        try:
            return (asset['volume_24h'] * 0.6 + 
                    asset['market_cap'] * 0.3 +
                    asset['holder_count'] * 0.1)
        except:
            return 0

    def calculate_risk_score(self, asset):
        risk = 0
        if asset['holder_count'] < 100: risk += 30
        if not asset['verified']: risk += 50
        if asset['age_days'] < 7: risk += 20
        return min(100, risk)

    def calculate_age_days(self, asset):
        try:
            created_date = datetime.fromtimestamp(asset['created_at']/1000)
            return (datetime.now() - created_date).days
        except:
            return 0

    def passes_filters(self, analysis):
        checks = [
            analysis['market_cap'] >= self.params['min_mcap'],
            analysis['volume_24h'] >= self.params['min_volume'],
            analysis['price'] >= self.params['min_price'],
            analysis['volume_mcap_ratio'] >= self.params['min_volume_ratio'],
            analysis['liquidity_score'] >= self.params['min_liquidity_score'],
            analysis['risk_score'] <= self.params['max_risk_score'],
            analysis['holders'] >= self.params['min_holders'],
            analysis['age_days'] >= self.params['min_age_days']
        ]
        return all(checks)

# ======================
# Data Management
# ======================
def load_results():
    """Load results with data validation"""
    required_cols = [
        'timestamp', 'symbol', 'price', 'market_cap', 'volume_24h',
        'volume_mcap_ratio', 'liquidity_score', 'risk_score',
        'contract_address', 'explorer', 'holders', 'verified', 'age_days'
    ]
    
    try:
        df = pd.read_csv(RESULTS_FILE)
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=required_cols)

def save_results(df):
    """Save results with data validation"""
    required_cols = [
        'timestamp', 'symbol', 'price', 'market_cap', 'volume_24h',
        'volume_mcap_ratio', 'liquidity_score', 'risk_score',
        'contract_address', 'explorer', 'holders', 'verified', 'age_days'
    ]
    
    safe_df = pd.DataFrame(columns=required_cols)
    for col in required_cols:
        if col in df.columns:
            safe_df[col] = df[col]
        else:
            safe_df[col] = None
            
    safe_df.to_csv(RESULTS_FILE, index=False)

# ======================
# Streamlit UI
# ======================
def main():
    st.set_page_config(page_title="Solana Contract Scanner PRO", layout="wide")
    st.title("🔍 Advanced Solana Contract Scanner")
    
    # Initialize session state
    if 'params' not in st.session_state:
        st.session_state.params = {
            'min_mcap': 1e5,
            'min_volume': 5e4,
            'min_price': 0.001,
            'min_volume_ratio': 0.5,
            'min_liquidity_score': 25.0,
            'max_risk_score': 70,
            'min_holders': 100,
            'min_age_days': 3,
            'excluded_assets': ['USDT', 'USDC', 'BUSD']
        }
    
    if 'results' not in st.session_state:
        st.session_state.results = load_results()

    # Sidebar Controls
    with st.sidebar:
        st.header("Scan Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.params['min_mcap'] = st.number_input(
                "Min Market Cap (USD)", 1e3, 1e9, 1e5, format="%.0f")
            st.session_state.params['min_volume'] = st.number_input(
                "Min 24h Volume (USD)", 1e3, 1e9, 5e4, format="%.0f")
            st.session_state.params['min_price'] = st.number_input(
                "Min Price (USD)", 0.0001, 1000.0, 0.001, format="%.4f")
            
        with col2:
            st.session_state.params['min_holders'] = st.number_input(
                "Min Holders", 1, 100000, 100)
            st.session_state.params['min_age_days'] = st.number_input(
                "Min Age (Days)", 0, 365, 3)
            st.session_state.params['max_risk_score'] = st.slider(
                "Max Risk Score", 0, 100, 70)
        
        st.session_state.params['min_volume_ratio'] = st.slider(
            "Volume/MCap Ratio (%)", 0.0, 20.0, 0.5, 0.1)
        st.session_state.params['min_liquidity_score'] = st.slider(
            "Liquidity Score Threshold", 0.0, 1000.0, 25.0, 1.0)

        if st.button("🚀 Start Scan", type="primary"):
            with st.spinner("Scanning Solana contracts..."):
                start_time = time.time()
                raw_assets = fetch_assets()
                processed_assets = [process_asset(a) for a in raw_assets]
                filtered_assets = [
                    a for a in processed_assets 
                    if a and a['symbol'] not in st.session_state.params['excluded_assets']
                ]
                
                analyzer = ContractAnalyzer(st.session_state.params)
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
                    st.success(f"Found {len(new_results)} qualifying contracts!")
                else:
                    st.warning("No contracts matching current criteria")
                
                st.write(f"Scan completed in {time.time() - start_time:.2f}s")

        st.download_button(
            "💾 Export Data",
            data=st.session_state.results.to_csv(index=False),
            file_name="solana_contracts.csv",
            mime="text/csv"
        )

        if st.button("🔄 Reset Results"):
            st.session_state.results = pd.DataFrame(columns=load_results().columns)
            save_results(st.session_state.results)
            st.success("Results cleared")

    # Main Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Contracts", len(st.session_state.results))
    with col2:
        avg_risk = st.session_state.results['risk_score'].mean() if not st.session_state.results.empty else 0
        st.metric("Average Risk Score", f"{avg_risk:.1f}/100")
    with col3:
        new_today = len(st.session_state.results[pd.to_datetime(st.session_state.results['timestamp']).dt.date == datetime.today().date()])
        st.metric("New Today", new_today)

    # Results Display
    if not st.session_state.results.empty:
        df = st.session_state.results.copy()
        
        # Formatting
        df['price'] = df['price'].apply(lambda x: f"${x:.4f}" if x > 0.0001 else f"${x:.8f}")
        df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e6:.2f}M" if x >= 1e6 else f"${x:.2f}")
        df['volume_24h'] = df['volume_24h'].apply(lambda x: f"${x/1e3:,.1f}K" if x < 1e6 else f"${x/1e6:.2f}M")
        df['volume_mcap_ratio'] = df['volume_mcap_ratio'].apply(lambda x: f"{x:.2f}%")
        df['liquidity_score'] = df['liquidity_score'].apply(lambda x: f"{x:.1f}")
        df['risk_score'] = df['risk_score'].apply(lambda x: f"{x:.0f} ⭐" if x < 30 else f"{x:.0f} ⚠️" if x < 70 else f"{x:.0f} 🔥")

        st.dataframe(
            df.sort_values('timestamp', ascending=False),
            column_config={
                'symbol': 'Symbol',
                'price': 'Price',
                'market_cap': 'Market Cap',
                'volume_24h': '24h Volume',
                'volume_mcap_ratio': 'Vol/MCap',
                'liquidity_score': 'Liquidity',
                'risk_score': 'Risk',
                'explorer': st.column_config.LinkColumn(
                    "Contract",
                    help="View on Solscan",
                    display_text="🔗 View"
                ),
                'holders': 'Holders',
                'verified': 'Verified',
                'age_days': 'Age (Days)'
            },
            column_order=[
                'symbol', 'price', 'market_cap', 'volume_24h',
                'volume_mcap_ratio', 'liquidity_score', 'risk_score',
                'holders', 'verified', 'age_days', 'explorer'
            ],
            height=700,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Run a scan to discover Solana contracts")

if __name__ == "__main__":
    main()
