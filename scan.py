import streamlit as st
import pandas as pd
import requests
import base58
import time
from datetime import datetime
from solana.rpc.api import Client

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
SOLANA_RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"
RESULTS_FILE = "solana_contract_scan.csv"

# Initialize Solana client
solana_client = Client(SOLANA_RPC_ENDPOINT)

# ======================
# Blockchain Data Functions
# ======================
def fetch_token_list():
    """Fetch complete token list from Jupiter Aggregator"""
    try:
        response = requests.get(JUPITER_TOKEN_LIST, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Token list fetch error: {str(e)}")
        return []

def get_token_details(mint_address):
    """Get on-chain token details including supply and metadata"""
    try:
        # Get token supply
        supply_info = solana_client.get_token_supply(mint_address).value
        decimals = supply_info.decimals if supply_info else 9
        raw_supply = int(supply_info.amount) if supply_info else 0
        supply = raw_supply / (10 ** decimals)

        # Get mint timestamp from address
        decoded = base58.b58decode(mint_address)
        timestamp = int.from_bytes(decoded[:4], byteorder='big') / 1000
        age_days = (datetime.now() - datetime.fromtimestamp(timestamp)).days

        # Get price from Jupiter price API
        price_response = requests.get(
            f"https://price.jup.ag/v4/price?ids={mint_address}",
            timeout=5
        )
        price_data = price_response.json() if price_response.status_code == 200 else {}
        price = price_data.get('data', {}).get(mint_address, {}).get('price', 0)

        return {
            'supply': supply,
            'decimals': decimals,
            'age_days': age_days,
            'price': float(price),
            'market_cap': float(price) * supply,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Detail fetch error for {mint_address}: {str(e)}")
        return None

# ======================
# Data Processing
# ======================
def process_token_entry(token):
    """Process raw token data into structured format"""
    details = get_token_details(token['address'])
    if not details:
        return None

    return {
        'symbol': token['symbol'].upper(),
        'name': token['name'],
        'contract_address': token['address'],
        'price': details['price'],
        'market_cap': details['market_cap'],
        'supply': details['supply'],
        'age_days': details['age_days'],
        'decimals': details['decimals'],
        'verified': token.get('extensions', {}).get('verified', False),
        'liquidity_score': calculate_liquidity_score(token, details),
        'explorer': f"https://solscan.io/token/{token['address']}",
        **details
    }

def calculate_liquidity_score(token, details):
    """Calculate composite liquidity score"""
    score = 0
    # Market cap component (0-50)
    score += min(50, details['market_cap'] / 1e6 * 0.5)
    # Verification bonus
    score += 30 if token.get('extensions', {}).get('verified') else 0
    # Age component
    score += min(20, details['age_days'] * 0.2)
    return round(score, 1)

# ======================
# Analysis Engine
# ======================
class BlockchainAnalyzer:
    def __init__(self, params):
        self.params = params
        self.required_fields = [
            'symbol', 'contract_address', 'price', 
            'market_cap', 'liquidity_score', 'verified'
        ]

    def analyze_token(self, token):
        """Apply analysis filters to token data"""
        if not token or token['price'] <= 0:
            return None

        checks = [
            token['market_cap'] >= self.params['min_mcap'],
            token['price'] >= self.params['min_price'],
            token['liquidity_score'] >= self.params['min_liquidity'],
            token['verified'] or not self.params['verified_only'],
            token['age_days'] >= self.params['min_age']
        ]

        return token if all(checks) else None

# ======================
# UI & Data Management
# ======================
def initialize_session():
    """Initialize Streamlit session state"""
    if 'params' not in st.session_state:
        st.session_state.params = {
            'min_mcap': 1e4,
            'min_price': 0.001,
            'min_liquidity': 50,
            'verified_only': False,
            'min_age': 7
        }
    
    if 'results' not in st.session_state:
        try:
            st.session_state.results = pd.read_csv(RESULTS_FILE)
        except:
            st.session_state.results = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'contract_address', 'price',
                'market_cap', 'liquidity_score', 'verified', 'age_days'
            ])

def save_results():
    """Persist results to CSV"""
    st.session_state.results.to_csv(RESULTS_FILE, index=False)

def display_results():
    """Render results in formatted dataframe"""
    if st.session_state.results.empty:
        st.info("No scan results available. Run a scan first.")
        return

    df = st.session_state.results.copy()
    df['price'] = df['price'].apply(
        lambda x: f"${x:.4f}" if x >= 0.0001 else f"${x:.8f}")
    df['market_cap'] = df['market_cap'].apply(
        lambda x: f"${x/1e3:,.1f}K" if x < 1e6 else f"${x/1e6:.2f}M")
    
    st.dataframe(
        df.sort_values('market_cap', ascending=False),
        column_config={
            'symbol': 'Symbol',
            'price': 'Price',
            'market_cap': 'Market Cap',
            'liquidity_score': st.column_config.ProgressColumn(
                'Liquidity',
                format="%.1f",
                min_value=0,
                max_value=100
            ),
            'contract_address': 'Contract Address',
            'verified': 'Verified',
            'age_days': 'Age (Days)',
            'explorer': st.column_config.LinkColumn('Block Explorer')
        },
        height=700,
        use_container_width=True,
        hide_index=True
    )

# ======================
# Streamlit UI
# ======================
def main():
    st.set_page_config(
        page_title="Solana Contract Scanner",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🔭 Solana Smart Contract Analyzer")

    initialize_session()

    # Sidebar Controls
    with st.sidebar:
        st.header("Analysis Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.params['min_mcap'] = st.number_input(
                "Minimum Market Cap (USD)",
                min_value=0.0,
                value=1e4,
                step=1e3,
                format="%.0f"
            )
            st.session_state.params['min_price'] = st.number_input(
                "Minimum Price (USD)",
                min_value=0.0,
                value=0.001,
                step=0.001,
                format="%.4f"
            )
            
        with col2:
            st.session_state.params['min_liquidity'] = st.slider(
                "Liquidity Score Threshold",
                min_value=0,
                max_value=100,
                value=50
            )
            st.session_state.params['min_age'] = st.slider(
                "Minimum Contract Age (Days)",
                min_value=0,
                max_value=365,
                value=7
            )
        
        st.session_state.params['verified_only'] = st.checkbox(
            "Verified Contracts Only",
            value=False
        )

        if st.button("🔍 Start Blockchain Scan", type="primary"):
            with st.spinner("Scanning Solana blockchain..."):
                start_time = time.time()
                tokens = fetch_token_list()
                analyzer = BlockchainAnalyzer(st.session_state.params)
                
                new_results = []
                for token in tokens[:100]:  # Limit to first 100 for performance
                    processed = process_token_entry(token)
                    if result := analyzer.analyze_token(processed):
                        new_results.append(result)
                    time.sleep(0.1)  # Rate limiting

                if new_results:
                    new_df = pd.DataFrame(new_results)
                    st.session_state.results = pd.concat(
                        [st.session_state.results, new_df],
                        ignore_index=True
                    ).drop_duplicates(['contract_address'], keep='last')
                    save_results()
                    st.success(f"Found {len(new_results)} qualifying contracts!")
                else:
                    st.warning("No matching contracts found")
                
                st.write(f"Scan duration: {time.time() - start_time:.2f} seconds")

        st.download_button(
            "📥 Download Results",
            data=st.session_state.results.to_csv(index=False),
            file_name="solana_contracts.csv",
            mime="text/csv"
        )

        if st.button("🔄 Reset Results"):
            st.session_state.results = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'contract_address', 'price',
                'market_cap', 'liquidity_score', 'verified', 'age_days'
            ])
            save_results()
            st.success("Results cleared successfully")

    # Main Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Contracts", len(st.session_state.results))
    with col2:
        avg_score = st.session_state.results['liquidity_score'].mean() \
            if not st.session_state.results.empty else 0
        st.metric("Average Liquidity Score", f"{avg_score:.1f}/100")
    with col3:
        new_today = len(st.session_state.results[
            pd.to_datetime(st.session_state.results['timestamp']).dt.date == datetime.today().date()
        ]) if not st.session_state.results.empty else 0
        st.metric("New Today", new_today)

    display_results()

if __name__ == "__main__":
    main()
