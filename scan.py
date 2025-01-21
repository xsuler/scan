import streamlit as st
import pandas as pd
import requests
import base58
import time
from datetime import datetime
from solders.pubkey import Pubkey
from solana.rpc.api import Client

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
SOLANA_RPC_ENDPOINT = "https://api.mainnet-beta.solana.com"
RESULTS_FILE = "solana_contract_scan.csv"

# Initialize Solana client
solana_client = Client(SOLANA_RPC_ENDPOINT)

# ======================
# Core Blockchain Functions
# ======================

def fetch_token_list():
    """Fetch and validate token list from Jupiter Aggregator"""
    try:
        response = requests.get(JUPITER_TOKEN_LIST, timeout=15)
        response.raise_for_status()
        tokens = response.json()
        return [t for t in tokens if _is_valid_spl_token(t.get('address'))]
    except Exception as e:
        st.error(f"🚨 Token list error: {str(e)}")
        return []

def _is_valid_spl_token(mint_address: str) -> bool:
    """Validate SPL token address structure"""
    try:
        if mint_address == "So11111111111111111111111111111111111111112":
            return False
        Pubkey.from_string(mint_address)
        return True
    except (ValueError, AttributeError):
        return False

def get_token_price(mint_address: str) -> float:
    """Get price from Jupiter v2 API with enhanced error handling"""
    try:
        response = requests.get(
            f"{JUPITER_PRICE_API}?ids={mint_address}",
            timeout=10
        )
        if response.status_code == 200:
            price_data = response.json()
            if (price_data.get('data') and 
                mint_address in price_data['data'] and 
                'price' in price_data['data'][mint_address]):
                return float(price_data['data'][mint_address]['price'])
        return 0.0
    except Exception as e:
        st.error(f"💸 Price error for {mint_address}: {str(e)}")
        return 0.0

def get_token_metadata(mint_address: str):
    """Get social/ecosystem metadata for token"""
    try:
        response = requests.get(JUPITER_TOKEN_LIST, timeout=10)
        tokens = response.json()
        for t in tokens:
            if t['address'] == mint_address:
                return {
                    'twitter': t.get('extensions', {}).get('twitter'),
                    'website': t.get('extensions', {}).get('website'),
                    'tags': t.get('tags', []),
                    'verified': t.get('extensions', {}).get('verified', False)
                }
        return {}
    except:
        return {}

def get_holder_metrics(mint_address: str):
    """Get holder distribution metrics"""
    try:
        response = solana_client.get_token_largest_accounts(Pubkey.from_string(mint_address))
        accounts = response.value
        if not accounts:
            return {}
            
        total = sum(int(acc.amount) for acc in accounts)
        top10 = sum(int(acc.amount) for acc in accounts[:10])
        return {
            'holder_count': len(accounts),
            'top10_holders': (top10 / total) * 100 if total > 0 else 0
        }
    except:
        return {}

# ======================
# Data Processing Engine
# ======================

def process_token_entry(token):
    """Full token data processing pipeline"""
    mint_address = token.get('address')
    if not mint_address or not _is_valid_spl_token(mint_address):
        return None

    try:
        # Get blockchain data
        supply_info = solana_client.get_token_supply(Pubkey.from_string(mint_address))
        if not supply_info.value:
            return None
            
        decimals = supply_info.value.decimals
        raw_supply = int(supply_info.value.amount)
        supply = raw_supply / (10 ** decimals)
        price = get_token_price(mint_address)
        if price <= 0:
            return None

        # Calculate age from mint address
        decoded = bytes(Pubkey.from_string(mint_address))
        timestamp = int.from_bytes(decoded[:4], byteorder='big') / 1000
        age_days = (datetime.now() - datetime.fromtimestamp(timestamp)).days

        # Get additional metrics
        metadata = get_token_metadata(mint_address)
        holders = get_holder_metrics(mint_address)
        
        # Calculate market cap
        market_cap = price * supply

        return {
            'symbol': token['symbol'].upper(),
            'name': token['name'],
            'contract_address': mint_address,
            'price': price,
            'market_cap': market_cap,
            'supply': supply,
            'age_days': age_days,
            'decimals': decimals,
            'twitter': bool(metadata.get('twitter')),
            'website': bool(metadata.get('website')),
            'tags': ', '.join(metadata.get('tags', [])),
            'verified': metadata.get('verified', False),
            'holder_count': holders.get('holder_count', 0),
            'top10_holders': holders.get('top10_holders', 0),
            'liquidity_score': calculate_liquidity_score(
                price=price,
                market_cap=market_cap,
                age_days=age_days,
                verified=metadata.get('verified'),
                holder_count=holders.get('holder_count'),
                top10_holders=holders.get('top10_holders'),
                social_score=sum([bool(metadata.get('twitter')), bool(metadata.get('website'))])
            ),
            'explorer': f"https://solscan.io/token/{mint_address}",
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"🔧 Processing error for {mint_address}: {str(e)}")
        return None

def calculate_liquidity_score(**metrics):
    """Professional liquidity scoring algorithm"""
    score = 0
    
    # Market Cap Score (0-30 points)
    score += min(30, metrics['market_cap'] / 1e6 * 0.3) if metrics['market_cap'] else 0
    
    # Verification Score (0-15 points)
    score += 15 if metrics['verified'] else 0
    
    # Age Score (0-20 points)
    score += min(20, metrics['age_days'] * 0.2)
    
    # Social Score (0-10 points)
    score += metrics['social_score'] * 5  # 5 points per social presence
    
    # Holder Distribution Score (0-25 points)
    holder_score = min(15, metrics['holder_count'] / 1000)  # 1 point per 1000 holders
    concentration_score = 10 - (metrics['top10_holders'] * 0.1)  # Lower concentration = better
    score += holder_score + max(0, concentration_score)

    return round(min(100, score), 1)

# ======================
# Analysis & UI Components
# ======================

class BlockchainAnalyzer:
    def __init__(self, params):
        self.params = params

    def analyze_token(self, token):
        """Apply professional-grade filters"""
        if not token:
            return None

        checks = [
            token['market_cap'] >= self.params['min_mcap'],
            token['price'] >= self.params['min_price'],
            token['liquidity_score'] >= self.params['min_liquidity'],
            token['age_days'] >= self.params['min_age'],
            token['holder_count'] >= self.params['min_holders'],
            (not self.params['verified_only'] or token['verified']),
            token['top10_holders'] <= self.params['max_concentration']
        ]
        return token if all(checks) else None

def initialize_session():
    """Initialize Streamlit session state"""
    if 'params' not in st.session_state:
        st.session_state.params = {
            'min_mcap': 1e4,
            'min_price': 0.001,
            'min_liquidity': 50,
            'min_age': 7,
            'min_holders': 100,
            'max_concentration': 90,
            'verified_only': False
        }
    
    if 'results' not in st.session_state:
        try:
            st.session_state.results = pd.read_csv(RESULTS_FILE)
        except:
            st.session_state.results = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'contract_address', 'price', 
                'market_cap', 'liquidity_score', 'verified', 'age_days',
                'holder_count', 'top10_holders', 'tags', 'explorer'
            ])

def save_results():
    """Save results with data validation"""
    st.session_state.results.to_csv(RESULTS_FILE, index=False)

def display_metrics():
    """Show real-time analysis metrics"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Contracts", len(st.session_state.results))
    with col2:
        avg_score = st.session_state.results['liquidity_score'].mean()
        st.metric("Avg Liquidity Score", f"{avg_score:.1f}/100")
    with col3:
        new_today = len(st.session_state.results[
            pd.to_datetime(st.session_state.results['timestamp']).dt.date == datetime.today().date()
        ])
        st.metric("New Today", new_today)
    with col4:
        verified_count = st.session_state.results['verified'].sum()
        st.metric("Verified Contracts", verified_count)

def display_results():
    """Professional results display"""
    if st.session_state.results.empty:
        st.info("📭 No results found. Run a scan first.")
        return

    df = st.session_state.results.copy()
    
    # Formatting
    df['price'] = df['price'].apply(lambda x: f"${x:.4f}" if x >= 0.0001 else f"${x:.8f}")
    df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e6:.2f}M" if x >= 1e6 else f"${x/1e3:,.0f}K")
    df['top10_holders'] = df['top10_holders'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        df.sort_values('liquidity_score', ascending=False),
        column_config={
            'symbol': 'Symbol',
            'contract_address': 'Contract Address',
            'price': 'Price',
            'market_cap': 'Market Cap',
            'liquidity_score': st.column_config.ProgressColumn(
                'Liquidity Score',
                format="%.1f",
                min_value=0,
                max_value=100
            ),
            'age_days': 'Age (Days)',
            'holder_count': 'Holders',
            'top10_holders': 'Top 10%',
            'verified': 'Verified',
            'tags': 'Categories',
            'explorer': st.column_config.LinkColumn('Explorer')
        },
        column_order=[
            'symbol', 'price', 'market_cap', 'liquidity_score',
            'age_days', 'holder_count', 'top10_holders', 'verified',
            'tags', 'contract_address', 'explorer'
        ],
        height=700,
        use_container_width=True,
        hide_index=True
    )

# ======================
# Streamlit UI
# ======================

def main():
    st.set_page_config(
        page_title="Professional Solana Scanner",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🔬 Professional Solana Contract Analyzer")
    
    # Session initialization
    initialize_session()

    # Sidebar Controls
    with st.sidebar:
        st.header("Analysis Parameters")
        
        st.subheader("Core Metrics")
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
        st.session_state.params['min_liquidity'] = st.slider(
            "Liquidity Score Threshold",
            0, 100, 50
        )
        
        st.subheader("Advanced Filters")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.params['min_age'] = st.slider(
                "Minimum Age (Days)",
                0, 365, 7
            )
            st.session_state.params['min_holders'] = st.number_input(
                "Minimum Holders",
                1, 100000, 100
            )
        with col2:
            st.session_state.params['max_concentration'] = st.slider(
                "Max Top 10% Concentration",
                1, 100, 90
            )
            st.session_state.params['verified_only'] = st.checkbox(
                "Verified Contracts Only"
            )
        
        if st.button("🚀 Start Deep Analysis", type="primary"):
            with st.spinner("🔍 Scanning blockchain..."):
                start_time = time.time()
                try:
                    tokens = fetch_token_list()
                    analyzer = BlockchainAnalyzer(st.session_state.params)
                    
                    results = []
                    for token in tokens[:150]:  # Process top 150 tokens
                        if processed := process_token_entry(token):
                            if analyzed := analyzer.analyze_token(processed):
                                results.append(analyzed)
                        time.sleep(0.1)  # Rate limiting

                    if results:
                        new_df = pd.DataFrame(results)
                        st.session_state.results = pd.concat(
                            [st.session_state.results, new_df]
                        ).drop_duplicates('contract_address', keep='last')
                        save_results()
                        st.success(f"✅ Found {len(results)} qualifying contracts!")
                    else:
                        st.warning("⚠️ No contracts matched criteria")
                    
                    st.write(f"⏱️ Scan completed in {time.time() - start_time:.2f}s")
                except Exception as e:
                    st.error(f"🔥 Critical error: {str(e)}")

        st.download_button(
            "💾 Export Full Report",
            data=st.session_state.results.to_csv(index=False),
            file_name="solana_contract_analysis.csv",
            mime="text/csv"
        )

        if st.button("🔄 Reset Database"):
            st.session_state.results = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'contract_address', 'price',
                'market_cap', 'liquidity_score', 'verified', 'age_days',
                'holder_count', 'top10_holders', 'tags', 'explorer'
            ])
            save_results()
            st.success("♻️ Database cleared successfully")

    # Main Display
    display_metrics()
    display_results()

if __name__ == "__main__":
    main()
