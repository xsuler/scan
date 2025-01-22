import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
from solders.pubkey import Pubkey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
DEFAULT_BATCH_SIZE = 5
ROW_HEIGHT = 35
HEADER_HEIGHT = 50

# Initialize session state
if 'analysis' not in st.session_state:
    st.session_state.update({
        'analysis': {
            'running': False,
            'tokens': [],
            'results': pd.DataFrame(),
            'current_index': 0,
            'start_time': None,
            'total_tokens': 0
        },
        'config': {
            'batch_size': DEFAULT_BATCH_SIZE,
            'strict_mode': True,
            'score_weights': {'liquidity': 0.4, 'stability': 0.4, 'depth': 0.2},
            'price_impact_levels': ['10', '100'],
            'show_advanced': False,
            'columns': ['score', 'symbol', 'price', 'liquidity', 'confidence', 'explorer']
        }
    })

class EnhancedAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
    
    def safe_float(self, value, default=0.0):
        try:
            return float(value) if value not in [None, np.nan, ''] else default
        except:
            return default

    def validate_token(self, token):
        try:
            if not Pubkey.from_string(token.get('address', '')):
                return False

            required_fields = ['symbol', 'name', 'decimals', 'logoURI']
            for field in required_fields:
                if not token.get(field):
                    return False

            if st.session_state.config['strict_mode']:
                if "community" in token.get('tags', []):
                    return False
                if token.get('chainId') != 101:
                    return False

            return True
        except:
            return False

    def get_token_list(self):
        try:
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
            response.raise_for_status()
            return [t for t in response.json() if self.validate_token(t)]
        except Exception as e:
            st.error(f"Token fetch failed: {str(e)}")
            return []

    def analyze_token(self, token):
        try:
            price_data = self.get_price_data(token)
            liquidity_data = self.get_liquidity_data(token)
            
            analysis = {
                'symbol': token.get('symbol', 'Unknown'),
                'address': token['address'],
                'price': self.safe_float(price_data.get('price', 0)),
                'buy_price': self.safe_float(price_data.get('buy_price', 0)),
                'sell_price': self.safe_float(price_data.get('sell_price', 0)),
                'price_impact': liquidity_data.get('price_impact', {}),
                'liquidity': liquidity_data.get('score', 0),
                'confidence': price_data.get('confidence', 'medium'),
                'explorer': f"https://solscan.io/token/{token['address']}",
                'score': 0
            }
            
            analysis.update(self.calculate_metrics(analysis))
            return analysis
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None

    def get_price_data(self, token):
        try:
            response = self.session.get(
                f"{JUPITER_PRICE_API}?ids={token['address']}&showExtraInfo=true",
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            price_data = data.get('data', {}).get(token['address'], {})
            return {
                'price': self.safe_float(price_data.get('price', 0)),
                'buy_price': self.safe_float(price_data.get('extraInfo', {}).get('quotedPrice', {}).get('buyPrice', 0)),
                'sell_price': self.safe_float(price_data.get('extraInfo', {}).get('quotedPrice', {}).get('sellPrice', 0)),
                'confidence': price_data.get('extraInfo', {}).get('confidenceLevel', 'medium'),
                'depth': price_data.get('extraInfo', {}).get('depth', {})
            }
        except:
            return {}

    def get_liquidity_data(self, token):
        try:
            decimals = token.get('decimals', 9)
            amount = int(1000 * (10 ** decimals))
            response = self.session.get(
                f"{JUPITER_QUOTE_API}?inputMint={token['address']}&outputMint={USDC_MINT}&amount={amount}",
                timeout=15
            )
            response.raise_for_status()
            quote = response.json()
            
            price_impact = self.safe_float(quote.get('priceImpactPct', 1))
            impacts = {
                '10': self.safe_float(quote.get('priceImpactPct', 1)),
                '100': self.safe_float(quote.get('priceImpactPct', 1)) * 10
            }
            
            return {
                'score': max(0.0, 100.0 - (price_impact * 10000)),
                'price_impact': impacts
            }
        except:
            return {'score': 0, 'price_impact': {}}

    def calculate_metrics(self, analysis):
        config = st.session_state.config
        metrics = {}
        
        # Price stability
        spread = abs(analysis['buy_price'] - analysis['sell_price'])
        metrics['price_stability'] = 1 - (spread / analysis['price']) if analysis['price'] > 0 else 0
        
        # Market depth
        impacts = [analysis['price_impact'].get(lvl, 1) for lvl in config['price_impact_levels']]
        metrics['market_depth'] = 1 - np.clip(np.mean(impacts), 0, 1)
        
        # Score calculation
        weights = config['score_weights']
        raw_score = (
            weights['liquidity'] * (analysis['liquidity'] / 100) +
            weights['stability'] * metrics['price_stability'] +
            weights['depth'] * metrics['market_depth']
        )
        
        # Confidence penalty
        penalty = 0.1 if analysis['confidence'] != 'high' else 0
        metrics['score'] = max(0, (raw_score - penalty) * 100)
        
        return metrics

def main():
    st.set_page_config(layout="wide", page_title="Advanced Token Analyzer")
    analyzer = EnhancedAnalyzer()
    
    # Custom CSS
    st.markdown("""
    <style>
        .settings-section { padding: 15px; border-radius: 10px; border: 1px solid #2d3b4e; }
        .metric-box { padding: 10px; background: #1a1a1a; border-radius: 5px; }
        @media (max-width: 768px) {
            .stButton>button { width: 100% !important; }
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.title("🔧 Advanced Token Analyzer")
        st.image("https://jup.ag/svg/jupiter-logo.svg", width=200)
        
        # Configuration Section
        with st.expander("⚙️ Analysis Settings", expanded=True):
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.config['batch_size'] = st.number_input(
                        "Batch Size", 1, 20, DEFAULT_BATCH_SIZE,
                        help="Tokens processed per iteration"
                    )
                with col2:
                    st.session_state.config['strict_mode'] = st.checkbox(
                        "Strict Validation", True,
                        help="Enable strict token filtering"
                    )
                
                st.session_state.config['show_advanced'] = st.checkbox(
                    "Show Advanced Settings", False
                )
                
                if st.session_state.config['show_advanced']:
                    with st.container():
                        st.subheader("Score Weights")
                        cols = st.columns(3)
                        with cols[0]:
                            st.session_state.config['score_weights']['liquidity'] = st.slider(
                                "Liquidity", 0.0, 1.0, 0.4, 0.05
                            )
                        with cols[1]:
                            st.session_state.config['score_weights']['stability'] = st.slider(
                                "Stability", 0.0, 1.0, 0.4, 0.05
                            )
                        with cols[2]:
                            st.session_state.config['score_weights']['depth'] = st.slider(
                                "Market Depth", 0.0, 1.0, 0.2, 0.05
                            )
                        
                        st.subheader("Price Impact Levels")
                        st.session_state.config['price_impact_levels'] = st.multiselect(
                            "Select Impact Levels",
                            ['10', '100', '500', '1000'],
                            default=['10', '100'],
                            help="Price impact levels to consider in analysis"
                        )
                        
                        st.subheader("Display Columns")
                        st.session_state.config['columns'] = st.multiselect(
                            "Select Columns to Display",
                            ['score', 'symbol', 'price', 'buy_price', 'sell_price', 
                             'liquidity', 'confidence', 'explorer', 'price_impact'],
                            default=st.session_state.config['columns'],
                            help="Select columns to show in results"
                        )

        # Action Buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Start", help="Begin analysis"):
                start_analysis(analyzer)
        with col2:
            if st.button("⏹ Stop", help="Halt analysis"):
                stop_analysis()
        with col3:
            if st.button("🧹 Clear", help="Reset results"):
                clear_results()

        # Progress Section
        if st.session_state.analysis['running']:
            show_progress()

        # Results Display
        if not st.session_state.analysis['results'].empty:
            st.divider()
            show_results()

    # Main processing logic
    if st.session_state.analysis['running']:
        process_batch(analyzer)

def start_analysis(analyzer):
    tokens = analyzer.get_token_list()
    if not tokens:
        st.error("No valid tokens found")
        return

    st.session_state.analysis.update({
        'running': True,
        'tokens': tokens,
        'results': pd.DataFrame(),
        'current_index': 0,
        'start_time': time.time(),
        'total_tokens': len(tokens)
    })
    st.rerun()

def process_batch(analyzer):
    analysis = st.session_state.analysis
    batch_size = st.session_state.config['batch_size']
    start_idx = analysis['current_index']
    end_idx = min(start_idx + batch_size, analysis['total_tokens'])

    for idx in range(start_idx, end_idx):
        token = analysis['tokens'][idx]
        result = analyzer.analyze_token(token)
        if result:
            new_df = pd.DataFrame([result])
            analysis['results'] = pd.concat([analysis['results'], new_df], ignore_index=True)
        analysis['current_index'] += 1

    if analysis['current_index'] >= analysis['total_tokens']:
        analysis['running'] = False
        if analysis['results'].empty:
            st.warning("No qualified tokens found")
    else:
        time.sleep(0.1)
        st.rerun()

def show_progress():
    analysis = st.session_state.analysis
    progress = analysis['current_index'] / analysis['total_tokens']
    
    st.progress(progress, text="Analysis Progress")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Processed", f"{analysis['current_index']}/{analysis['total_tokens']}")
    with cols[1]:
        elapsed = time.time() - analysis['start_time']
        speed = analysis['current_index'] / elapsed if elapsed > 0 else 0
        st.metric("Speed", f"{speed:.1f} tkn/s")
    with cols[2]:
        st.metric("Found", len(analysis['results']))

def show_results():
    df = st.session_state.analysis['results']
    config = st.session_state.config
    
    # Filter columns based on config
    columns = [c for c in config['columns'] if c in df.columns]
    if not columns:
        return
        
    sorted_df = df.sort_values('score', ascending=False)[columns]
    
    # Configure column display
    column_config = {
        'score': st.column_config.NumberColumn(
            'Score', format="%.1f", help="Composite quality score"
        ),
        'symbol': st.column_config.TextColumn('Token'),
        'price': st.column_config.NumberColumn(
            'Price', format="$%.4f", help="Current market price"
        ),
        'buy_price': st.column_config.NumberColumn(
            'Buy Price', format="$%.4f", help="Best buy price"
        ),
        'sell_price': st.column_config.NumberColumn(
            'Sell Price', format="$%.4f", help="Best sell price"
        ),
        'liquidity': st.column_config.ProgressColumn(
            'Liquidity', format="%.1f", min_value=0, max_value=100
        ),
        'confidence': st.column_config.SelectboxColumn(
            'Confidence', options=['low', 'medium', 'high']
        ),
        'explorer': st.column_config.LinkColumn('Explorer'),
        'price_impact': st.column_config.BarChartColumn(
            'Price Impact', y_min=0, y_max=100,
            help="Price impact at different trade sizes"
        )
    }

    st.dataframe(
        sorted_df,
        column_config=column_config,
        height=min(HEADER_HEIGHT + len(df) * ROW_HEIGHT, 600),
        use_container_width=True,
        hide_index=True
    )

def stop_analysis():
    st.session_state.analysis['running'] = False
    st.rerun()

def clear_results():
    st.session_state.analysis.update({
        'running': False,
        'tokens': [],
        'results': pd.DataFrame(),
        'current_index': 0,
        'start_time': None,
        'total_tokens': 0
    })
    st.rerun()

if __name__ == "__main__":
    main()
