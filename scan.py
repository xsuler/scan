import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
import os
import pickle
from solders.pubkey import Pubkey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置参数
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
STATE_FILE = "analysis_state.pkl"
DEFAULT_BATCH_SIZE = 5
ROW_HEIGHT = 35
HEADER_HEIGHT = 50

def load_state():
    """从文件加载保存的状态"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"加载状态失败: {str(e)}")
    return None

def save_state(state):
    """保存当前状态到文件"""
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(state, f)
    except Exception as e:
        print(f"保存状态失败: {str(e)}")

def clear_state():
    """清除保存的状态"""
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except Exception as e:
            print(f"清除状态失败: {str(e)}")

# 初始化或恢复状态
if 'analysis' not in st.session_state:
    saved_state = load_state()
    if saved_state:
        st.session_state.update(saved_state)
    else:
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
                'columns': ['score', 'symbol', 'price', 'liquidity', 'confidence', 'explorer']
            }
        })

class StatefulAnalyzer:
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
            st.error(f"Token获取失败: {str(e)}")
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
            st.error(f"分析失败: {str(e)}")
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
        
        spread = abs(analysis['buy_price'] - analysis['sell_price'])
        metrics['price_stability'] = 1 - (spread / analysis['price']) if analysis['price'] > 0 else 0
        
        impacts = [analysis['price_impact'].get(lvl, 1) for lvl in config['price_impact_levels']]
        metrics['market_depth'] = 1 - np.clip(np.mean(impacts), 0, 1)
        
        weights = config['score_weights']
        raw_score = (
            weights['liquidity'] * (analysis['liquidity'] / 100) +
            weights['stability'] * metrics['price_stability'] +
            weights['depth'] * metrics['market_depth']
        )
        
        penalty = 0.1 if analysis['confidence'] != 'high' else 0
        metrics['score'] = max(0, (raw_score - penalty) * 100)
        
        return metrics

def main():
    st.set_page_config(layout="wide", page_title="状态持久化代币分析仪")
    analyzer = StatefulAnalyzer()
    
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 100vw !important;
            min-width: 100vw !important;
            transform: translateX(0) !important;
            z-index: 999999;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        .main .block-container {
            padding-top: 0;
            position: relative;
            z-index: 1;
        }
        @media (max-width: 768px) {
            div.stButton > button {
                width: 100% !important;
                margin: 8px 0 !important;
            }
            div[data-testid="column"] {
                flex: 0 0 100% !important;
                width: 100% !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🔍 状态持久化分析仪")
        st.image("https://jup.ag/svg/jupiter-logo.svg", width=200)
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                start_btn = st.button("🚀 开始分析")
            with col2:
                stop_btn = st.button("⏹ 停止")
            with col3:
                clear_btn = st.button("🧹 清除")

        with st.expander("⚙️ 配置面板", expanded=True):
            st.session_state.config['batch_size'] = st.number_input(
                "批量处理数量", 1, 20, DEFAULT_BATCH_SIZE,
                help="每次处理代币数量"
            )
            st.session_state.config['strict_mode'] = st.checkbox(
                "严格验证模式", True,
                help="过滤社区代币和非主网代币"
            )

        with st.expander("🧪 高级设置", expanded=True):
            st.subheader("评分权重")
            cols = st.columns(3)
            with cols[0]:
                st.session_state.config['score_weights']['liquidity'] = st.slider(
                    "流动性", 0.0, 1.0, 0.4, 0.05
                )
            with cols[1]:
                st.session_state.config['score_weights']['stability'] = st.slider(
                    "稳定性", 0.0, 1.0, 0.4, 0.05
                )
            with cols[2]:
                st.session_state.config['score_weights']['depth'] = st.slider(
                    "市场深度", 0.0, 1.0, 0.2, 0.05
                )
            
            st.subheader("价格冲击等级")
            st.session_state.config['price_impact_levels'] = st.multiselect(
                "选择等级",
                ['10', '100', '500', '1000'],
                default=['10', '100']
            )
            
            st.subheader("显示选项")
            st.session_state.config['columns'] = st.multiselect(
                "显示列",
                ['score', 'symbol', 'price', 'buy_price', 'sell_price', 
                 'liquidity', 'confidence', 'explorer', 'price_impact'],
                default=st.session_state.config['columns']
            )

        if st.session_state.analysis['running']:
            show_progress()

        if not st.session_state.analysis['results'].empty:
            st.divider()
            show_results()

    # 处理按钮事件
    if start_btn:
        start_analysis(analyzer)
    if stop_btn:
        stop_analysis()
    if clear_btn:
        clear_analysis()

    # 自动继续处理
    if st.session_state.analysis['running']:
        process_batch(analyzer)
        save_state(dict(st.session_state))  # 实时保存状态

def start_analysis(analyzer):
    if not st.session_state.analysis['running']:
        tokens = analyzer.get_token_list()
        if not tokens:
            st.error("未找到有效代币")
            return

        st.session_state.analysis.update({
            'running': True,
            'tokens': tokens,
            'results': pd.DataFrame(),
            'current_index': 0,
            'start_time': time.time(),
            'total_tokens': len(tokens)
        })
        save_state(dict(st.session_state))
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
        clear_state()
        if analysis['results'].empty:
            st.warning("未找到合格代币")
    else:
        time.sleep(0.1)
        st.rerun()

def show_progress():
    analysis = st.session_state.analysis
    progress = analysis['current_index'] / analysis['total_tokens']
    
    st.progress(progress, text="分析进度")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("已处理", f"{analysis['current_index']}/{analysis['total_tokens']}")
    with cols[1]:
        elapsed = time.time() - analysis['start_time']
        speed = analysis['current_index'] / elapsed if elapsed > 0 else 0
        st.metric("速度", f"{speed:.1f} tkn/s")
    with cols[2]:
        st.metric("发现数", len(analysis['results']))

def show_results():
    df = st.session_state.analysis['results']
    config = st.session_state.config
    
    column_config = {
        'score': st.column_config.NumberColumn('评分', format="%.1f"),
        'symbol': st.column_config.TextColumn('代币'),
        'price': st.column_config.NumberColumn('价格', format="$%.4f"),
        'buy_price': st.column_config.NumberColumn('买价', format="$%.4f"),
        'sell_price': st.column_config.NumberColumn('卖价', format="$%.4f"),
        'liquidity': st.column_config.ProgressColumn('流动性', format="%.1f", min_value=0, max_value=100),
        'confidence': st.column_config.SelectboxColumn('信心', options=['low', 'medium', 'high']),
        'explorer': st.column_config.LinkColumn('浏览器'),
        'price_impact': st.column_config.BarChartColumn('价格冲击', y_min=0, y_max=100)
    }

    st.dataframe(
        df[config['columns']],
        column_config=column_config,
        height=min(HEADER_HEIGHT + len(df) * ROW_HEIGHT, 600),
        use_container_width=True,
        hide_index=True
    )

def stop_analysis():
    if st.session_state.analysis['running']:
        st.session_state.analysis['running'] = False
        clear_state()
        st.rerun()

def clear_analysis():
    st.session_state.analysis.update({
        'running': False,
        'tokens': [],
        'results': pd.DataFrame(),
        'current_index': 0,
        'start_time': None,
        'total_tokens': 0
    })
    clear_state()
    st.rerun()

if __name__ == "__main__":
    main()
