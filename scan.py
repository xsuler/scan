import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
import os
import json
from pathlib import Path
from solders.pubkey import Pubkey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置参数
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
CHECKPOINT_DIR = Path("./checkpoints")
DEFAULT_BATCH_SIZE = 5
ROW_HEIGHT = 35
HEADER_HEIGHT = 50

# 创建检查点目录
CHECKPOINT_DIR.mkdir(exist_ok=True)

class CheckpointManager:
    @staticmethod
    def get_checkpoint_files():
        return {
            "tokens": CHECKPOINT_DIR / "tokens.pkl",
            "results": CHECKPOINT_DIR / "results.parquet",
            "progress": CHECKPOINT_DIR / "progress.json",
            "config": CHECKPOINT_DIR / "config.json"
        }

    @staticmethod
    def save_checkpoint(tokens, results, progress, config):
        files = CheckpointManager.get_checkpoint_files()
        try:
            # 保存代币列表
            tokens.to_pickle(files["tokens"])
            
            # 追加结果到Parquet文件
            if not results.empty:
                if files["results"].exists():
                    existing = pd.read_parquet(files["results"])
                    results = pd.concat([existing, results])
                results.to_parquet(files["results"])
            
            # 保存处理进度
            with open(files["progress"], "w") as f:
                json.dump(progress, f)
            
            # 保存配置
            with open(files["config"], "w") as f:
                json.dump(config, f)
            
            return True
        except Exception as e:
            st.error(f"保存检查点失败: {str(e)}")
            return False

    @staticmethod
    def load_checkpoint():
        files = CheckpointManager.get_checkpoint_files()
        if not all(f.exists() for f in files.values()):
            return None

        try:
            tokens = pd.read_pickle(files["tokens"])
            results = pd.read_parquet(files["results"])
            with open(files["progress"], "r") as f:
                progress = json.load(f)
            with open(files["config"], "r") as f:
                config = json.load(f)
            return {
                "tokens": tokens,
                "results": results,
                "progress": progress,
                "config": config
            }
        except Exception as e:
            st.error(f"加载检查点失败: {str(e)}")
            return None

    @staticmethod
    def clear_checkpoints():
        files = CheckpointManager.get_checkpoint_files()
        for f in files.values():
            if f.exists():
                try:
                    f.unlink()
                except Exception as e:
                    st.error(f"删除文件{f.name}失败: {str(e)}")

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

    def validate_token(self, token, strict_mode):
        try:
            if not Pubkey.from_string(token.get('address', '')):
                return False

            required_fields = ['symbol', 'name', 'decimals', 'logoURI']
            for field in required_fields:
                if not token.get(field):
                    return False

            if strict_mode:
                if "community" in token.get('tags', []):
                    return False
                if token.get('chainId') != 101:
                    return False

            return True
        except:
            return False

    def get_token_list(self, strict_mode):
        try:
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
            response.raise_for_status()
            return [t for t in response.json() if self.validate_token(t, strict_mode)]
        except Exception as e:
            st.error(f"Token获取失败: {str(e)}")
            return None

    def analyze_batch(self, tokens):
        results = []
        for token in tokens:
            try:
                analysis = self._analyze_token(token)
                if analysis:
                    results.append(analysis)
            except Exception as e:
                st.error(f"分析代币失败: {str(e)}")
        return pd.DataFrame(results)

    def _analyze_token(self, token):
        price_data = self.get_price_data(token)
        liquidity_data = self.get_liquidity_data(token)
        
        analysis = {
            'symbol': token.get('symbol', 'Unknown'),
            'address': token['address'],
            'price': self.safe_float(price_data.get('price', 0)),
            'buy_price': self.safe_float(price_data.get('buy_price', 0)),
            'sell_price': self.safe_float(price_data.get('sell_price', 0)),
            'liquidity': liquidity_data.get('score', 0),
            'confidence': price_data.get('confidence', 'medium'),
            'explorer': f"https://solscan.io/token/{token['address']}",
            'score': self.calculate_score(price_data, liquidity_data)
        }
        return analysis

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
                'confidence': price_data.get('extraInfo', {}).get('confidenceLevel', 'medium')
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
            return {
                'score': max(0.0, 100.0 - (price_impact * 10000))
            }
        except:
            return {'score': 0}

    def calculate_score(self, price_data, liquidity_data):
        try:
            liquidity = liquidity_data['score'] / 100
            spread = abs(price_data['buy_price'] - price_data['sell_price'])
            stability = 1 - (spread / price_data['price']) if price_data['price'] > 0 else 0
            confidence = 0.9 if price_data['confidence'] == 'high' else 0.7
            return (liquidity * 0.4 + stability * 0.4 + confidence * 0.2) * 100
        except:
            return 0

def main():
    st.set_page_config(layout="wide", page_title="带检查点的代币分析仪")
    analyzer = EnhancedAnalyzer()
    
    # 初始化session状态
    if 'analysis' not in st.session_state:
        checkpoint = CheckpointManager.load_checkpoint()
        if checkpoint:
            st.session_state.update({
                'analysis': {
                    'running': True,
                    'tokens': checkpoint['tokens'],
                    'results': checkpoint['results'],
                    'current_index': checkpoint['progress']['current_index'],
                    'start_time': checkpoint['progress']['start_time'],
                    'total_tokens': checkpoint['progress']['total_tokens']
                },
                'config': checkpoint['config']
            })
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
                    'columns': ['score', 'symbol', 'price', 'liquidity', 'confidence', 'explorer']
                }
            })

    # 侧边栏界面
    with st.sidebar:
        st.title("🔍 带检查点的分析仪")
        st.image("https://jup.ag/svg/jupiter-logo.svg", width=200)
        
        # 控制面板
        with st.expander("⚙️ 控制面板", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                start_btn = st.button("🚀 开始")
            with col2:
                stop_btn = st.button("⏹ 停止")
            with col3:
                clear_btn = st.button("🧹 清除")

            st.session_state.config['batch_size'] = st.number_input(
                "批量大小", 1, 20, DEFAULT_BATCH_SIZE
            )
            st.session_state.config['strict_mode'] = st.checkbox(
                "严格模式", True
            )

        # 显示进度
        if st.session_state.analysis['running']:
            show_progress()

        # 显示结果
        if not st.session_state.analysis['results'].empty:
            st.divider()
            show_results()

    # 处理按钮事件
    if start_btn and not st.session_state.analysis['running']:
        start_analysis(analyzer)
        
    if stop_btn and st.session_state.analysis['running']:
        stop_analysis()
        
    if clear_btn:
        clear_analysis()

    # 处理批次
    if st.session_state.analysis['running']:
        process_batch(analyzer)

def start_analysis(analyzer):
    tokens = analyzer.get_token_list(st.session_state.config['strict_mode'])
    if not tokens:
        st.error("未获取到有效代币")
        return
    
    st.session_state.analysis.update({
        'running': True,
        'tokens': tokens,
        'results': pd.DataFrame(),
        'current_index': 0,
        'start_time': time.time(),
        'total_tokens': len(tokens)
    })
    save_checkpoint()

def process_batch(analyzer):
    analysis = st.session_state.analysis
    batch_size = st.session_state.config['batch_size']
    
    start_idx = analysis['current_index']
    end_idx = min(start_idx + batch_size, analysis['total_tokens'])
    batch_tokens = analysis['tokens'][start_idx:end_idx]
    
    # 分析当前批次
    batch_results = analyzer.analyze_batch(batch_tokens)
    
    # 更新状态
    if not batch_results.empty:
        analysis['results'] = pd.concat([analysis['results'], batch_results], ignore_index=True)
    analysis['current_index'] = end_idx
    
    # 保存检查点
    save_checkpoint()
    
    # 检查是否完成
    if analysis['current_index'] >= analysis['total_tokens']:
        analysis['running'] = False
        CheckpointManager.clear_checkpoints()
        st.rerun()
    else:
        time.sleep(0.1)
        st.rerun()

def save_checkpoint():
    progress = {
        'current_index': st.session_state.analysis['current_index'],
        'start_time': st.session_state.analysis['start_time'],
        'total_tokens': st.session_state.analysis['total_tokens']
    }
    CheckpointManager.save_checkpoint(
        pd.Series(st.session_state.analysis['tokens']),
        st.session_state.analysis['results'],
        progress,
        st.session_state.config
    )

def show_progress():
    analysis = st.session_state.analysis
    progress = analysis['current_index'] / analysis['total_tokens']
    
    st.progress(progress)
    
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
    columns = st.session_state.config['columns']
    
    column_config = {
        'score': st.column_config.NumberColumn('评分', format="%.1f"),
        'symbol': st.column_config.TextColumn('代币'),
        'price': st.column_config.NumberColumn('价格', format="$%.4f"),
        'liquidity': st.column_config.ProgressColumn('流动性', format="%.1f", min_value=0, max_value=100),
        'confidence': st.column_config.SelectboxColumn('信心', options=['low', 'medium', 'high']),
        'explorer': st.column_config.LinkColumn('浏览器')
    }
    
    st.dataframe(
        df[columns],
        column_config=column_config,
        height=min(HEADER_HEIGHT + len(df) * ROW_HEIGHT, 600),
        use_container_width=True,
        hide_index=True
    )

def stop_analysis():
    if st.session_state.analysis['running']:
        st.session_state.analysis['running'] = False
        save_checkpoint()
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
    CheckpointManager.clear_checkpoints()
    st.rerun()

if __name__ == "__main__":
    main()
