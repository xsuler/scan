import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
import os
import pickle
from pathlib import Path
from solders.pubkey import Pubkey
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置常量
JUPITER_TOKEN_LIST = "https://token.jup.ag/all"
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_PRICE_API = "https://api.jup.ag/price/v2"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
CHECKPOINT_FILE = "checkpoint.pkl"
ROW_HEIGHT = 35
HEADER_HEIGHT = 50
DEFAULT_BATCH_SIZE = 1

# 初始化检查点目录
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

class AnalysisState:
    @staticmethod
    def save_state(state):
        try:
            with open(CHECKPOINT_DIR / CHECKPOINT_FILE, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            st.error(f"状态保存失败: {str(e)}")

    @staticmethod
    def load_state():
        file_path = CHECKPOINT_DIR / CHECKPOINT_FILE
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"状态加载失败: {str(e)}")
        return None

    @staticmethod
    def clear_state():
        try:
            (CHECKPOINT_DIR / CHECKPOINT_FILE).unlink(missing_ok=True)
        except Exception as e:
            st.error(f"状态清除失败: {str(e)}")

class TokenAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self._configure_session()
        self.debug_log = []

    def _configure_session(self):
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def validate_token(self, token, excluded_tags):
        validation_result = {
            "valid": False,
            "reasons": [],
            "symbol": token.get("symbol", "UNKNOWN"),
            "address": token.get("address", "")
        }

        try:
            if not Pubkey.from_string(token.get("address", "")):
                validation_result["reasons"].append("Invalid address")
                return False

            required_fields = ["symbol", "name", "decimals", "logoURI"]
            for field in required_fields:
                if not token.get(field):
                    validation_result["reasons"].append(f"Missing {field}")
                    return False

            token_tags = set(token.get("tags", []))
            excluded = set(excluded_tags)
            if token_tags & excluded:
                validation_result["reasons"].append(f"Excluded tags: {', '.join(token_tags & excluded)}")
                return False

            if token.get("chainId") != 101:
                validation_result["reasons"].append("Non-mainnet token")
                return False

            validation_result["valid"] = True
            return True
        except Exception as e:
            validation_result["reasons"].append(f"Validation error: {str(e)}")
            return False
        finally:
            self.debug_log.append(validation_result)

    def fetch_token_list(self, excluded_tags):
        try:
            response = self.session.get(JUPITER_TOKEN_LIST, timeout=15)
            response.raise_for_status()
            tokens = response.json()
            return [t for t in tokens if self.validate_token(t, excluded_tags)]
        except Exception as e:
            st.error(f"代币获取失败: {str(e)}")
            return []

    def analyze_token(self, token):
        try:
            price_data = self._get_price_data(token)
            liquidity_score = self._calculate_liquidity(token)
            
            analysis = {
                "symbol": token.get("symbol", "UNKNOWN"),
                "address": token["address"],
                "price": price_data["price"],
                "buy_price": price_data["buy_price"],
                "sell_price": price_data["sell_price"],
                "price_impact_10k": price_data["impact_10k"],
                "price_impact_100k": price_data["impact_100k"],
                "liquidity_score": liquidity_score,
                "confidence": price_data["confidence"],
                "explorer": f"https://solscan.io/token/{token['address']}",
                "score": 0
            }
            
            analysis["score"] = self._calculate_score(analysis)
            return analysis
        except Exception as e:
            st.error(f"分析代币失败: {str(e)}")
            return None

    def _get_price_data(self, token):
        try:
            response = self.session.get(
                f"{JUPITER_PRICE_API}?ids={token['address']}&showExtraInfo=true",
                timeout=15
            )
            response.raise_for_status()
            data = response.json().get("data", {}).get(token["address"], {})
            
            return {
                "price": self._safe_float(data.get("price", 0)),
                "buy_price": self._safe_float(data.get("extraInfo", {}).get("quotedPrice", {}).get("buyPrice", 0)),
                "sell_price": self._safe_float(data.get("extraInfo", {}).get("quotedPrice", {}).get("sellPrice", 0)),
                "impact_10k": self._safe_float(data.get("extraInfo", {}).get("depth", {}).get("sellPriceImpactRatio", {}).get("depth", {}).get("10", 0)),
                "impact_100k": self._safe_float(data.get("extraInfo", {}).get("depth", {}).get("sellPriceImpactRatio", {}).get("depth", {}).get("100", 0)),
                "confidence": data.get("extraInfo", {}).get("confidenceLevel", "medium")
            }
        except Exception as e:
            st.error(f"价格数据获取失败: {str(e)}")
            return {}

    def _calculate_liquidity(self, token):
        try:
            decimals = token.get("decimals", 9)
            amount = int(1000 * (10 ** decimals))
            response = self.session.get(
                f"{JUPITER_QUOTE_API}?inputMint={token['address']}&outputMint={USDC_MINT}&amount={amount}",
                timeout=15
            )
            response.raise_for_status()
            quote = response.json()
            price_impact = self._safe_float(quote.get("priceImpactPct", 1))
            return max(0.0, 100.0 - (price_impact * 10000))
        except Exception as e:
            st.error(f"流动性计算失败: {str(e)}")
            return 0.0

    def _calculate_score(self, analysis):
        try:
            weights = {
                "liquidity": 0.4,
                "price_stability": 0.35,
                "market_depth": 0.25
            }
            
            spread = abs(analysis["buy_price"] - analysis["sell_price"])
            price_stability = 1 - (spread / analysis["price"]) if analysis["price"] > 0 else 0
            
            market_depth = 1 - (analysis["price_impact_10k"] + analysis["price_impact_100k"]) / 2
            
            base_score = (
                weights["liquidity"] * (analysis["liquidity_score"] / 100) +
                weights["price_stability"] * price_stability +
                weights["market_depth"] * market_depth
            )
            
            confidence_factor = 1.0 if analysis["confidence"] == "high" else 0.8
            return max(0, min(100, base_score * 100 * confidence_factor))
        except Exception as e:
            st.error(f"评分计算失败: {str(e)}")
            return 0

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value) if value not in [None, np.nan, ""] else default
        except:
            return default

def initialize_session():
    if "analysis" not in st.session_state:
        saved_state = AnalysisState.load_state()
        if saved_state:
            st.session_state.update(saved_state)
        else:
            st.session_state.update({
                "analysis": {
                    "running": False,
                    "tokens": [],
                    "results": pd.DataFrame(),
                    "current_index": 0,
                    "start_time": None,
                    "total_tokens": 0,
                    "current_token": None
                },
                "config": {
                    "excluded_tags": ["community"],
                    "batch_size": DEFAULT_BATCH_SIZE,
                    "live_sort": True,
                    "columns": ["score", "symbol", "price", "liquidity_score", "confidence", "explorer"]
                }
            })

def setup_ui():
    st.set_page_config(layout="wide", page_title="Solana代币分析仪")
    
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
        .metric-box {
            padding: 10px;
            background: #1a1a1a;
            border-radius: 5px;
            margin: 5px 0;
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

def render_sidebar(analyzer):
    with st.sidebar:
        st.title("🔍 Solana代币分析仪")
        st.image("https://jup.ag/svg/jupiter-logo.svg", width=200)

        with st.expander("⚙️ 控制面板", expanded=True):
            cols = st.columns(3)
            with cols[0]:
                start_btn = st.button("🚀 开始分析")
            with cols[1]:
                stop_btn = st.button("⏹ 停止分析")
            with cols[2]:
                clear_btn = st.button("🧹 清除结果")

            st.session_state.config["excluded_tags"] = st.multiselect(
                "排除标签",
                options=["community", "old-registry", "unknown"],
                default=st.session_state.config["excluded_tags"]
            )

        with st.expander("📊 显示选项", expanded=True):
            st.session_state.config["live_sort"] = st.checkbox(
                "实时排序结果", True,
                help="根据评分自动排序"
            )
            st.session_state.config["columns"] = st.multiselect(
                "显示列",
                ["score", "symbol", "price", "buy_price", "sell_price",
                 "liquidity_score", "confidence", "explorer", "price_impact_10k", "price_impact_100k"],
                default=st.session_state.config["columns"]
            )

        if st.session_state.analysis["running"]:
            render_progress()

        if st.session_state.analysis.get("current_token"):
            render_current_token()

        if not st.session_state.analysis["results"].empty:
            st.divider()
            render_results()

    handle_actions(start_btn, stop_btn, clear_btn, analyzer)

def render_progress():
    analysis = st.session_state.analysis
    progress = analysis["current_index"] / analysis["total_tokens"]
    
    st.progress(progress, text="分析进度")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("已处理", f"{analysis['current_index']}/{analysis['total_tokens']}")
    with cols[1]:
        elapsed = time.time() - analysis["start_time"]
        speed = analysis["current_index"] / elapsed if elapsed > 0 else 0
        st.metric("处理速度", f"{speed:.1f} tkn/s")
    with cols[2]:
        st.metric("发现数量", len(analysis["results"]))

def render_current_token():
    with st.expander("🔍 当前代币", expanded=True):
        token = st.session_state.analysis["current_token"]
        st.write(f"**代币符号**: {token.get('symbol', '未知')}")
        st.write(f"**合约地址**: `{token['address'][:12]}...`")
        st.write(f"**链ID**: {token.get('chainId', '未知')}")
        st.write(f"**标签**: {', '.join(token.get('tags', []))}")

def render_results():
    df = st.session_state.analysis["results"]
    config = st.session_state.config
    
    if config["live_sort"]:
        df = df.sort_values("score", ascending=False)
    
    column_config = {
        "score": st.column_config.NumberColumn("评分", format="%.1f"),
        "symbol": st.column_config.TextColumn("代币"),
        "price": st.column_config.NumberColumn("价格", format="$%.4f"),
        "buy_price": st.column_config.NumberColumn("买价", format="$%.4f"),
        "sell_price": st.column_config.NumberColumn("卖价", format="$%.4f"),
        "liquidity_score": st.column_config.ProgressColumn("流动性", format="%.1f", min_value=0, max_value=100),
        "confidence": st.column_config.SelectboxColumn("信心等级", options=["low", "medium", "high"]),
        "explorer": st.column_config.LinkColumn("区块链浏览器"),
        "price_impact_10k": st.column_config.NumberColumn("10K冲击", format="%.2f%%"),
        "price_impact_100k": st.column_config.NumberColumn("100K冲击", format="%.2f%%")
    }

    st.dataframe(
        df[config["columns"]],
        column_config=column_config,
        height=min(HEADER_HEIGHT + len(df) * ROW_HEIGHT, 600),
        use_container_width=True,
        hide_index=True
    )

def handle_actions(start_btn, stop_btn, clear_btn, analyzer):
    if start_btn and not st.session_state.analysis["running"]:
        start_analysis(analyzer)
        
    if stop_btn and st.session_state.analysis["running"]:
        stop_analysis()
        
    if clear_btn:
        clear_analysis()

def start_analysis(analyzer):
    tokens = analyzer.fetch_token_list(st.session_state.config["excluded_tags"])
    if not tokens:
        st.error("未找到有效代币")
        return

    st.session_state.analysis.update({
        "running": True,
        "tokens": tokens,
        "results": pd.DataFrame(),
        "current_index": 0,
        "start_time": time.time(),
        "total_tokens": len(tokens),
        "current_token": None
    })
    process_batch(analyzer)

def process_batch(analyzer):
    analysis = st.session_state.analysis
    if not analysis["running"]:
        return

    batch_size = st.session_state.config["batch_size"]
    start_idx = analysis["current_index"]
    end_idx = min(start_idx + batch_size, analysis["total_tokens"])

    for idx in range(start_idx, end_idx):
        analysis["current_token"] = analysis["tokens"][idx]
        result = analyzer.analyze_token(analysis["current_token"])
        
        if result:
            new_df = pd.DataFrame([result])
            analysis["results"] = pd.concat([analysis["results"], new_df], ignore_index=True)
        
        analysis["current_index"] += 1
        time.sleep(0.1)
        
        AnalysisState.save_state(dict(st.session_state))

    if analysis["current_index"] >= analysis["total_tokens"]:
        analysis["running"] = False
        AnalysisState.clear_state()
    else:
        st.rerun()

def stop_analysis():
    if st.session_state.analysis["running"]:
        st.session_state.analysis["running"] = False
        AnalysisState.save_state(dict(st.session_state))
        st.rerun()

def clear_analysis():
    st.session_state.analysis.update({
        "running": False,
        "tokens": [],
        "results": pd.DataFrame(),
        "current_index": 0,
        "start_time": None,
        "total_tokens": 0,
        "current_token": None
    })
    AnalysisState.clear_state()
    st.rerun()

if __name__ == "__main__":
    initialize_session()
    setup_ui()
    analyzer = TokenAnalyzer()
    render_sidebar(analyzer)
    if st.session_state.analysis["running"]:
        process_batch(analyzer)
