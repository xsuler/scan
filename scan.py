import streamlit as st
import pandas as pd
import asyncio
import time
import pickle
import os
from pathlib import Path
from bsc_analyzer import BSCQuantAnalyzer

# 配置常量
CHECKPOINT_DIR = Path("./bsc_checkpoints")
CHECKPOINT_FILE = "analysis_state.pkl"
ROW_HEIGHT = 35
HEADER_HEIGHT = 50
DEFAULT_BATCH_SIZE = 5

class AnalysisState:
    @staticmethod
    def save_state(state):
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        try:
            with open(CHECKPOINT_DIR / CHECKPOINT_FILE, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            st.error(f"状态保存失败: {str(e)}")

    @staticmethod
    def load_state():
        file = CHECKPOINT_DIR / CHECKPOINT_FILE
        if file.exists():
            try:
                with open(file, "rb") as f:
                    return pickle.load(f)
            except:
                pass
        return None

    @staticmethod
    def clear_state():
        try:
            (CHECKPOINT_DIR / CHECKPOINT_FILE).unlink(missing_ok=True)
        except:
            pass

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
                    "excluded_tokens": [],
                    "batch_size": DEFAULT_BATCH_SIZE,
                    "live_sort": True,
                    "columns": ["score", "symbol", "liquidity", "volatility", "holders_gini", "age_days", "explorer"]
                }
            })

def setup_ui():
    st.set_page_config(layout="wide", page_title="BSC代币分析仪")
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 320px !important;
        }
        .stDataFrame { 
            width: 100% !important; 
        }
        .metric-box {
            padding: 10px;
            background: #1a1a1a;
            border-radius: 5px;
            margin: 5px 0;
        }
        @media (max-width: 768px) {
            section[data-testid="stSidebar"] {
                width: 100% !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar(analyzer):
    with st.sidebar:
        st.title("🔍 BSC代币分析仪")
        st.image("https://bscscan.com/images/svg/brands/bsc.svg", width=120)

        with st.expander("⚙️ 控制面板", expanded=True):
            cols = st.columns(2)
            with cols[0]: start_btn = st.button("🚀 开始分析")
            with cols[1]: stop_btn = st.button("⏹ 停止分析")
            
            st.session_state.config["batch_size"] = st.slider(
                "批量大小", 1, 20, DEFAULT_BATCH_SIZE,
                help="每次分析代币数量"
            )
            
            if st.button("🧹 清除结果"):
                st.session_state.analysis["results"] = pd.DataFrame()
                AnalysisState.clear_state()
                st.rerun()

        if st.session_state.analysis["running"]:
            render_progress()

        if not st.session_state.analysis["results"].empty:
            st.divider()
            render_results()

    handle_actions(start_btn, stop_btn, analyzer)

def render_progress():
    analysis = st.session_state.analysis
    progress = analysis["current_index"] / analysis["total_tokens"] if analysis["total_tokens"] > 0 else 0
    
    st.progress(progress, text="分析进度")
    cols = st.columns(3)
    with cols[0]: st.metric("已处理", f"{analysis['current_index']}/{analysis['total_tokens']}")
    with cols[1]: st.metric("运行时间", f"{time.time()-analysis['start_time']:.1f}s")
    with cols[2]: st.metric("发现数量", len(analysis["results"]))

    if st.session_state.analysis.get("current_token"):
        with st.expander("当前分析代币", expanded=False):
            token = st.session_state.analysis["current_token"]
            st.write(f"**合约地址**: `{token[:12]}...`")
            st.write(f"**进度**: {analysis['current_index'] / analysis['total_tokens']:.1%}")

def render_results():
    df = st.session_state.analysis["results"]
    config = st.session_state.config
    
    if config["live_sort"] and not df.empty:
        df = df.sort_values("score", ascending=False)
    
    column_config = {
        "score": st.column_config.NumberColumn("评分", format="%.1f", width="small"),
        "symbol": st.column_config.TextColumn("代币", width="small"),
        "liquidity": st.column_config.NumberColumn("流动性(USD)", format="%.2f", width="medium"),
        "volatility": st.column_config.NumberColumn("波动率", format="%.2%", width="small"),
        "holders_gini": st.column_config.NumberColumn("持币基尼系数", format="%.2f", width="medium"),
        "age_days": st.column_config.NumberColumn("流通天数", format="%.0f", width="small"),
        "explorer": st.column_config.LinkColumn("区块链浏览器", display_text="查看", width="medium")
    }

    st.dataframe(
        df[config["columns"]],
        column_config=column_config,
        height=min(HEADER_HEIGHT + len(df)*ROW_HEIGHT, 600),
        use_container_width=True,
        hide_index=True
    )

def handle_actions(start_btn, stop_btn, analyzer):
    if start_btn and not st.session_state.analysis["running"]:
        asyncio.run(start_analysis(analyzer))
    if stop_btn and st.session_state.analysis["running"]:
        stop_analysis()

async def start_analysis(analyzer):
    st.session_state.analysis.update({
        "running": True,
        "start_time": time.time(),
        "current_index": 0,
        "results": pd.DataFrame(),
        "total_tokens": 0,
        "tokens": []
    })
    
    try:
        tokens = await analyzer.get_token_list()
        valid_tokens = []
        for token in tokens:
            if await analyzer.validate_token(token):
                valid_tokens.append(token)
        
        st.session_state.analysis.update({
            "tokens": valid_tokens,
            "total_tokens": len(valid_tokens)
        })
        
        await process_batch(analyzer)
    except Exception as e:
        st.error(f"分析启动失败: {str(e)}")
    finally:
        st.session_state.analysis["running"] = False
        AnalysisState.save_state(dict(st.session_state))
        st.rerun()

async def process_batch(analyzer):
    analysis = st.session_state.analysis
    batch_size = st.session_state.config["batch_size"]
    
    while analysis["running"] and analysis["current_index"] < analysis["total_tokens"]:
        batch = analysis["tokens"][analysis["current_index"] : analysis["current_index"]+batch_size]
        results = []
        
        for token in batch:
            try:
                st.session_state.analysis["current_token"] = token
                result = await analyzer.analyze_token(token)
                results.append(result)
                analysis["current_index"] += 1
            except Exception as e:
                st.error(f"分析代币 {token} 失败: {str(e)}")
        
        if results:
            new_df = pd.DataFrame(results)
            analysis["results"] = pd.concat([analysis["results"], new_df], ignore_index=True)
            AnalysisState.save_state(dict(st.session_state))
        
        st.rerun()
        await asyncio.sleep(0.5)
    
    analysis["running"] = False
    AnalysisState.save_state(dict(st.session_state))

def stop_analysis():
    if st.session_state.analysis["running"]:
        st.session_state.analysis["running"] = False
        AnalysisState.save_state(dict(st.session_state))
        st.rerun()

if __name__ == "__main__":
    initialize_session()
    setup_ui()
    analyzer = BSCQuantAnalyzer()
    render_sidebar(analyzer)
    st.markdown("""
    <div style="margin-left: 320px; padding: 20px;">
        <h2>分析结果展示区</h2>
        <p>选择左侧控制面板开始分析BSC链上代币</p>
    </div>
    """, unsafe_allow_html=True)
