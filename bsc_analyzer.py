import numpy as np
import pandas as pd
from web3 import Web3
from web3.middleware import geth_poa_middleware
import asyncio
from collections import deque
from typing import Dict, List

class BSCQuantAnalyzer:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider('https://bsc-dataseed.binance.org/'))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.dex_router = '0x10ED43C718714eb63d5aA57B78B54704E256024E'  # PancakeSwap Router
        self.cached_pairs = deque(maxlen=1000)

    async def get_token_list(self) -> List[str]:
        """获取BSC链上前1000个交易对的代币列表"""
        factory_abi = [
            {"inputs":[],"name":"allPairsLength","outputs":[{"name":"","type":"uint256"}],"type":"function"},
            {"inputs":[{"name":"","type":"uint256"}],"name":"allPairs","outputs":[{"name":"","type":"address"}],"type":"function"}
        ]
        
        try:
            router = self.w3.eth.contract(address=self.dex_router, abi=self._router_abi())
            factory_addr = router.functions.factory().call()
            factory = self.w3.eth.contract(address=factory_addr, abi=factory_abi)
            
            pair_count = min(factory.functions.allPairsLength().call(), 1000)
            tokens = set()
            
            for i in range(pair_count):
                pair = factory.functions.allPairs(i).call()
                pair_contract = self.w3.eth.contract(address=pair, abi=self._pair_abi())
                tokens.update([pair_contract.functions.token0().call(),
                             pair_contract.functions.token1().call()])
                self.cached_pairs.append(pair)
                
            return list(tokens)
        except Exception as e:
            print(f"获取代币列表失败: {e}")
            return []

    async def validate_token(self, token_address: str) -> bool:
        """验证BEP20代币有效性"""
        try:
            contract = self.w3.eth.contract(address=token_address, abi=self._bep20_abi())
            return all([
                bool(contract.functions.symbol().call()),
                contract.functions.decimals().call() <= 18,
                contract.functions.totalSupply().call() > 0
            ])
        except:
            return False

    async def analyze_token(self, token_address: str) -> Dict:
        """执行完整代币分析"""
        dex_data = await self._get_dex_metrics(token_address)
        holder_metrics = await self._get_holder_metrics(token_address)
        age = await self._get_token_age(token_address)
        tech_indicators = self._calculate_ta(dex_data.get('prices', [1]))
        
        return {
            'score': self._calculate_score(dex_data, holder_metrics, age, tech_indicators),
            'symbol': await self._get_symbol(token_address),
            'liquidity': dex_data.get('liquidity', 0),
            'volatility': self._calc_volatility(dex_data.get('prices', [])),
            'price_impact_10k': dex_data.get('impact_10k', 1),
            'price_impact_100k': dex_data.get('impact_100k', 1),
            'holders_gini': holder_metrics.get('gini', 1),
            'holders_top10': holder_metrics.get('top10', 1),
            'age_days': age * 365,
            'contract': token_address,
            'explorer': f"https://bscscan.com/token/{token_address}"
        }

    async def _get_dex_metrics(self, token_address: str) -> Dict:
        """获取去中心化交易所指标"""
        try:
            pair = await self._find_main_pair(token_address)
            if not pair:
                return {'liquidity':0, 'prices':[], 'impact_10k':1, 'impact_100k':1}
            
            pair_contract = self.w3.eth.contract(address=pair, abi=self._pair_abi())
            reserves = pair_contract.functions.getReserves().call()
            token0 = pair_contract.functions.token0().call()
            
            # 获取代币精度
            token0_contract = self.w3.eth.contract(address=token0, abi=self._bep20_abi())
            decimals0 = token0_contract.functions.decimals().call()
            decimals1 = 18  # 假设配对的是BNB
            
            price = reserves[0] / (reserves[1] + 1e-9) * 10**(decimals1 - decimals0) if token0 == token_address \
                   else reserves[1] / (reserves[0] + 1e-9) * 10**(decimals0 - decimals1)
            
            return {
                'liquidity': (reserves[0] * 10**-decimals0 * price) + (reserves[1] * 10**-decimals1),
                'prices': [price],
                'impact_10k': self._calc_price_impact(pair_contract, 10000, token_address),
                'impact_100k': self._calc_price_impact(pair_contract, 100000, token_address)
            }
        except Exception as e:
            print(f"DEX数据获取失败: {e}")
            return {'liquidity':0, 'prices':[], 'impact_10k':1, 'impact_100k':1}

    async def _get_holder_metrics(self, token_address: str) -> Dict:
        """持币分布分析"""
        try:
            contract = self.w3.eth.contract(address=token_address, abi=self._bep20_abi())
            transfers = []
            
            # 扫描最近50个区块提高效率
            latest_block = self.w3.eth.block_number
            for bn in range(latest_block-50, latest_block):
                block = self.w3.eth.get_block(bn, full_transactions=True)
                for tx in block.transactions:
                    if tx['to'] == token_address:
                        transfers.append(tx['from'])
            
            # 获取持币余额（前500个地址）
            holders = list(set(transfers))[:500]
            balances = [contract.functions.balanceOf(h).call() for h in holders]
            balances = [b/10**contract.functions.decimals().call() for b in balances if b > 0]
            
            if not balances:
                return {'gini': 1, 'top10': 1}
            
            sorted_balances = np.sort(balances)
            total = sum(sorted_balances)
            cum_balances = np.cumsum(sorted_balances)
            
            # 计算基尼系数
            n = len(sorted_balances)
            gini = (np.sum((2 * np.arange(1, n+1) - n - 1) * sorted_balances)) / (n * total)
            
            # 前10持仓占比
            top10 = sum(sorted_balances[-10:])/total if len(balances)>=10 else 1
            
            return {'gini': min(gini, 1), 'top10': min(top10, 1)}
        except:
            return {'gini': 1, 'top10': 1}

    async def _get_token_age(self, token_address: str) -> float:
        """计算代币年龄（年）"""
        try:
            creation_block = self.w3.eth.get_transaction_receipt(token_address).blockNumber
            current_block = self.w3.eth.block_number
            return (current_block - creation_block) * 3 / (60 * 24 * 365)  # BSC每3秒一个区块
        except:
            return 0

    def _calculate_ta(self, prices: List[float]) -> Dict:
        """使用Pandas计算技术指标"""
        if len(prices) < 14:
            return {'rsi': 0.5, 'macd': 0.5}
        
        close = pd.Series(prices).ffill().bfill()
        
        # RSI计算
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = (100 - 100 / (1 + rs)).iloc[-1] / 100  # 归一化到0-1
        
        # MACD计算
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # 标准化处理
        macd_diff = (macd_line.iloc[-1] - signal_line.iloc[-1])
        macd = np.tanh(macd_diff / (close.std() + 1e-9)) * 0.5 + 0.5
        
        return {
            'rsi': max(0, min(1, rsi)),
            'macd': max(0, min(1, macd))
        }

    def _calculate_score(self, dex: Dict, holders: Dict, age: float, ta: Dict) -> float:
        """综合评分计算"""
        weights = {
            'liquidity': 0.35,
            'volatility': 0.25,
            'holders': 0.25,
            'age': 0.1,
            'ta': 0.05
        }
        
        # 流动性评分（对数标准化）
        liquidity = np.log1p(dex['liquidity']) / np.log1p(1e7)
        
        # 波动率评分（逆向）
        volatility = 1 - min(np.std(np.diff(dex['prices'])/ (np.mean(dex['prices'][:-1]) + 1e-9)) * 10, 1) \
            if len(dex['prices'])>1 else 0.5
        
        # 持币分布评分
        holders_score = (1 - holders['gini']) * (1 - holders['top10'])
        
        # 代币年龄评分
        age_score = min(age, 1)
        
        # 技术指标评分
        ta_score = (ta['rsi'] + ta['macd']) / 2
        
        return round(100 * (
            weights['liquidity'] * liquidity +
            weights['volatility'] * volatility +
            weights['holders'] * holders_score +
            weights['age'] * age_score +
            weights['ta'] * ta_score
        ), 2)

    async def _find_main_pair(self, token_address: str) -> str:
        """寻找主要交易对"""
        for pair in self.cached_pairs:
            pair_contract = self.w3.eth.contract(address=pair, abi=self._pair_abi())
            if token_address in [pair_contract.functions.token0().call(),
                                pair_contract.functions.token1().call()]:
                return pair
        return None

    def _calc_price_impact(self, pair_contract, amount: int, token_address: str) -> float:
        """计算大额交易价格影响"""
        try:
            reserves = pair_contract.functions.getReserves().call()
            token0 = pair_contract.functions.token0().call()
            amount_in = amount * 10**18  # 假设输入的是BNB
            
            if token0 == token_address:
                new_reserve0 = reserves[0] + amount_in
                new_price = reserves[1] / new_reserve0
                impact = abs((reserves[1]/reserves[0] - new_price) / (reserves[1]/reserves[0]))
            else:
                new_reserve1 = reserves[1] + amount_in
                new_price = reserves[0] / new_reserve1
                impact = abs((reserves[0]/reserves[1] - new_price) / (reserves[0]/reserves[1]))
            
            return min(impact, 1)
        except:
            return 1

    def _calc_volatility(self, prices: List[float]) -> float:
        """计算价格波动率"""
        if len(prices) < 2:
            return 0
        returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-9)
        return np.std(returns)

    async def _get_symbol(self, token_address: str) -> str:
        """获取代币符号"""
        try:
            contract = self.w3.eth.contract(address=token_address, abi=self._bep20_abi())
            return contract.functions.symbol().call()
        except:
            return "UNKNOWN"

    def _bep20_abi(self) -> List[Dict]:
        """BEP20标准ABI"""
        return [
            {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"type":"function"},
            {"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"}
        ]

    def _pair_abi(self) -> List[Dict]:
        """交易对合约ABI"""
        return [
            {"constant":True,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"getReserves","outputs":[{"name":"_reserve0","type":"uint112"},{"name":"_reserve1","type":"uint112"},{"name":"_blockTimestampLast","type":"uint32"}],"type":"function"}
        ]

    def _router_abi(self) -> List[Dict]:
        """路由合约ABI"""
        return [
            {"constant":True,"inputs":[],"name":"factory","outputs":[{"name":"","type":"address"}],"type":"function"}
        ]
