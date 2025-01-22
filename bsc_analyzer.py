import numpy as np
import pandas as pd
from web3 import Web3
from web3.middleware import geth_poa_middleware
import talib
import asyncio
from collections import deque

class BSCQuantAnalyzer:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider('https://bsc-dataseed.binance.org/'))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.dex_router = '0x10ED43C718714eb63d5aA57B78B54704E256024E'  # PancakeSwap Router
        self.cached_pairs = deque(maxlen=1000)

    async def get_token_list(self):
        """获取BSC链上前1000个交易对中的代币"""
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
                tokens.add(pair_contract.functions.token0().call())
                tokens.add(pair_contract.functions.token1().call())
                self.cached_pairs.append(pair)
                
            return list(tokens)
        except Exception as e:
            print(f"获取代币列表失败: {e}")
            return []

    async def validate_token(self, token_address):
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

    async def analyze_token(self, token_address):
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

    async def _get_dex_metrics(self, token_address):
        """获取DEX市场深度数据"""
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
            
            price = reserves[0] / reserves[1] * 10**(decimals1 - decimals0) if token0 == token_address \
                   else reserves[1] / reserves[0] * 10**(decimals0 - decimals1)
            
            return {
                'liquidity': (reserves[0] * 10**-decimals0 * price) + (reserves[1] * 10**-decimals1),
                'prices': [price],
                'impact_10k': self._calc_price_impact(pair_contract, 10000, token_address),
                'impact_100k': self._calc_price_impact(pair_contract, 100000, token_address)
            }
        except Exception as e:
            print(f"DEX数据获取失败: {e}")
            return {'liquidity':0, 'prices':[], 'impact_10k':1, 'impact_100k':1}

    async def _get_holder_metrics(self, token_address):
        """持币分布分析"""
        try:
            contract = self.w3.eth.contract(address=token_address, abi=self._bep20_abi())
            transfers = []
            
            # 扫描最近100个区块
            latest_block = self.w3.eth.block_number
            for bn in range(latest_block-100, latest_block):
                block = self.w3.eth.get_block(bn, full_transactions=True)
                for tx in block.transactions:
                    if tx['to'] == token_address:
                        transfers.append(tx['from'])
            
            # 获取持币余额
            holders = set(transfers[-1000:])  # 取最近1000个交易地址
            balances = [contract.functions.balanceOf(h).call() for h in holders]
            balances = [b/10**contract.functions.decimals().call() for b in balances if b > 0]
            
            if not balances:
                return {'gini': 1, 'top10': 1}
            
            sorted_balances = np.sort(balances)
            total = sum(sorted_balances)
            cum_balances = np.cumsum(sorted_balances)
            
            gini = 1 - 2 * np.trapz(cum_balances/total) / len(balances)
            top10 = sum(sorted_balances[-10:])/total if len(balances)>=10 else 1
            
            return {'gini': gini, 'top10': top10}
        except:
            return {'gini': 1, 'top10': 1}

    async def _get_token_age(self, token_address):
        """计算代币年龄"""
        try:
            creation_block = self.w3.eth.get_transaction_receipt(token_address).blockNumber
            current_block = self.w3.eth.block_number
            return (current_block - creation_block) * 3 / (60 * 24 * 365)  # BSC每3秒一个区块
        except:
            return 0

    def _calculate_ta(self, prices):
        """技术指标分析"""
        if len(prices) < 14:
            return {'rsi': 0.5, 'macd': 0.5}
        
        close = pd.Series(prices).ffill().bfill().values
        rsi = talib.RSI(close, timeperiod=14)[-1]/100
        macd, signal, _ = talib.MACD(close)
        macd_score = 0.5 + (macd[-1] - signal[-1])/np.std(close)
        return {'rsi': max(0, min(1, rsi)), 'macd': max(0, min(1, macd_score))}

    def _calculate_score(self, dex, holders, age, ta):
        weights = {
            'liquidity': 0.3,
            'volatility': 0.2,
            'holders': 0.25,
            'age': 0.1,
            'ta': 0.15
        }
        
        liquidity = np.log1p(dex['liquidity'])/np.log1p(1e6)
        volatility = 1 - min(np.std(np.diff(dex['prices'])/dex['prices'][:-1])*10, 1) if len(dex['prices'])>1 else 0.5
        holders_score = (1 - holders['gini']) * (1 - holders['top10'])
        age_score = min(age, 1)
        ta_score = (ta['rsi'] + ta['macd']) / 2
        
        return round(100 * (
            weights['liquidity'] * liquidity +
            weights['volatility'] * volatility +
            weights['holders'] * holders_score +
            weights['age'] * age_score +
            weights['ta'] * ta_score
        ), 2)

    async def _find_main_pair(self, token_address):
        """寻找主要交易对"""
        for pair in self.cached_pairs:
            pair_contract = self.w3.eth.contract(address=pair, abi=self._pair_abi())
            if token_address in [pair_contract.functions.token0().call(),
                                pair_contract.functions.token1().call()]:
                return pair
        return None

    def _calc_price_impact(self, pair_contract, amount, token_address):
        """计算大额交易价格影响"""
        try:
            reserves = pair_contract.functions.getReserves().call()
            token0 = pair_contract.functions.token0().call()
            amount_in = amount * 10**18  # 假设输入的是BNB
            
            if token0 == token_address:
                return abs((reserves[1] - reserves[1]*reserves[0]/(reserves[0]+amount_in)) / reserves[1])
            else:
                return abs((reserves[0] - reserves[0]*reserves[1]/(reserves[1]+amount_in)) / reserves[0])
        except:
            return 1

    def _calc_volatility(self, prices):
        """计算价格波动率"""
        if len(prices) < 2:
            return 0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

    async def _get_symbol(self, token_address):
        """获取代币符号"""
        try:
            contract = self.w3.eth.contract(address=token_address, abi=self._bep20_abi())
            return contract.functions.symbol().call()
        except:
            return "UNKNOWN"

    def _bep20_abi(self):
        return [
            {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"type":"function"},
            {"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"}
        ]

    def _pair_abi(self):
        return [
            {"constant":True,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"type":"function"},
            {"constant":True,"inputs":[],"name":"getReserves","outputs":[{"name":"_reserve0","type":"uint112"},{"name":"_reserve1","type":"uint112"},{"name":"_blockTimestampLast","type":"uint32"}],"type":"function"}
        ]

    def _router_abi(self):
        return [
            {"constant":True,"inputs":[],"name":"factory","outputs":[{"name":"","type":"address"}],"type":"function"}
      ]
