import pandas as pd
import numpy as np
import os
import time
import random
import calendar
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 忽略 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 設定與常數 (Settings and Constants)
# ==========================================
class Config:
    # 檔案路徑
    DATA_FILE = 'cleaned_stock_data1.xlsx'
    RESULT_FILE = 'strategy_results.xlsx'
    
    # 資金管理
    INITIAL_CAPITAL = 20_000_000  # 2千萬
    TAX_RATE = 0.003 # 交易稅 (台股一般是0.3%，但題目寫0.1%？ 確認題目)
    # 題目: 交易稅：0.1% -> 修正為 0.001
    TAX_RATE = 0.001
    FEE_RATE = 0.001425 # 手續費 (題目未提，但通常有，題目只提交易稅與滑價)
    # 題目: 滑價：0.3% -> 這是交易成本的一種模擬，通常用於進出價格的調整
    # 題目未提手續費，只提 "交易稅0.1%" 和 "滑價0.3%"。
    # 嚴格依照題目：成本 = 稅 + 滑價。
    # 滑價通常是 買入價 * (1 + 滑價)，賣出價 * (1 - 滑價)。
    # 或者將滑價視為費用扣除。
    # 題目寫 "交易稅0.1%", "滑價0.3%". 為了嚴謹，我將滑價應用於價格。
    SLIPPAGE = 0.003
    
    # ACO 參數
    ANT_COUNT = 5       # 螞蟻數量
    GENERATIONS = 5      # 世代數 (可調整)
    EVAPORATION = 0.4    # 費洛蒙揮發率
    ALPHA = 1.0          # 費洛蒙重要性因子
    
    # 參數範圍 (用於 ACO 探索)
    
    # n1, n2, n3: RSI天數 (例如 3~24)
    # V_Bar: 成交量濾網 (千萬元) (10~500)
    # bias60_bar: 乖離率濾網 (例如 5~30)
    # PCT_STP: 停損 % (例如 5~20)
    PARAM_RANGES = {
        'n1': list(range(5, 6,2)),  
        'n2': list(range(20, 21,2)),
        'n3': list(range(40, 41,2)),
        'S_H': list(range(2, 5, 1 )),
        'V_Bar': list(range(7, 11, 2)),
        'bias60_bar': list(range(80, 90, 5)),
        'PCT_STP': list(range(7, 19, 2))
    }

# ==========================================
# 資料讀取類別 (Data Loader)
# ==========================================
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.adj_close = None
        self.volume = None
        self.foreign = None
        
    def load_data(self):
        print("讀取資料中... (Loading Data)")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"找不到檔案: {self.filepath}")
            
        xls = pd.ExcelFile(self.filepath)
        
        # 讀取 P (還原收盤價), Q (成交量), F (外資 - 此策略暫未用到但讀取備用)
        # sheet_names check
        valid_sheets = [s for s in xls.sheet_names if s in ['P', 'Q', 'F']]
        if 'P' not in valid_sheets or 'Q' not in valid_sheets:
            raise ValueError("Excel 缺少 P 或 Q 工作表")

        self.adj_close = pd.read_excel(xls, 'P', index_col=0)
        self.volume = pd.read_excel(xls, 'Q', index_col=0)
        
        # 確保索引是 DateTime
        self.adj_close.index = pd.to_datetime(self.adj_close.index)
        self.volume.index = pd.to_datetime(self.volume.index)
        
        # 確保欄位一致 (取交集)
        common_cols = self.adj_close.columns.intersection(self.volume.columns)
        self.adj_close = self.adj_close[common_cols]
        self.volume = self.volume[common_cols]
        
        print(f"資料讀取完成。期間: {self.adj_close.index[0]} 至 {self.adj_close.index[-1]}")
        return self.adj_close, self.volume

# ==========================================
# 策略邏輯類別 (Strategy)
# ==========================================
class Strategy:
    def __init__(self, data_close, data_volume, params):
        self.close = data_close
        self.volume = data_volume  # 單位: 張
        self.params = params
        
        # 參數解包
        self.n1 = params['n1']
        self.n2 = params['n2']
        self.n3 = params['n3']
        self.S_H = params['S_H']
        self.V_Bar = params['V_Bar'] # 千萬元
        self.bias60_bar = params['bias60_bar']
        self.PCT_STP = params['PCT_STP']
        
        # 計算指標
        self.indicators = {}
        self._calculate_indicators()
        
    def _calculate_rsi(self, series, period):
        delta = series.diff()
        
        # Wilder's Smoothing (EMA with alpha=1/period)
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_indicators(self):
        # 1. RSI
        rsi1 = self.close.apply(lambda x: self._calculate_rsi(x, self.n1))
        rsi2 = self.close.apply(lambda x: self._calculate_rsi(x, self.n2))
        rsi3 = self.close.apply(lambda x: self._calculate_rsi(x, self.n3))
        
        # 平均 RSI
        self.indicators['rsi_avg'] = (rsi1 + rsi2 + rsi3) / 3
        
        # 2. 20日均量 (MA20_Vol)
        self.indicators['ma20_vol'] = self.volume.rolling(window=20).mean()
        
        # 3. 60日乖離率 (Bias60)
        ma60 = self.close.rolling(window=60).mean()
        self.indicators['bias60'] = ((self.close - ma60) / ma60) * 100
        
    def run_backtest(self):
        # 初始化回測變數
        capital = Config.INITIAL_CAPITAL
        position_size = capital / self.S_H # 固定進場金額
        positions = {} # {ticker: {'shares': int, 'cost': float, 'entry_date': date}}
        
        # 紀錄
        equity_curve = [] # [{'Date': date, 'Equity': float}]
        trades = [] # List of trade records
        daily_holdings = [] # List of daily holdings
        daily_candidates = [] # 符合條件股票數
        
        dates = self.close.index
        # 從最大 days 開始，避免指標 NaN
        start_idx = max(self.n3, 60) 
        
        for i in range(start_idx, len(dates) - 1):
            t_date = dates[i]
            next_date = dates[i+1] # T+1 日，執行交易
            
            # --- T日 資料與計算 ---
            # 1. 計算 RSI 分數, 排序
            # 2. 篩選
            
            # 當日數據
            current_close = self.close.loc[t_date]
            current_rsi = self.indicators['rsi_avg'].loc[t_date]
            current_ma20_vol = self.indicators['ma20_vol'].loc[t_date] # 張數
            current_bias60 = self.indicators['bias60'].loc[t_date]
            
            # 候選名單: RSI 由高到低排序
            # 去除 RSI 為 NaN 的
            rank_series = current_rsi.dropna().sort_values(ascending=False)
            ranked_tickers = rank_series.index.tolist()
            
            candidates = []
            
            # 篩選邏輯
            for ticker in ranked_tickers:
                # 條件 (2) 成交量濾網: 20日均量 * T日收盤價 * 1000 > V_Bar (千萬元)
                # 成交量單位為張，故 * 1000 轉為股數。 V_Bar 單位為千萬元，故 * 10,000,000 轉為元
                # 公式: (均量 * 1000 * 收盤價) > V_Bar * 10,000,000
                vol_val = current_ma20_vol.get(ticker, 0)
                price = current_close.get(ticker, 0)
                
                if pd.isna(vol_val) or pd.isna(price) or price == 0:
                    continue
                    
                txn_val = vol_val * 1000 * price
                if txn_val <= self.V_Bar * 10_000_000:
                    continue
                    
                # 條件 (3) 60日乖離低於 bias60_bar
                bias = current_bias60.get(ticker)
                # "若資料不足忽略此條件": 即 bias 為 NaN 時視為通過 (或是忽略該股票? 題目語意可能是忽略"條件"即不檢查)
                # 通常 "忽略此條件" 意味著 Pass。但若 Bias 為 NaN 代表上市不滿 60 天。
                # 這裡假設: 如果有 Bias 值，必須 < bar。如果沒有 (NaN)，則通過。
                if not pd.isna(bias):
                    if bias >= self.bias60_bar:
                        continue
                
                candidates.append(ticker)
                if len(candidates) >= self.S_H:
                    break
            
            daily_candidates.append({'Date': t_date, 'Count': len(candidates), 'Candidates': str(candidates)})
            
            # --- 交易執行 (T+1日) ---
            # 價格
            # 買進: T+1 收盤價 (題目: "於T+1日收盤價買進")
            # 賣出: T+1 收盤價 (題目: "於T+1日收盤價賣出")
            # 嚴格依照題目：這裡不包含盤中停損，而是根據 T 日訊號在 T+1 收盤執行。
            # 但停損是 Exit Signal (1) 停損 PCT_STP%。是指 T+1 盤中碰到? 還是 T+1 收盤確認跌破?
            # 題目寫: "出場條件 (1) 停損... 於T+1日收盤價賣出"。
            # 這暗示是 T 日檢查到 "收盤價 < 停損價" (或其他觸發)，則 T+1 收盤賣出。
            # 或者是 T+1 日當天跌破，則 T+1 收盤賣出?
            # 通常回測日結算：T日收盤檢查持倉是否觸發停損。
            # 邏輯：檢查 T 日 Close 是否低於 成本 * (1 - STP%)。若是，則 T+1 賣出。
            
            # T+1 價格
            next_close = self.close.loc[next_date]
            
            # 1. 賣出邏輯 (停損 & 換股)
            # 策略核心: "取最大S_H檔持有"。這意味著我們要持有 candidates 中的股票。
            # 如果持有的股票不在 candidates 中 (且滿倉)，是否賣出?
            # 題目 "出場條件" 只有停損。
            # 但 "進場條件 (1) 取最大S_H檔持有" 暗示了這是一個 Rankings 策略。
            # 如果不換股，就不是 Sort 策略了。
            # 因此嚴格解讀：
            #  - 必須賣出 "非 Top S_H" 的持股，以便買入 "Top S_H"。
            #  - 或者，只要持股還在，就不賣，除非停損? (Buy and Hold logic)
            #  - 題目名稱 "RSI SORT"，通常由 RSI 高者取代低者。
            #  - 這裡採用標準 Sort 邏輯：目標持倉是 candidates。
            #  - 任何目前持有但不在 candidates 的，理應賣出 (Rebalance)。
            #  - 任何在 candidates 但未持有的，理應買入。
            #  - 優先執行 "停損"。
            
            # 為了符合 "嚴格" 以及 "Sort" 特性，我將採用 Rebalance 邏輯。
            # 即: 賣出 [持有 - Candidates], 買入 [Candidates - 持有]
            # 同時檢查停損。
            
            stocks_to_sell = []
            
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                cost = pos['cost']
                curr_price = current_close.get(ticker) # T日收盤價
                
                # 停損檢查
                if not pd.isna(curr_price) and curr_price < cost * (1 - self.PCT_STP / 100):
                    stocks_to_sell.append((ticker, 'Stop Loss'))
                # 換股檢查 (Rebalance)
                elif ticker not in candidates:
                    # 如果不在前 S_H 名，是否賣出？
                    # 為了嚴格遵守 "取最大 S_H 檔持有"，如果不賣，我們就持有了 "非最大 S_H 檔"。
                    # 所以應該賣出。
                    stocks_to_sell.append((ticker, 'Rebalance (Rank Drop)'))
            
            # 執行賣出
            for ticker, reason in stocks_to_sell:
                if ticker in positions:
                    pos = positions[ticker]
                    # T+1 價格
                    exec_price = next_close.get(ticker)
                    if pd.isna(exec_price) or exec_price == 0:
                        continue # 無法交易
                        
                    # 滑價 (賣出價變低)
                    sell_price = exec_price * (1 - Config.SLIPPAGE)
                    # 稅
                    tax = sell_price * pos['shares'] * Config.TAX_RATE
                    # 手續費 (若有) -> 這裡題目未提手續費，故只計稅。但變數有 FEE_RATE
                    # 嚴格題目：只計稅 0.1% 和 滑價 0.3%。
                    # 滑價已反映在價格。
                    
                    revenue = sell_price * pos['shares'] - tax
                    pnl = revenue - (pos['cost'] * pos['shares']) # 簡易 PnL
                    # 精確 PnL 應包含買入成本
                    
                    ret = (sell_price - pos['cost']) / pos['cost']
                    
                    capital += revenue
                    
                    trades.append({
                        'Date': next_date,
                        'Ticker': ticker,
                        'Action': 'Sell',
                        'Price': exec_price,
                        'Shares': pos['shares'],
                        'Reason': reason,
                        'PnL': pnl,
                        'Return': ret
                    })
                    
                    del positions[ticker]
            
            # 3. 買進邏輯
            # 檢查資金是否足夠？或是每檔獨立資金？
            # 題目: "進場S_B：總資金 / S_H檔"。
            # 這通常指 Initial Capital / S_H，或者 Current Capital / S_H?
            # 題目寫 "總資金：2千萬元(固定)"。這暗示 Position Sizing 固定為 20M / S_H。
            # 即便賺錢，加碼金額也不變？或者 Total Asset / S_H?
            # "總資金 2千萬元 (固定)" -> 應指初始規劃。
            # 若採用 Fixed Fractional，則隨權益變動。但 (固定) 二字可能指 Fixed Amount。
            # 保守起見，使用 20,000,000 / S_H 作為每檔預算。
            
            entry_budget = Config.INITIAL_CAPITAL / self.S_H
            
            for ticker in candidates:
                if ticker in positions:
                    continue # 已持有
                
                if len(positions) >= self.S_H:
                    break # 已滿倉
                
                # 執行買進
                exec_price = next_close.get(ticker)
                
                if pd.isna(exec_price) or exec_price == 0:
                    continue
                
                # 滑價 (買入價變高)
                buy_price = exec_price * (1 + Config.SLIPPAGE)
                
                # 計算股數 (無條件捨去至整股)
                # 預算 entry_budget
                # 成本 = buy_price * shares * (1) -> 買入無稅? 台股買入無稅。
                if buy_price > entry_budget:
                    continue # 買不起一張
                
                shares = int(entry_budget // buy_price)
                if shares == 0:
                    continue
                    
                cost_amt = shares * buy_price
                
                # 檢查目前現金是否足夠 (Real Cash Check)
                # 如果回測是模擬資金池，必須檢查 Cash。
                # 但題目說 "總資金(固定)" 可能是指部位規模計算方式。
                # 這裡假設我有無限現金? 不，應該受 Capital 限制。
                if capital < cost_amt:
                    # 現金不足，無法買進
                    continue
                
                capital -= cost_amt
                positions[ticker] = {
                    'shares': shares,
                    'cost': buy_price,
                    'entry_date': next_date
                }
                
                trades.append({
                    'Date': next_date,
                    'Ticker': ticker,
                    'Action': 'Buy',
                    'Price': exec_price,
                    'Shares': shares,
                    'Reason': 'Entry: Rank Top',
                    'PnL': 0,
                    'Return': 0
                })
            
            # 結算當日權益 (T+1)
            # Equity = Cash + MV of Positions
            curr_equity = capital
            holdings_info = []
            for t, pos in positions.items():
                p = next_close.get(t)
                if pd.isna(p): p = pos['cost'] # 若無價格，用成本估算
                mv = p * pos['shares']
                curr_equity += mv
                holdings_info.append(f"{t}({pos['shares']})")
                
            equity_curve.append({'Date': next_date, 'Equity': curr_equity})
            daily_holdings.append({'Date': next_date, 'Holdings_Count': len(positions), 'Details': str(holdings_info)})
            
        # 整理結果
        self.equity_df = pd.DataFrame(equity_curve).set_index('Date')
        self.trades_df = pd.DataFrame(trades)
        self.daily_holdings_df = pd.DataFrame(daily_holdings)
        self.daily_candidates_df = pd.DataFrame(daily_candidates)
        
        return self._calculate_metrics()

    def _calculate_metrics(self):
        if self.equity_df.empty:
            return {'CAGR': -99.0, 'Calmar': -99.0}
            
        initial = Config.INITIAL_CAPITAL
        final = self.equity_df['Equity'].iloc[-1]
        
        # Years
        start_date = self.equity_df.index[0]
        end_date = self.equity_df.index[-1]
        years = (end_date - start_date).days / 365.25
        
        if years <= 0: return {'CAGR': -0.99, 'Calmar': -99}

        # CAGR
        cagr = (final / initial) ** (1 / years) - 1
        
        # MaxDD
        equity = self.equity_df['Equity']
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min() # negative value
        
        # Calmar
        calmar = -cagr / max_dd if max_dd != 0 else 0
        
        return {
            'CAGR': cagr,
            'Calmar': calmar,
            'MaxDD': max_dd,
            'Final_Equity': final
        }

# ==========================================
# 螞蟻演算法 (ACO) 類別
# ==========================================
class AntColonyOptimizer:
    def __init__(self, data_close, data_volume):
        self.close = data_close
        self.volume = data_volume
        self.best_solution = None
        self.best_fitness = -np.inf
        # 初始化費洛蒙: 每個參數的每個可能值
        self.pheromones = {}
        for param, values in Config.PARAM_RANGES.items():
            self.pheromones[param] = {v: 1.0 for v in values}
            
    def _select_value(self, param):
        # 輪盤法選擇
        values = Config.PARAM_RANGES[param]
        probs = [self.pheromones[param][v] ** Config.ALPHA for v in values]
        total = sum(probs)
        probs = [p / total for p in probs]
        return np.random.choice(values, p=probs)

    def _select_value_from_subset(self, param, subset):
        if not subset:
            # 防呆: 如果沒有合適的值，隨機選一個 (理論上不應發生，除非範圍設定有誤)
            return self._select_value(param)
            
        probs = [self.pheromones[param][v] ** Config.ALPHA for v in subset]
        total = sum(probs)
        if total == 0:
            probs = [1.0 / len(subset)] * len(subset)
        else:
            probs = [p / total for p in probs]
        return np.random.choice(subset, p=probs)
    
    def run(self):
        print("\n開始 ACO 最佳化... (Starting ACO)")
        
        for gen in range(Config.GENERATIONS):
            gen_best_fitness = -np.inf
            gen_best_ant = None
            ants_results = []
            
            print(f"--- 世代 {gen + 1} / {Config.GENERATIONS} ---")
            
            for ant in range(Config.ANT_COUNT):
                # 建構解
                params = {}
                
                # 特殊處理: n1 < n2 < n3
                # 1. 選 n1
                params['n1'] = self._select_value('n1')
                
                # 2. 選 n2 > n1
                subset_n2 = [v for v in Config.PARAM_RANGES['n2'] if v > params['n1']]
                params['n2'] = self._select_value_from_subset('n2', subset_n2)
                
                # 3. 選 n3 > n2
                subset_n3 = [v for v in Config.PARAM_RANGES['n3'] if v > params['n2']]
                params['n3'] = self._select_value_from_subset('n3', subset_n3)
                
                # 其他參數
                for k in ['S_H', 'V_Bar', 'bias60_bar', 'PCT_STP']:
                    params[k] = self._select_value(k)
                
                # 執行策略
                strat = Strategy(self.close, self.volume, params)
                metrics = strat.run_backtest()
                fitness = metrics['CAGR'] # 目標: 最大化 CAGR
                
                ants_results.append((params, fitness, metrics))
                
                # 更新世代最佳
                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_ant = params
                    
                # 更新全域最佳
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = params
                    print(f"  [新紀錄] Ant {ant}: CAGR={fitness:.2%}, Calmar={metrics['Calmar']:.2f}")
                    print(f"  Params: {params}")

            # 費洛蒙更新 (蒸發 + 堆積)
            # 蒸發
            for param in self.pheromones:
                for v in self.pheromones[param]:
                    self.pheromones[param][v] *= (1 - Config.EVAPORATION)
            
            # 堆積 (菁英策略: 只強化表現好的)
            # 這裡簡單強化前 50% 的螞蟻
            sorted_ants = sorted(ants_results, key=lambda x: x[1], reverse=True)
            top_ants = sorted_ants[:max(1, len(sorted_ants)//2)]
            
            for params, fitness, _ in top_ants:
                # 避免負值 fitness 影響 (若 CAGR < 0, 不獎勵或少獎勵)
                # 這裡使用 rank based 或 normalized fitness
                # 簡單作法: 加固定量 或 與 fitness 成正比 (若正)
                deposit = max(0, fitness) if fitness > 0 else 0.01 
                for k, v in params.items():
                    self.pheromones[k][v] += deposit
            
            print(f"  世代最佳: CAGR={gen_best_fitness:.2%}")

        print("\nACO 完成。")
        print(f"最佳參數: {self.best_solution}")
        print(f"最佳 CAGR: {self.best_fitness:.2%}")
        return self.best_solution

# ==========================================
# 主程式 (Main)
# ==========================================
def main():
    # 1. 讀取資料
    loader = DataLoader(Config.DATA_FILE)
    close_data, volume_data = loader.load_data()
    
    # 2. 執行 ACO 找最佳參數
    optimizer = AntColonyOptimizer(close_data, volume_data)
    best_params = optimizer.run()
    
    # 3. 使用最佳參數跑最後一次策略並輸出詳細報告
    print("\n使用最佳參數產生報告...")
    final_strat = Strategy(close_data, volume_data, best_params)
    metrics = final_strat.run_backtest()
    
    # 4. 輸出 Excel
    with pd.ExcelWriter(Config.RESULT_FILE, engine='openpyxl') as writer:
        final_strat.trades_df.to_excel(writer, sheet_name='Trades')
        final_strat.equity_df.to_excel(writer, sheet_name='Equity_Curve')
        final_strat.daily_holdings_df.to_excel(writer, sheet_name='Equity_Hold')
        final_strat.daily_candidates_df.to_excel(writer, sheet_name='Daily_Candidate')
        
        # Summary
        summary_data = {
            'Metric': ['CAGR', 'MaxDD', 'Calmar', 'Final Equity', 'Best Params'],
            'Value': [
                metrics['CAGR'], 
                metrics['MaxDD'], 
                metrics['Calmar'], 
                metrics['Final_Equity'],
                str(best_params)
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
    print(f"結果已儲存至 {Config.RESULT_FILE}")
    
    # 畫權益曲線圖
    plt.figure(figsize=(10, 6))
    plt.plot(final_strat.equity_df.index, final_strat.equity_df['Equity'], label='Equity')
    plt.title(f"Equity Curve (CAGR: {metrics['CAGR']:.2%})")
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.savefig('equity_curve.png')
    print("權益曲線圖已儲存為 equity_curve.png")

if __name__ == '__main__':
    main()
