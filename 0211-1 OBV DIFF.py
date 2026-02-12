import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt

# 忽略 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

class Config:
    """設定與常數 (Settings and Constants)"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(BASE_DIR, 'cleaned_stock_data1.xlsx')
    INDEX_FILE = os.path.join(BASE_DIR, '../Clean data 0209/Index_data.xlsx') # 假設相對路徑，若不在則需調整
    RESULT_FILE = os.path.join(BASE_DIR, 'strategy_results.xlsx')
    
    # 資金管理
    INITIAL_CAPITAL = 20_000_000  # 2千萬 (固定)
    TAX_RATE = 0.001     # 交易稅 0.1%
    SLIPPAGE = 0.003     # 滑價 0.3%
    
    # ACO 參數
    ANT_COUNT = 50       # 螞蟻數量 (可調整)
    GENERATIONS = 5     # 世代數 (可調整)
    EVAPORATION = 0.5    # 費洛蒙揮發率
    ALPHA = 1.0          # 費洛蒙重要性因子 (控制探索與利用的平衡)
    
    # 策略固定參數
    RSI_PERIOD = 14      # RSI天數 (固定 14)
    
    # 參數範圍 (用於 ACO 探索)
    PARAM_RANGES = {
        'S_H': list(range(2, 11, 1)),      # S_H: 最大持倉檔數 (2~10)
        'RE_DAYS': list(range(5, 6, 5)),  # RE_DAYS: 再平衡天數 (5~60)
        'EXIT_MA': list(range(15, 21, 5)),  # EXIT_MA: 出場均線 (5~60)
        'OBV_RANK': list(range(1, 6, 1)),  # OBV_RANK: 每次買進時，選擇OBV前幾名的股票 (1~5)
        'OBV_WINDOW': list(range(5, 15, 5)), # OBV_WINDOW: OBV增幅計算天數 (2~10)
        'IDX_MA1': list(range(10, 30, 10)),   # IDX_MA1: 指數濾網均線1 (20~200)
        'IDX_MA2': list(range(20, 60, 20))    # IDX_MA2: 指數濾網均線2 (20~200)
    }

class DataLoader:
    """資料讀取類別 (Data Loader)"""
    def __init__(self, filepath, index_filepath=None):
        self.filepath = filepath
        self.index_filepath = index_filepath
        self.adj_close = None
        self.volume = None
        self.index_data = None
        
    def load_data(self):
        print("讀取個股資料中... (Loading Stock Data)")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"找不到檔案: {self.filepath}")
            
        xls = pd.ExcelFile(self.filepath)
        
        # 檢查並讀取 P (還原收盤價), Q (成交量)
        if 'P' not in xls.sheet_names or 'Q' not in xls.sheet_names:
            raise ValueError("Excel 缺少 P 或 Q 工作表")

        self.adj_close = pd.read_excel(xls, 'P', index_col=0)
        self.volume = pd.read_excel(xls, 'Q', index_col=0)
        
        # 確保索引格式與欄位一致
        self.adj_close.index = pd.to_datetime(self.adj_close.index)
        self.volume.index = pd.to_datetime(self.volume.index)
        
        common_cols = self.adj_close.columns.intersection(self.volume.columns)
        self.adj_close = self.adj_close[common_cols]
        self.volume = self.volume[common_cols]
        
        # 填補缺失值
        self.adj_close.ffill(inplace=True)
        self.volume.fillna(0, inplace=True)
        
        print(f"個股資料讀取完成。期間: {self.adj_close.index[0].date()} 至 {self.adj_close.index[-1].date()}")

        # 讀取指數資料
        if self.index_filepath:
            print(f"讀取指數資料中... (Loading Index Data from {self.index_filepath})")
            if not os.path.exists(self.index_filepath):
                 # 嘗試自動搜尋 Index_data.xlsx
                 potential_path = os.path.join(os.path.dirname(self.filepath), '../Clean data 0209/Index_data.xlsx')
                 if os.path.exists(potential_path):
                     self.index_filepath = potential_path
                 else:
                     print(f"警告: 找不到指數檔案 {self.index_filepath}，將無法使用指數濾網。")
                     self.index_data = None
                     return self.adj_close, self.volume, self.index_data

            try:
                # 假設指數資料第一欄是日期，第二欄是價格 (或是第一欄index是日期)
                # 通常 Index_data.xlsx 結構可能是一個 Series
                idx_df = pd.read_excel(self.index_filepath, index_col=0)
                idx_df.index = pd.to_datetime(idx_df.index)
                
                # 取第一欄作為指數價格 (通常是 'Close' 或唯一欄位)
                self.index_data = idx_df.iloc[:, 0]
                
                # 對齊日期 (Reindex to stock dates, ffill)
                self.index_data = self.index_data.reindex(self.adj_close.index).ffill()
                print("指數資料讀取完成。")
                
            except Exception as e:
                print(f"讀取指數資料失敗: {e}")
                self.index_data = None

        return self.adj_close, self.volume, self.index_data

class Strategy:
    """策略邏輯類別 (Strategy)"""
    def __init__(self, data_close, data_volume, index_data, params):
        self.close = data_close
        self.volume = data_volume  # 單位: 張
        self.index_data = index_data
        self.params = params
        
        # 參數解包
        self.S_H = params['S_H']
        self.RE_DAYS = params['RE_DAYS']
        self.EXIT_MA = params['EXIT_MA']
        self.OBV_RANK = params['OBV_RANK']
        self.OBV_WINDOW = params['OBV_WINDOW']
        self.IDX_MA1 = params.get('IDX_MA1', 200) 
        self.IDX_MA2 = params.get('IDX_MA2', 60) 
        
        self.indicators = {}
        self._calculate_indicators()
        
    def _calculate_rsi_wilder(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).fillna(50)
    
    def _calculate_indicators(self):
        # 1. RSI (14 days, Wilder's)
        self.indicators['rsi'] = self.close.apply(lambda x: self._calculate_rsi_wilder(x, Config.RSI_PERIOD))
        
        # 2. OBV & OBV Score
        change = self.close.diff()
        direction = np.sign(change)
        direction.iloc[0] = 0
        obv_flow = direction * self.volume
        self.indicators['obv'] = obv_flow.cumsum()
        
        # OBV Score (近OBV_WINDOW天增幅 * 收盤價)
        obv_diff = self.indicators['obv'].diff(self.OBV_WINDOW)
        self.indicators['obv_score'] = obv_diff * self.close
        
        # 3. EXIT_MA (出場均線)
        self.indicators['exit_ma'] = self.close.rolling(window=self.EXIT_MA).mean()
        
        # 4. 指數濾網 (IDX_MA1, IDX_MA2)
        if self.index_data is not None:
            self.indicators['idx_ma1'] = self.index_data.rolling(window=self.IDX_MA1).mean()
            self.indicators['idx_ma2'] = self.index_data.rolling(window=self.IDX_MA2).mean()
        
    def run_backtest(self):
        capital = Config.INITIAL_CAPITAL
        entry_budget = Config.INITIAL_CAPITAL / self.S_H 
        
        positions = {} # {ticker: {'shares': int, 'cost': float, 'entry_date': date}}
        equity_curve = [] 
        trades = [] 
        daily_holdings = [] 
        daily_candidates = [] 
        
        dates = self.close.index
        # 確保 idx_ma 也有足夠資料 (如果有 index data)
        idx_ma_period = max(self.IDX_MA1, self.IDX_MA2) if self.index_data is not None else 0
        start_idx = max(Config.RSI_PERIOD, self.OBV_WINDOW, self.EXIT_MA, idx_ma_period, 60) 
        
        for i in range(start_idx, len(dates) - 1):
            t_date = dates[i]
            next_date = dates[i+1] # T+1 日，執行交易
            
            # --- T日 資料 ---
            current_close = self.close.loc[t_date]
            current_rsi = self.indicators['rsi'].loc[t_date]
            current_obv_score = self.indicators['obv_score'].loc[t_date]
            current_exit_ma = self.indicators['exit_ma'].loc[t_date]
            
            # 指數濾網檢查 (雙濾網: 兩者皆跌破才禁止進場)
            allow_entry = True
            if self.index_data is not None:
                curr_idx_price = self.index_data.loc[t_date]
                curr_idx_ma1 = self.indicators['idx_ma1'].loc[t_date]
                curr_idx_ma2 = self.indicators['idx_ma2'].loc[t_date]
                
                # 檢查條件
                cond1 = (not pd.isna(curr_idx_ma1) and curr_idx_price < curr_idx_ma1)
                cond2 = (not pd.isna(curr_idx_ma2) and curr_idx_price < curr_idx_ma2)
                
                # 如果同時滿足 "跌破MA1" 且 "跌破MA2" -> 禁止進場
                if cond1 and cond2:
                    allow_entry = False
            
            is_rebalance_day = ((i - start_idx) % self.RE_DAYS == 0)
            entry_candidates = []
            
            if is_rebalance_day:
                if allow_entry:
                    # 篩選: RSI > 70 -> 排序: OBV Score -> 取前 OBV_RANK
                    valid_candidates = current_rsi[current_rsi > 70].index.tolist()
                    scores = current_obv_score.loc[valid_candidates].dropna()
                    ranked_candidates = scores.sort_values(ascending=False).index.tolist()
                    entry_candidates = ranked_candidates[:self.OBV_RANK]
                    
                    daily_candidates.append({
                        'Date': t_date, 
                        'Count': len(entry_candidates), 
                        'Candidates': str(entry_candidates),
                        'All_Valid': len(valid_candidates),
                        'Filter': 'Pass'
                    })
                else:
                    daily_candidates.append({'Date': t_date, 'Count': 0, 'Candidates': 'Blocked by Index Filter', 'Filter': 'Fail'})
            else:
                daily_candidates.append({'Date': t_date, 'Count': 0, 'Candidates': 'No Entry Check', 'Filter': 'NA'})

            # --- 交易決策 ---
            sell_list = [] # (ticker, reason)
            buy_list = []  # ticker
            
            # 1. 賣出訊號檢查: 跌破 EXIT_MA (T日訊號)
            for ticker in list(positions.keys()):
                price = current_close.get(ticker)
                ma_val = current_exit_ma.get(ticker)
                if not pd.isna(price) and not pd.isna(ma_val) and price < ma_val:
                    sell_list.append((ticker, f'Exit: Price < MA{self.EXIT_MA}'))
            
            # 2. 進場邏輯 (僅在 Rebalance 日執行)
            if is_rebalance_day and allow_entry:
                current_sell_tickers = [t for t, r in sell_list]
                for ticker in entry_candidates:
                    if ticker not in positions and ticker not in current_sell_tickers:
                         buy_list.append(ticker)
            
            # --- 執行交易 (T+1日) ---
            next_close_prices = self.close.loc[next_date]
            
            # 執行賣出
            for ticker, reason in sell_list:
                if ticker in positions:
                    pos = positions[ticker]
                    exec_price = next_close_prices.get(ticker)
                    
                    if pd.isna(exec_price) or exec_price == 0:
                        continue 
                        
                    sell_price = exec_price * (1 - Config.SLIPPAGE)
                    tax = sell_price * pos['shares'] * Config.TAX_RATE
                    revenue = sell_price * pos['shares'] - tax
                    pnl = revenue - (pos['cost'] * pos['shares'])
                    ret = (sell_price - pos['cost']) / pos['cost']
                    
                    capital += revenue
                    trades.append({
                        'Date': next_date, 'Ticker': ticker, 'Action': 'Sell',
                        'Price': exec_price, 'Shares': pos['shares'],
                        'Reason': reason, 'PnL': pnl, 'Return': ret
                    })
                    del positions[ticker]
            
            # 執行買進
            for ticker in buy_list:
                if len(positions) >= self.S_H:
                    break 
                
                exec_price = next_close_prices.get(ticker)
                if pd.isna(exec_price) or exec_price == 0:
                    continue
                
                buy_price = exec_price * (1 + Config.SLIPPAGE)
                if buy_price > entry_budget:
                    continue
                
                shares = int(entry_budget // buy_price)
                if shares == 0:
                    continue
                
                cost_amt = shares * buy_price
                if capital < cost_amt:
                    continue 
                
                capital -= cost_amt
                positions[ticker] = {
                    'shares': shares, 'cost': buy_price, 'entry_date': next_date
                }
                
                trades.append({
                    'Date': next_date, 'Ticker': ticker, 'Action': 'Buy',
                    'Price': exec_price, 'Shares': shares,
                    'Reason': 'Entry: Top S_H & RSI > 70', 'PnL': 0, 'Return': 0
                })
            
            # 結算當日權益
            curr_equity = capital
            holdings_info = []
            for t, pos in positions.items():
                p = next_close_prices.get(t)
                if pd.isna(p): p = pos['cost'] 
                mv = p * pos['shares']
                curr_equity += mv
                holdings_info.append(f"{t}({pos['shares']})")
                
            equity_curve.append({'Date': next_date, 'Equity': curr_equity})
            daily_holdings.append({'Date': next_date, 'Holdings_Count': len(positions), 'Details': str(holdings_info)})
            
        self.equity_df = pd.DataFrame(equity_curve)
        if not self.equity_df.empty:
            self.equity_df.set_index('Date', inplace=True)
            
        self.trades_df = pd.DataFrame(trades)
        self.daily_holdings_df = pd.DataFrame(daily_holdings)
        self.daily_candidates_df = pd.DataFrame(daily_candidates)
        
        return self._calculate_metrics()

    def _calculate_metrics(self):
        if self.equity_df.empty:
            return {'CAGR': -99.0, 'Calmar': -99.0, 'MaxDD': -1.0, 'Final_Equity': 0}
            
        initial = Config.INITIAL_CAPITAL
        final = self.equity_df['Equity'].iloc[-1]
        
        # Years
        start_date = self.equity_df.index[0]
        end_date = self.equity_df.index[-1]
        years = (end_date - start_date).days / 365.25
        
        if years <= 0: return {'CAGR': -0.99, 'Calmar': -99, 'MaxDD': -1.0, 'Final_Equity': final}

        cagr = (final / initial) ** (1 / years) - 1
        equity = self.equity_df['Equity']
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min() 
        calmar = -cagr / max_dd if max_dd != 0 else 0
        
        return {
            'CAGR': cagr, 'Calmar': calmar, 'MaxDD': max_dd, 'Final_Equity': final
        }

class AntColonyOptimizer:
    """螞蟻演算法 (ACO) 類別"""
    def __init__(self, data_close, data_volume, index_data):
        self.close = data_close
        self.volume = data_volume
        self.index_data = index_data
        self.best_solution = None
        self.best_fitness = -np.inf
        self.pheromones = {
            param: {v: 1.0 for v in values} 
            for param, values in Config.PARAM_RANGES.items()
        }
            
    def _select_value(self, param):
        values = Config.PARAM_RANGES[param]
        probs = [self.pheromones[param][v] ** Config.ALPHA for v in values]
        total = sum(probs)
        probs = [p / total for p in probs] if total > 0 else [1.0/len(values)]*len(values)
        return np.random.choice(values, p=probs)
    
    def run(self):
        print("\n開始 ACO 最佳化... (Starting ACO)")
        
        for gen in range(Config.GENERATIONS):
            gen_best_fitness = -np.inf
            ants_results = []
            
            print(f"--- 世代 {gen + 1} / {Config.GENERATIONS} ---")
            
            for ant in range(Config.ANT_COUNT):
                params = {k: self._select_value(k) for k in Config.PARAM_RANGES}
                
                strat = Strategy(self.close, self.volume, self.index_data, params)
                metrics = strat.run_backtest()
                fitness = metrics['CAGR']
                
                ants_results.append((params, fitness))
                
                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = params
                    print(f"  [新紀錄] Ant {ant}: CAGR={fitness:.2%}, Calmar={metrics['Calmar']:.2f}")
                    print(f"  Params: {params}")

            # 費洛蒙更新 (蒸發 + 堆積)
            for param in self.pheromones:
                for v in self.pheromones[param]:
                    self.pheromones[param][v] *= (1 - Config.EVAPORATION)
            
            sorted_ants = sorted(ants_results, key=lambda x: x[1], reverse=True)
            top_ants = sorted_ants[:max(1, len(sorted_ants)//2)]
            
            for params, fitness in top_ants:
                deposit = max(0.01, fitness) 
                for k, v in params.items():
                    self.pheromones[k][v] += deposit
            
            print(f"  世代最佳: CAGR={gen_best_fitness:.2%}")

        print("\nACO 完成。")
        print(f"最佳參數: {self.best_solution}")
        print(f"最佳 CAGR: {self.best_fitness:.2%}")
        return self.best_solution

def main():
    # 執行模式: 'ACO' (最佳化) 或 'MANUAL' (手動)
    RUN_MODE = 'MANUAL' 

    MANUAL_PARAMS = {
        'S_H': 5, 'RE_DAYS': 15, 'EXIT_MA': 15, 'OBV_RANK': 3, 'OBV_WINDOW': 15, 
        'IDX_MA1': 30, 'IDX_MA2': 60
    }

    try:
        loader = DataLoader(Config.DATA_FILE, Config.INDEX_FILE)
        close_data, volume_data, index_data = loader.load_data()
    except Exception as e:
        print(f"錯誤: 無法讀取資料 - {e}")
        return
    
    if RUN_MODE == 'ACO':
        optimizer = AntColonyOptimizer(close_data, volume_data, index_data)
        best_params = optimizer.run()
    else:
        print(f"使用手動參數: {MANUAL_PARAMS}")
        best_params = MANUAL_PARAMS
    
    print("\n使用參數產生最終報告...")
    final_strat = Strategy(close_data, volume_data, index_data, best_params)
    metrics = final_strat.run_backtest()
    
    print(f"儲存結果至 {Config.RESULT_FILE}...")
    try:
        with pd.ExcelWriter(Config.RESULT_FILE, engine='openpyxl') as writer:
            final_strat.trades_df.to_excel(writer, sheet_name='Trades')
            final_strat.equity_df.to_excel(writer, sheet_name='Equity_Curve')
            final_strat.daily_holdings_df.to_excel(writer, sheet_name='Equity_Hold')
            final_strat.daily_candidates_df.to_excel(writer, sheet_name='Daily_Candidate')
            final_strat.daily_candidates_df.to_excel(writer, sheet_name='Daily_Record')
            
            summary_data = {
                'Metric': ['CAGR', 'MaxDD', 'Calmar', 'Final Equity', 'Best Params'],
                'Value': [
                    metrics['CAGR'], metrics['MaxDD'], metrics['Calmar'], 
                    metrics['Final_Equity'], str(best_params)
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
        print("Excel 儲存完成。")
        
        if not final_strat.equity_df.empty:
            plt.figure(figsize=(10, 6))
            equity_series = final_strat.equity_df['Equity']
            plt.plot(equity_series.index, equity_series, label='Equity')
            cummax = equity_series.cummax()
            plt.fill_between(equity_series.index, equity_series, cummax, color='red', alpha=0.1, label='Drawdown Area')
            
            plt.title(f"Equity Curve (CAGR: {metrics['CAGR']:.2%})")
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.legend()
            plt.grid(True)
            output_png = os.path.join(Config.BASE_DIR, 'equity_curve.png')
            plt.savefig(output_png)
            print(f"權益曲線圖已儲存為 {output_png}")
        else:
            print("無交易數據，無法繪圖。")
            
    except Exception as e:
        print(f"儲存結果失敗: {e}")

if __name__ == '__main__':
    main()
