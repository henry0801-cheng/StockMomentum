import pandas as pd
import numpy as np
import os
import time
import random
import warnings
import matplotlib.pyplot as plt

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
    TAX_RATE = 0.001    # 交易稅 0.1%
    SLIPPAGE = 0.003    # 滑價 0.3%
    
    # ACO 參數
    ANT_COUNT = 5      # 螞蟻數量
    GENERATIONS = 3      # 世代數
    EVAPORATION = 0.5    # 費洛蒙揮發率
    ALPHA = 1.0          # 費洛蒙重要性因子
    
    # 參數範圍 (用於 ACO 探索)
    # S_H: 最大持倉檔數
    # P_BAR: 11日內創50天新高次數門檻
    # BIAS_60_MAX: 60日乖離率上限 (%)
    # EXIT_MA: 出場均線天數
    PARAM_RANGES = {
        'S_H': [5],
        'P_BAR': [5,6,7,8],
        'BIAS_60_MAX': [80, 85],
        'EXIT_MA': [10,13,15,17]
    }

# ==========================================
# 資料讀取類別 (Data Loader)
# ==========================================
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.adj_close = None
        self.volume = None
        
    def load_data(self):
        print("讀取資料中... (Loading Data)")
        
        # 嘗試解析路徑
        if not os.path.exists(self.filepath):
            # 1. 嘗試相對於腳本的路徑
            script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
            p1 = os.path.join(script_dir, self.filepath)
            
            # 2. 嘗試當前工作目錄
            p2 = os.path.join('.', self.filepath)
            
            if os.path.exists(p1):
                self.filepath = p1
            elif os.path.exists(p2):
                self.filepath = p2
            else:
                # 3. 嘗試上一層 (假設在子目錄執行)
                p3 = os.path.join('..', self.filepath)
                if os.path.exists(p3):
                    self.filepath = p3
                else:
                    print(f"警告: 找不到檔案 {self.filepath}，嘗試在 {script_dir} 尋找失敗。")

        xls = pd.ExcelFile(self.filepath)
        
        # 讀取 P (還原收盤價), Q (成交量)
        valid_sheets = [s for s in xls.sheet_names if s in ['P', 'Q']]
        if 'P' not in valid_sheets or 'Q' not in valid_sheets:
            raise ValueError("Excel 缺少 P 或 Q 工作表")

        self.adj_close = pd.read_excel(xls, 'P', index_col=0)
        self.volume = pd.read_excel(xls, 'Q', index_col=0)
        
        # 確保索引是 DateTime
        self.adj_close.index = pd.to_datetime(self.adj_close.index)
        self.volume.index = pd.to_datetime(self.volume.index)
        
        # 確保欄位一致
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
        self.volume = data_volume
        self.params = params
        
        # 參數解包
        self.S_H = params['S_H']
        self.P_BAR = params['P_BAR']
        self.BIAS_60_MAX = params['BIAS_60_MAX']
        self.EXIT_MA = params['EXIT_MA']
        
        # 計算指標
        self.indicators = {}
        self._calculate_indicators()
        
    def _calculate_indicators(self):
        # 1. 11日內創50天新高的次數
        # 50天新高: 今日收盤價 == 過去50天(含今日)的最高價
        rolling_max_50 = self.close.rolling(window=50, min_periods=50).max()
        is_high_50 = (self.close == rolling_max_50)
        
        # 計算11日內次數 (rolling sum)
        self.indicators['count_high_50'] = is_high_50.rolling(window=11, min_periods=11).sum()
        
        # 2. 60日乖離率 (Bias60)
        ma60 = self.close.rolling(window=60, min_periods=60).mean()
        self.indicators['bias60'] = ((self.close - ma60) / ma60) * 100
        
        # 3. 出場均線 (EXIT_MA)
        # 根據參數計算對應的 MA
        self.indicators['exit_ma'] = self.close.rolling(window=self.EXIT_MA, min_periods=self.EXIT_MA).mean()
        
    def run_backtest(self):
        # 初始化回測變數
        capital = Config.INITIAL_CAPITAL
        # 每檔資金上限 (固定)
        position_limit = capital / self.S_H
        
        positions = {} # {ticker: {'shares': int, 'cost': float, 'entry_date': date}}
        
        # 紀錄
        equity_curve = [] 
        trades = [] 
        daily_holdings = [] 
        daily_candidates = [] 
        
        dates = self.close.index
        # 從最大 days 開始，避免指標 NaN
        start_idx = max(60, self.EXIT_MA) 
        
        for i in range(start_idx, len(dates) - 1):
            t_date = dates[i]
            next_date = dates[i+1] # T+1 日，執行交易
            
            # --- T日 資料與訊號 ---
            current_close = self.close.loc[t_date]
            current_count_high = self.indicators['count_high_50'].loc[t_date]
            current_bias60 = self.indicators['bias60'].loc[t_date]
            current_exit_ma = self.indicators['exit_ma'].loc[t_date]
            
            # --- 賣出訊號檢查 (T日確認) ---
            # 賣出條件: 股價跌破 EXIT_MA
            # 對於已持有的股票檢查
            stocks_to_sell = []
            
            for ticker in list(positions.keys()):
                price = current_close.get(ticker)
                ma_val = current_exit_ma.get(ticker)
                
                if pd.isna(price) or pd.isna(ma_val):
                    continue
                    
                if price < ma_val:
                    stocks_to_sell.append((ticker, f'Exit: Price < MA{self.EXIT_MA}'))
            
            # --- 買進訊號檢查 (T日確認) (Vectorized Optimization) ---
            # 買進條件:
            # (1) 11日內創50天新高次數 > P_BAR
            # (2) 60天乖離小於 BIAS_60_MAX
            
            # 1. 篩選符合條件的股票 mask
            mask = (current_count_high > self.P_BAR) & (current_bias60 < self.BIAS_60_MAX)
            mask = mask.fillna(False)
            
            valid_tickers = mask.index[mask]
            
            # 2. 建立候選 DF 以便排序
            if len(valid_tickers) > 0:
                cand_cnt = current_count_high[valid_tickers]
                cand_bias = current_bias60[valid_tickers]
                cand_df = pd.DataFrame({'cnt': cand_cnt, 'bias': cand_bias})
                
                # 3. 移除已持有 (雖然後面會 check，但在排序前移除更有效率)
                held_tickers = list(positions.keys())
                cand_df = cand_df.drop(held_tickers, errors='ignore')
                
                # 4. 排序: 次數高(desc) > 乖離低(asc)
                cand_df = cand_df.sort_values(by=['cnt', 'bias'], ascending=[False, True])
                
                candidate_tickers = cand_df.index.tolist()
            else:
                candidate_tickers = []
            
            daily_candidates.append({
                'Date': t_date, 
                'Count': len(candidate_tickers), 
                'Candidates': str(candidate_tickers[:10]) 
            })
            
            # --- T+1日 執行交易 ---
            next_close = self.close.loc[next_date]
            
            # 1. 執行賣出
            for ticker, reason in stocks_to_sell:
                if ticker in positions:
                    pos = positions[ticker]
                    exec_price = next_close.get(ticker)
                    
                    if pd.isna(exec_price) or exec_price == 0:
                        continue
                        
                    # 賣出價格 (滑價)
                    sell_price = exec_price * (1 - Config.SLIPPAGE)
                    # 稅
                    tax = sell_price * pos['shares'] * Config.TAX_RATE
                    
                    revenue = sell_price * pos['shares'] - tax
                    pnl = revenue - (pos['cost'] * pos['shares'])
                    ret = (sell_price - pos['cost']) / pos['cost']
                    
                    capital += revenue
                    del positions[ticker]
                    
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
            
            # 2. 執行買進
            # 檢查剩餘倉位
            available_slots = self.S_H - len(positions)
            
            if available_slots > 0 and candidate_tickers:
                for ticker in candidate_tickers:
                    if available_slots <= 0:
                        break
                    
                    exec_price = next_close.get(ticker)
                    if pd.isna(exec_price) or exec_price == 0:
                        continue
                        
                    # 買進價格 (滑價)
                    buy_price = exec_price * (1 + Config.SLIPPAGE)
                    
                    # 資金計算
                    budget = min(capital, position_limit)
                    
                    if budget < buy_price:
                        continue 
                        
                    shares = int(budget // buy_price)
                    if shares == 0:
                        continue
                        
                    cost_amt = shares * buy_price
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
                        'Reason': 'Signal Entry',
                        'PnL': 0,
                        'Return': 0
                    })
                    
                    available_slots -= 1
            
            # --- 結算 T+1 權益 ---
            curr_equity = capital
            holdings_info = []
            for t, pos in positions.items():
                p = next_close.get(t)
                if pd.isna(p): p = pos['cost']
                mv = p * pos['shares']
                curr_equity += mv
                holdings_info.append(f"{t}({pos['shares']})")
                
            equity_curve.append({'Date': next_date, 'Equity': curr_equity})
            daily_holdings.append({
                'Date': next_date, 
                'Holdings_Count': len(positions), 
                'Details': str(holdings_info)
            })
            
        # 整理結果
        self.equity_df = pd.DataFrame(equity_curve).set_index('Date')
        self.trades_df = pd.DataFrame(trades)
        self.daily_holdings_df = pd.DataFrame(daily_holdings)
        self.daily_candidates_df = pd.DataFrame(daily_candidates)
        
        return self._calculate_metrics()

    def _calculate_metrics(self):
        if self.equity_df.empty:
            return {'CAGR': -0.99, 'Calmar': -99.0, 'MaxDD': -1.0, 'Final_Equity': 0}
            
        initial = Config.INITIAL_CAPITAL
        final = self.equity_df['Equity'].iloc[-1]
        
        # Years
        start_date = self.equity_df.index[0]
        end_date = self.equity_df.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        
        if years <= 0: return {'CAGR': -0.99, 'Calmar': -99.0, 'MaxDD': -1.0, 'Final_Equity': final}

        # CAGR
        if final <= 0:
            cagr = -1.0
        else:
            cagr = (final / initial) ** (1 / years) - 1
        
        # MaxDD
        equity = self.equity_df['Equity']
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min() # negative
        
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
        # 初始化費洛蒙
        self.pheromones = {}
        for param, values in Config.PARAM_RANGES.items():
            self.pheromones[param] = {v: 1.0 for v in values}
            
    def _select_value(self, param):
        values = Config.PARAM_RANGES[param]
        probs = [self.pheromones[param][v] ** Config.ALPHA for v in values]
        total = sum(probs)
        if total == 0:
             probs = [1.0/len(probs)] * len(probs)
        else:
             probs = [p / total for p in probs]
        return np.random.choice(values, p=probs)

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
                for k in Config.PARAM_RANGES.keys():
                    params[k] = self._select_value(k)
                
                # 執行策略
                strat = Strategy(self.close, self.volume, params)
                metrics = strat.run_backtest()
                fitness = metrics['CAGR']
                
                ants_results.append((params, fitness, metrics))
                
                # 更新世代最佳
                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_ant = params
                    
                # 更新全域最佳
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = params
                    print(f"  [新紀錄] Ant {ant}: CAGR={fitness:.2%}, Calmar={metrics['Calmar']:.2f}, Params={params}")

            # 費洛蒙更新
            # 1. 蒸發
            for param in self.pheromones:
                for v in self.pheromones[param]:
                    self.pheromones[param][v] *= (1 - Config.EVAPORATION)
            
            # 2. 堆積 (菁英策略: 前50% + 全域最佳)
            sorted_ants = sorted(ants_results, key=lambda x: x[1], reverse=True)
            top_n = max(1, len(sorted_ants)//2)
            top_ants = sorted_ants[:top_n]
            
            # 額外獎勵全域最佳
            if self.best_solution:
                 for k, v in self.best_solution.items():
                    self.pheromones[k][v] += 0.5 # 額外加強
            
            for params, fitness, _ in top_ants:
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
    print(f"儲存結果至 {Config.RESULT_FILE}...")
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
        
    print("Excel 輸出完成。")
    
    # 畫權益曲線圖
    plt.figure(figsize=(12, 6))
    plt.plot(final_strat.equity_df.index, final_strat.equity_df['Equity'], label='Equity', color='blue')
    
    # 畫回撤陰影
    equity = final_strat.equity_df['Equity']
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    
    # 雙軸: 右軸畫回撤? 還是陰影? 
    # 題目要求 "含回撤陰影"。通常指在下方或背景。
    # 這裡用 fill_between
    plt.fill_between(final_strat.equity_df.index, equity, cummax, color='gray', alpha=0.3, label='Drawdown Area')

    plt.title(f"Equity Curve (CAGR: {metrics['CAGR']:.2%}, Calmar: {metrics['Calmar']:.2f})")
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.savefig('equity_curve.png')
    print("權益曲線圖已儲存為 equity_curve.png")

if __name__ == '__main__':
    main()
