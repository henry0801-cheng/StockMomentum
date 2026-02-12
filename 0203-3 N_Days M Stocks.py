import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import warnings
import sys
from copy import deepcopy

# 忽略警告訊息
warnings.simplefilter(action='ignore', category=FutureWarning)

# 設定繁體中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 使用者設定區塊 (User Configuration Area)
# ==========================================

# 1. 資料檔路徑
DATA_FILE_PATH = r"cleaned_stock_data2.xlsx"

# 2. 模式選擇
# "MANUAL": 使用下方手動參數執行一次回測
# "ACO": 使用蟻群演算法尋找最佳參數 (並顯示前 N 名結果以觀察高原)
#RUN_MODE = "MANUAL" 
RUN_MODE = "ACO" 

# 3. 手動參數設定 (僅在 RUN_MODE="MANUAL" 時生效)
MANUAL_PARAMS = {
    'n_days': [2, 6, 8],  # 可以是單一整數 (e.g., 4) 或 列表 (e.g., [3, 5, 10])
    'top_k': 2,
    'v_bar': 9.0,          # 單位: 千萬
    'pct_stp': 0.06,       # 14%
    'bias60_bar': 0.8     # 60日乖離率濾網
}

# 4. ACO 設定 (僅在 RUN_MODE="ACO" 時生效)
ACO_SETTINGS = {
    'n_ants': 100,          # 螞蟻數量
    'n_iterations': 5,     # 迭代次數
    'top_n_results': 20,   # 顯示前幾名結果 (用於觀察參數高原)
    'ranges': {
        'n1': list(range(2, 10, 1)),      
        'n2': list(range(5, 15, 1)),      
        'n3': list(range(5, 21, 1)),      
        'top_k': list(range(2, 5, 1)),       
        'v_bar': list(range(7, 11, 2)),      
        'pct_stp': [i/100 for i in range(6, 10, 2)], 
        'bias60_bar': [i/100 for i in range(80, 85, 5)], # 0.10 ~ 0.50
    }
}

# ==========================================
# 策略邏輯 (Strategy Logic)
# ==========================================

class Strategy:
    def __init__(self, data_file, n_days, top_k, v_bar, pct_stp, bias60_bar, data=None):
        """
        初始化 N 日動能選股策略 (嚴格模式)
        """
        self.data_file = data_file
        self.n_days = n_days # Can be int or list
        self.top_k = int(top_k)
        self.v_bar = v_bar
        self.pct_stp = pct_stp
        self.bias60_bar = bias60_bar
        
        # 固定交易參數
        self.initial_capital = 20_000_000  # 初始本金 2000 萬
        self.tax_rate = 0.001              # 交易稅 0.1%
        self.slip_rate = 0.003             # 滑價 0.3% (買賣皆扣)
        
        # 載入資料
        if data is not None:
             self.data = data
        else:
             self.data = self.load_data()
        
    def load_data(self):
        """讀取並對齊股價與成交量資料"""
        if not os.path.exists(self.data_file):
            # Try finding the file in current directory if absolute path fails
            if os.path.exists(os.path.basename(self.data_file)):
                 self.data_file = os.path.basename(self.data_file)
            else:
                 raise FileNotFoundError(f"找不到檔案: {self.data_file}")
            
        xls = pd.ExcelFile(self.data_file)
        
        # 讀取 'P' (收盤價) 和 'Q' (成交量)
        df_p = pd.read_excel(xls, 'P', index_col=0, parse_dates=True)
        df_q = pd.read_excel(xls, 'Q', index_col=0, parse_dates=True)
        
        # 確保索引與欄位對齊
        common_cols = df_p.columns.intersection(df_q.columns)
        common_index = df_p.index.intersection(df_q.index)
        
        df_p = df_p.loc[common_index, common_cols]
        df_q = df_q.loc[common_index, common_cols]
        
        return {'P': df_p, 'Q': df_q}

    def run_backtest(self):
        """執行回測主邏輯"""
        P = self.data['P']
        Q = self.data['Q']
        dates = P.index
        
        # --- 1. 預先計算特徵 (Features) ---
        # 判斷 n_days 是一組還是一個
        if isinstance(self.n_days, list) or isinstance(self.n_days, tuple):
            # 計算多個週期的平均漲幅
            max_n = max(self.n_days)
            
            # 計算平均 Return
            sum_returns = pd.DataFrame(0, index=P.index, columns=P.columns)
            valid_periods = 0
            
            for n in self.n_days:
                if n > 0:
                    r = P.pct_change(n)
                    sum_returns = sum_returns.add(r, fill_value=0)
                    valid_periods += 1
            
            if valid_periods > 0:
                returns_n = sum_returns / valid_periods
            else:
                raise ValueError("n_days list must contain positive integers")
                
            self.max_lookback = max_n 
            
        else:
            # 單一週期 (向下相容)
            self.n_days = int(self.n_days)
            returns_n = P.pct_change(self.n_days)
            self.max_lookback = self.n_days

        # 成交金額 (元) = P * Q * 1000
        # 門檻轉換: V_Bar (千萬) -> 元
        turnover_value = P * Q * 1000

        v_bar_threshold = self.v_bar * 10_000_000
        
        # Calculate MA60 and Bias60 for trade analysis
        # Bias60 = (Price - MA60) / MA60
        ma60 = P.rolling(window=60).mean()
        bias60 = (P - ma60) / ma60
        
        # --- 2. 初始化回測變數 ---
        cash = self.initial_capital
        holdings = {} 
        
        equity_curve = []   # 每日權益曲線
        trades = []         # 交易紀錄
        daily_records = []  # 每日持倉統計
        daily_candidates = [] # 每日候選名單紀錄
        equity_hold = []    # 每日持股明細
        
        rebalance_counter = 0
        self.pending_sells = [] 
        self.pending_buys = []  
        
        # --- 3. 逐日回測迴圈 ---
        for i in range(len(dates)):
            today = dates[i]
            
            # 前 N 天無資料，跳過但記錄資金
            if i < self.max_lookback:
                equity_curve.append({'Date': today, 'Equity': cash})
                equity_hold.append({'Date': today, 'Count': 0, 'Details': str({})})
                continue
            
            # 當日價格數據
            todays_prices = P.iloc[i]
            yesterday_prices = P.iloc[i-1] if i > 0 else None
            
            # Step A: 計算當前權益
            current_equity = cash
            current_holdings_info = {}
            for ticker, info in list(holdings.items()):
                if ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                     current_price = todays_prices[ticker]
                     market_value = info['shares'] * current_price
                     current_equity += market_value
                     current_holdings_info[ticker] = info['shares']
            
            equity_curve.append({'Date': today, 'Equity': current_equity})
            equity_hold.append({'Date': today, 'Count': len(holdings), 'Details': str(current_holdings_info)})

            # Step B: 執行「待賣出」訂單
            money_returned = 0
            for ticker in self.pending_sells:
                if ticker in holdings and ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                    price = todays_prices[ticker]
                    
                    # 漲跌停保護
                    if yesterday_prices is not None and ticker in yesterday_prices:
                        prev_close = yesterday_prices[ticker]
                        if prev_close > 0 and abs((price - prev_close) / prev_close) > 0.096:
                            continue 
                    
                    # 執行賣出
                    shares = holdings[ticker]['shares']
                    revenue = shares * price * (1 - self.slip_rate - self.tax_rate)
                    money_returned += revenue
                    
                    # 記錄交易
                    
                    # Calculate Return Rate (報酬率)
                    entry_price = holdings[ticker]['entry_price']
                    # cost = shares * entry_price * (1 + slip)
                    # revenue = shares * price * (1 - slip - tax)
                    # net_pnl = revenue - cost
                    # return_rate = net_pnl / cost
                    
                    cost = shares * entry_price * (1 + self.slip_rate)
                    net_pnl = revenue - cost
                    return_rate = net_pnl / cost if cost > 0 else 0
                    
                    # Get Entry Bias60
                    entry_date = holdings[ticker]['entry_date']
                    entry_bias60 = np.nan
                    if entry_date in bias60.index and ticker in bias60.columns:
                        entry_bias60 = bias60.loc[entry_date, ticker]

                    trades.append({
                        'Ticker': ticker,
                        'BuyDate': holdings[ticker]['entry_date'],
                        'SellDate': today,
                        'BuyPrice': entry_price,
                        'SellPrice': price,
                        'Shares': shares,
                        'PnL': net_pnl,
                        'Return': return_rate,
                        'EntryBias60': entry_bias60,
                        'Reason': holdings[ticker].get('exit_reason', 'Rebalance')
                    })
                    del holdings[ticker]
            
            cash += money_returned
            self.pending_sells = [] 

            # Step C: 執行「待買入」訂單
            for order in self.pending_buys:
                if len(holdings) >= self.top_k:
                    break
                
                ticker = order['ticker']
                budget = order['budget']
                
                if ticker in holdings: continue
                
                if ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                    price = todays_prices[ticker]
                    
                    # 漲跌停保護
                    if yesterday_prices is not None and ticker in yesterday_prices:
                        prev_close = yesterday_prices[ticker]
                        if prev_close > 0 and abs((price - prev_close) / prev_close) > 0.096:
                            continue 
                    
                    cost_per_share = price * (1 + self.slip_rate)
                    
                    if cash >= cost_per_share * 1000: 
                        target_shares = int(budget / cost_per_share)
                        max_shares_by_cash = int(cash / cost_per_share)
                        shares_to_buy = min(target_shares, max_shares_by_cash)
                        
                        if shares_to_buy > 0:
                            actual_cost = shares_to_buy * cost_per_share
                            cash -= actual_cost
                            holdings[ticker] = {
                                'shares': shares_to_buy,
                                'entry_price': price,
                                'entry_date': today,
                                'exit_reason': ''
                            }
            self.pending_buys = [] 

            # Step D: 檢查出場訊號
            next_sells = []
            
            for ticker in list(holdings.keys()):
                if ticker not in todays_prices: continue
                price = todays_prices[ticker]
                
                if price < holdings[ticker]['entry_price'] * (1 - self.pct_stp):
                    holdings[ticker]['exit_reason'] = 'StopLoss'
                    if ticker not in next_sells: next_sells.append(ticker)
            
            # Step E: 再平衡訊號產生
            next_buys = []
            target_tickers = []
            
            is_rebalance_day = (rebalance_counter % 5 == 0)
            if is_rebalance_day:
                vals = turnover_value.loc[today]
                liquid_tickers = vals[vals > v_bar_threshold].index
                
                candidates = returns_n.loc[today, liquid_tickers].dropna()
                
                # Apply Bias60 Filter: Exclude if Bias60 > bias60_bar (if data exists)
                if bias60 is not None and len(bias60) > 0:
                    current_bias = bias60.loc[today, candidates.index]
                    # Valid candidates: NaN bias OR bias <= bias_bar
                    valid_mask = (current_bias.isna()) | (current_bias <= self.bias60_bar)
                    # Filter candidates
                    candidates = candidates[valid_mask]
                
                target_tickers = candidates.sort_values(ascending=False).head(self.top_k).index.tolist()
                
                for ticker in holdings:
                    if ticker not in target_tickers:
                        if ticker not in next_sells:
                            holdings[ticker]['exit_reason'] = 'RebalanceOut'
                            next_sells.append(ticker)
                
                slot_size = self.initial_capital / self.top_k
                for ticker in target_tickers:
                    if ticker not in holdings and ticker not in next_sells:
                        next_buys.append({'ticker': ticker, 'budget': slot_size})
            
            self.pending_sells = next_sells
            self.pending_buys = next_buys
            rebalance_counter += 1
            
            daily_candidates.append({'Date': today, 'Count': len(target_tickers), 'Tickers': str(target_tickers)})
            daily_records.append({
                'Date': today,
                'Held': len(holdings),
                'PendingSells': len(next_sells),
                'PendingBuys': len(next_buys)
            })

        # --- 4. 計算績效指標 ---
        total_days = (dates[-1] - dates[0]).days
        years = total_days / 365.25
        final_equity = equity_curve[-1]['Equity']
        cagr = (final_equity / self.initial_capital) ** (1/years) - 1 if years > 0 and final_equity > 0 else 0
        
        eq_series = pd.DataFrame(equity_curve).set_index('Date')['Equity']
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak
        max_dd = dd.min()
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        wins = len([t for t in trades if t['PnL'] > 0])
        win_rate = wins / len(trades) if len(trades) > 0 else 0
        
        results = {
            'CAGR': cagr,
            'MaxDD': max_dd,
            'Calmar': calmar,
            'WinRate': win_rate,
            'FinalEquity': final_equity,
            'Trades': pd.DataFrame(trades),
            'EquityCurve': pd.DataFrame(equity_curve),
            'EquityHold': pd.DataFrame(equity_hold),
            'DailyRecord': pd.DataFrame(daily_records),
            'DailyCandidate': pd.DataFrame(daily_candidates)
        }
        
        return results

    def save_results(self, results, filename='strategy_results.xlsx'):
        """儲存 Excel 結果與繪製權益曲線"""
        with pd.ExcelWriter(filename) as writer:
            summary = pd.DataFrame([{
                'CAGR': f"{results['CAGR']:.2%}",
                'MaxDD': f"{results['MaxDD']:.2%}",
                'Calmar': f"{results['Calmar']:.2f}",
                'WinRate': f"{results['WinRate']:.2%}",
                'FinalEquity': f"{results['FinalEquity']:,.0f}",
                'N_DAYS': str(self.n_days),
                'TOP_K': self.top_k,
                'V_BAR': self.v_bar,
                'PCT_STP': self.pct_stp,
                'BIAS60_BAR': self.bias60_bar
            }])
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            if not results['Trades'].empty:
                # Format Return and EntryBias60 as percentages
                df_trades = results['Trades'].copy()
                if 'Return' in df_trades.columns:
                    df_trades['Return'] = df_trades['Return'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                if 'EntryBias60' in df_trades.columns:
                    df_trades['EntryBias60'] = df_trades['EntryBias60'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                    
                df_trades.to_excel(writer, sheet_name='Trades', index=False)
            
            results['EquityCurve'].to_excel(writer, sheet_name='Equity_Curve', index=False)
            results['EquityHold'].to_excel(writer, sheet_name='Equity_Hold', index=False)
            results['DailyRecord'].to_excel(writer, sheet_name='Daily_Record', index=False)
            results['DailyCandidate'].to_excel(writer, sheet_name='Daily_Candidate', index=False)

        # 繪圖
        df_eq = results['EquityCurve']
        if not df_eq.empty:
            df_eq['Date'] = pd.to_datetime(df_eq['Date'])
            df_eq = df_eq.set_index('Date')
            
            plt.figure(figsize=(12, 6))
            plt.plot(df_eq['Equity'], label='Equity', color='blue')
            
            peak = df_eq['Equity'].cummax()
            plt.fill_between(df_eq.index, peak, df_eq['Equity'], color='red', alpha=0.1, label='Drawdown Area')
            
            plt.title('Equity Curve (權益曲線)')
            plt.legend()
            plt.grid(True)
            plt.savefig('equity_curve.png')
            plt.close()

# ==========================================
# 蟻群演算法 (ACO Logic)
# ==========================================

class AntColonyOptimizer:
    def __init__(self, data_file, settings):
        self.data_file = data_file
        self.settings = settings
        self.param_ranges = settings['ranges']
        self.keys = list(self.param_ranges.keys())
        self.pheromones = {
            k: {v: 1.0 for v in vals} for k, vals in self.param_ranges.items()
        }
        self.n_ants = settings['n_ants']
        self.n_iterations = settings['n_iterations']
        self.evaporation_rate = 0.2
        self.alpha = 1.0
        self.best_calmar = -np.inf
        self.best_fitness = -np.inf
        self.best_params = None
        self.best_result = None
        
        # 用於記錄所有結果以分析高原
        self.all_results = []

    def select_param(self, key):
        """費洛蒙輪盤選擇法"""
        vals = self.param_ranges[key]
        pheros = [self.pheromones[key][v] ** self.alpha for v in vals]
        total = sum(pheros)
        probs = [p/total for p in pheros]
        return random.choices(vals, weights=probs, k=1)[0]

    def run(self):
        print(f"開始 ACO 最佳化... 螞蟻: {self.n_ants}, 迭代: {self.n_iterations}")
        
        print("Loading data...")
        # 建立一個臨時策略物件來載入主要資料，避免重複讀取 IO
        temp_strat = Strategy(self.data_file, 10, 1, 9.0, 0.1, 0.2)
        shared_data = temp_strat.data
        
        for iteration in range(self.n_iterations):
            print(f"Iteration {iteration + 1}/{self.n_iterations}...")
            iteration_best_fitness = -np.inf
            iteration_best_ant = None
            
            for i in range(self.n_ants):
                # Retry loop to ensure n1, n2, n3 are distinct
                for _ in range(100):
                    params = {k: self.select_param(k) for k in self.keys}
                    if len({params['n1'], params['n2'], params['n3']}) == 3:
                        break
                else:
                    # If failed to find distinct values, force distinct by adding offsets
                    # (This is a fallback to avoid infinite loops if pheromones converge excessively)
                    ns = sorted(list({params['n1'], params['n2'], params['n3']}))
                    while len(ns) < 3:
                        new_val = (ns[-1] + 1) if ns else 2
                        if new_val not in ns: ns.append(new_val)
                        else: ns.append(new_val + 1)
                    params['n1'], params['n2'], params['n3'] = ns[0], ns[1], ns[2]

                # Sort n1, n2, n3 to ensure n1 < n2 < n3
                ns = sorted([params['n1'], params['n2'], params['n3']])
                params['n1'] = ns[0]
                params['n2'] = ns[1]
                params['n3'] = ns[2]
                n_days_list = [params['n1'], params['n2'], params['n3']]
                
                strat = Strategy(
                    self.data_file, 
                    n_days=n_days_list,
                    top_k=params['top_k'],
                    v_bar=params['v_bar'],
                    pct_stp=params['pct_stp'],
                    bias60_bar=params['bias60_bar'],
                    data=shared_data
                )
                
                try:
                    res = strat.run_backtest()
                    calmar = res['Calmar']
                    cagr = res['CAGR']
                    # Using a composite fitness score: Calmar + 5 * CAGR to value both
                    # (since Calmar is often > 1 and CAGR < 1, scaling CAGR makes them more comparable)
                    fitness = calmar + (cagr * 5)
                except Exception as e:
                    calmar = -999
                    cagr = -999
                    fitness = -999
                
                # 記錄結果
                record = params.copy()
                record['Fitness'] = fitness
                record['Calmar'] = calmar
                record['CAGR'] = cagr
                record['MaxDD'] = res['MaxDD']
                self.all_results.append(record)
                
                if fitness > iteration_best_fitness:
                    iteration_best_fitness = fitness
                    iteration_best_ant = params
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_calmar = calmar
                    self.best_params = params
                    self.best_result = res
                    print(f"  New Best! Fitness: {self.best_fitness:.2f} | Calmar: {calmar:.2f} | CAGR: {cagr:.2%} | Params: {self.best_params}")
            
            # 更新費洛蒙
            for k in self.keys:
                for v in self.pheromones[k]:
                    self.pheromones[k][v] *= (1 - self.evaporation_rate)
            
            deposit = 1.0 + max(0, iteration_best_fitness)
            if iteration_best_ant:
                for k, v in iteration_best_ant.items():
                    self.pheromones[k][v] += deposit

        print(f"\n最佳化完成。全域最佳 Fitness: {self.best_fitness:.2f}, Calmar: {self.best_calmar:.2f}")
        
        # 儲存最佳結果
        if self.best_result:
            final_strat = Strategy(
                self.data_file, 
                n_days=[self.best_params['n1'], self.best_params['n2'], self.best_params['n3']],
                top_k=self.best_params['top_k'],
                v_bar=self.best_params['v_bar'],
                pct_stp=self.best_params['pct_stp'],
                bias60_bar=self.best_params['bias60_bar'],
                data=shared_data
            )
            # 儲存 Excel
            final_strat.save_results(self.best_result, filename='strategy_aco_best.xlsx')
            
            # 儲存高原分析 (Top N 結果)
            df_all = pd.DataFrame(self.all_results)
            df_all = df_all.sort_values(by='Calmar', ascending=False).head(self.settings['top_n_results'])
            df_all.to_excel('strategy_aco_plateau.xlsx', index=False)
            print(f"Top {self.settings['top_n_results']} 結果已儲存至 strategy_aco_plateau.xlsx (可用於觀察參數高原)")
            print("最佳單一結果已儲存至 strategy_aco_best.xlsx")

# ==========================================
# 主程式 (Main Entry)
# ==========================================

if __name__ == "__main__":
    
    print(f"目前執行模式: {RUN_MODE}")
    print(f"資料檔: {DATA_FILE_PATH}")
    
    if RUN_MODE == "MANUAL":
        print(f"使用手動參數: {MANUAL_PARAMS}")
        
        if not os.path.exists(DATA_FILE_PATH):
            print(f"錯誤: 找不到資料檔 {DATA_FILE_PATH}")
        else:
            strat = Strategy(
                DATA_FILE_PATH,
                n_days=MANUAL_PARAMS['n_days'],
                top_k=MANUAL_PARAMS['top_k'],
                v_bar=MANUAL_PARAMS['v_bar'],
                pct_stp=MANUAL_PARAMS['pct_stp'],
                bias60_bar=MANUAL_PARAMS['bias60_bar']
            )
            res = strat.run_backtest()
            strat.save_results(res, filename='strategy_manual_results.xlsx')
            print(f"手動回測完成! 結果已儲存至 strategy_manual_results.xlsx")
            print(f"Calmar: {res['Calmar']:.2f}, CAGR: {res['CAGR']:.2%}, MaxDD: {res['MaxDD']:.2%}")

    elif RUN_MODE == "ACO":
        if not os.path.exists(DATA_FILE_PATH):
            print(f"錯誤: 找不到資料檔 {DATA_FILE_PATH}")
        else:
            optimizer = AntColonyOptimizer(DATA_FILE_PATH, ACO_SETTINGS)
            optimizer.run()
    
    else:
        print("未知的 RUN_MODE，請設定為 'MANUAL' 或 'ACO'")
