import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings

# 忽略 FutureWarning 以保持輸出乾淨
warnings.simplefilter(action='ignore', category=FutureWarning)

# 設定繁體中文顯示 (避免亂碼)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 參數定義 (可用於獨立執行與測試)
# ==========================================
# 資金管理
INITIAL_CAPITAL = 20000000  # 總資金 2000萬
TAX_RATE = 0.001            # 交易稅 0.1%
SLIPPAGE = 0.003            # 滑價 0.3%

# 檔案路徑
DATA_FILE = 'cleaned_stock_data1.xlsx'
RESULT_FILE = 'strategy_results.xlsx'

# ==========================================
# 1. 資料讀取與處理
# ==========================================
def load_data(filepath):
    """
    讀取 Excel 檔案中的 P (價格), Q (成交量), F (外資) 資料表
    """
    print(f"正在讀取資料: {filepath} ...")
    xls = pd.ExcelFile(filepath)
    df_adj_close = pd.read_excel(xls, 'P', index_col=0, parse_dates=True)
    df_volume = pd.read_excel(xls, 'Q', index_col=0, parse_dates=True)
    # df_foreign = pd.read_excel(xls, 'F', index_col=0, parse_dates=True) # 策略邏輯目前未使用 F 表，但保留讀取介面

    # 確保索引格式為 Datetime
    df_adj_close.index = pd.to_datetime(df_adj_close.index)
    df_volume.index = pd.to_datetime(df_volume.index)
    
    print("資料讀取完成。")
    return df_adj_close, df_volume

def calculate_rsi(series, period=14):
    """
    計算 RSI 指標
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # 避免除以零
    loss = loss.replace(0, np.nan) 
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 對於 loss 為 0 的情況 (即 RS 無限大)，RSI 設為 100
    rsi = rsi.fillna(100)
    
    # 處理開頭 NaNs
    return rsi

# ==========================================
# 2. 策略邏輯類別 (Strategy)
# ==========================================
class Strategy:
    def __init__(self, data_p, data_q, params):
        self.data_p = data_p
        self.data_q = data_q
        self.params = params
        
        # 解包參數
        self.N_Check = int(params['N_Check'])
        self.M_RSI = int(params['M_RSI'])
        self.K = int(params['K'])
        self.S_H = int(params['S_H'])
        self.V_Bar = params['V_Bar'] * 10000000 # 換算成元 (輸入單位為千萬)
        self.Y_Check = int(params['Y_Check'])
        
        # 計算指標
        self.rsi_m = data_p.apply(lambda x: calculate_rsi(x, self.M_RSI))
        self.rsi_14 = data_p.apply(lambda x: calculate_rsi(x, 14)) # 用於排序
        self.avg_vol_20 = data_q.rolling(window=20).mean() # 20日均量

    def run_backtest(self):
        """
        執行回測
        """
        dates = self.data_p.index
        stocks = self.data_p.columns
        
        capital = INITIAL_CAPITAL
        position_limit = capital / self.S_H # 單檔資金上限
        
        # 帳戶狀態
        cash = capital
        positions = {} # {stock: shares}
        
        # 紀錄
        trades = []
        equity_curve = []
        daily_holdings = []
        daily_candidates_log = []
        daily_record = [] # 進場與持倉檢查紀錄

        # 預計算 RSI > 70 的布林值矩陣 (False/True)
        rsi_gt_70 = self.rsi_m > 70
        
        # 預計算 N_Check 與 Y_Check 天數內的次數
        # 使用 rolling sum 計算過去 N/Y 天的 True 次數
        # shift(0) 包含當天 T
        rsi_count_n = rsi_gt_70.rolling(window=self.N_Check).sum()
        rsi_count_y = rsi_gt_70.rolling(window=self.Y_Check).sum()
        
        # 開始逐日回測
        # 從最大視窗開始，避免 NaN
        start_idx = max(self.M_RSI, 20, self.N_Check, self.Y_Check) 
        
        for t in range(start_idx, len(dates) - 1): # -1 因為要用到 T+1 買賣
            date_t = dates[t]
            date_next = dates[t+1]
            
            # --- 1. 更新資產價值 (Mark to Market) ---
            current_equity = cash
            todays_holdings = {}
            for stock, shares in positions.items():
                price_t = self.data_p.at[date_t, stock]
                if pd.isna(price_t): # 如果當日無價格 (如下市)，沿用前值或跳過 (簡化處理)
                    continue 
                current_equity += shares * price_t
                todays_holdings[stock] = shares
            
            equity_curve.append({'Date': date_t, 'Equity': current_equity})
            daily_holdings.append({'Date': date_t, 'Holdings': todays_holdings.copy(), 'Count': len(todays_holdings)})

            # --- 2. 出場檢查 (Exit Signal) ---
            # 條件: 最近 Y_Check 日 (含 T)，[M_RSI日 RSI] > 70 次數 < K
            # 於 T+1 日收盤價賣出
            stocks_to_sell = []
            for stock in list(positions.keys()):
                count_y = rsi_count_y.at[date_t, stock]
                
                # 檢查是否有數據 (非 NaN)
                if pd.isna(count_y): 
                    continue

                if count_y < self.K:
                    stocks_to_sell.append(stock)
            
            # 執行賣出
            for stock in stocks_to_sell:
                price_next = self.data_p.at[date_next, stock]
                if pd.isna(price_next): continue # 無法交易

                shares = positions[stock]
                amount = shares * price_next
                fee = amount * TAX_RATE
                slippage = amount * SLIPPAGE
                net_amount = amount - fee - slippage
                
                cash += net_amount
                
                # 紀錄交易
                # 計算損益需追蹤買入成本 (為簡化，此處只記賣出，損益需與買入配對，或後續處理)
                # 這裡簡單紀錄單筆賣出資訊
                trades.append({
                    'Date': date_next,
                    'Ticker': stock,
                    'Action': 'Sell',
                    'Price': price_next,
                    'Shares': shares,
                    'Amount': net_amount,
                    'Reason': f'RSI>{self.M_RSI} count({int(rsi_count_y.at[date_t, stock])}) < K({self.K}) in {self.Y_Check} days'
                })
                del positions[stock]

            # --- 3. 進場檢查 (Entry Signal) ---
            # 條件 1: 最近 N_Check 日 (含 T)，[M_RSI日 RSI] > 70 次數 > K
            # 條件 2: 20日均量 * T日收盤價 * 1000 > V_Bar
            
            # 先篩選符合基本條件的候選股
            candidates = []
            
            # 目前持倉數
            current_holdings_count = len(positions)
            available_slots = self.S_H - current_holdings_count
            
            if available_slots > 0:
                # 批次篩選以提升效能
                # 1. 次數條件
                cond1 = rsi_count_n.loc[date_t] > self.K
                
                # 2. 流動性條件
                # 成交量單位為張 (1000股)，價格單位為元
                # 20日均量 * T日收盤價 * 1000 > V_Bar
                avg_vol = self.avg_vol_20.loc[date_t]
                price_t = self.data_p.loc[date_t]
                liquidity = avg_vol * price_t * 1000
                cond2 = liquidity > self.V_Bar
                
                # 排除已持有的股票
                # cond3 = ~cond1.index.isin(positions.keys()) # 邏輯上要篩選所有股票，然後排除已持有
                
                valid_stocks_mask = cond1 & cond2
                valid_stocks = valid_stocks_mask[valid_stocks_mask].index.tolist()
                
                # 排除已持倉
                final_candidates = [s for s in valid_stocks if s not in positions]

                # 排序: 依 14日 RSI 由大到小
                candidates_rsi14 = self.rsi_14.loc[date_t, final_candidates]
                sorted_candidates = candidates_rsi14.sort_values(ascending=False).index.tolist()
                
                daily_candidates_log.append({
                    'Date': date_t,
                    'Candidates': sorted_candidates,
                    'Count': len(sorted_candidates)
                })

                # 買進
                # 當符合條件的股票數量，大於可買進的數量時，依14日RSI由大到小排序選擇。
                buy_list = sorted_candidates[:available_slots]
                
                for stock in buy_list:
                    price_next = self.data_p.at[date_next, stock]
                    if pd.isna(price_next): continue

                    # 計算可買股數
                    invest_amount = position_limit
                    if cash < invest_amount:
                        invest_amount = cash # 資金不足時用剩餘資金
                    
                    if invest_amount <= 0: break 

                    # 考慮滑價後的成本
                    # 買入成本 = 股數 * 價格 * (1 + 滑價)
                    # 股數 = 資金 / (價格 * (1 + 滑價))
                    max_shares = int(invest_amount / (price_next * (1 + SLIPPAGE)))
                    if max_shares == 0: continue
                    
                    cost = max_shares * price_next
                    slippage_cost = cost * SLIPPAGE
                    total_cost = cost + slippage_cost # 買入無交易稅
                    
                    if cash >= total_cost:
                        cash -= total_cost
                        positions[stock] = max_shares
                        
                        trades.append({
                            'Date': date_next,
                            'Ticker': stock,
                            'Action': 'Buy',
                            'Price': price_next,
                            'Shares': max_shares,
                            'Amount': -total_cost,
                            'Reason': f'Entry Signal. RSI14 Rank: {sorted_candidates.index(stock)+1}'
                        })
            else:
                 daily_candidates_log.append({
                    'Date': date_t,
                    'Candidates': [],
                    'Count': 0
                })

            daily_record.append({
                'Date': date_t,
                'Held_Count': len(positions),
                'Cash': cash,
                'Total_Equity': current_equity
            })

        # 回測結束，清算最後資產 (Optionally)或是直接以最後一天淨值計算
        final_equity = cash
        last_date = dates[-1]
        for stock, shares in positions.items():
            price_last = self.data_p.at[last_date, stock]
            if not pd.isna(price_last):
                final_equity += shares * price_last
        
        # 補上最後一天的權益
        equity_curve.append({'Date': last_date, 'Equity': final_equity})
        
        return pd.DataFrame(trades), pd.DataFrame(equity_curve).set_index('Date'), pd.DataFrame(daily_candidates_log), pd.DataFrame(daily_record)

    def evaluate_performance(self, equity_series):
        """
        計算績效指標: CAGR, MaxDD, Calmar
        """
        if equity_series.empty:
            return -999, -999, -999
            
        initial_value = INITIAL_CAPITAL
        final_value = equity_series.iloc[-1]
        
        # 年化報酬率 (CAGR)
        days = (equity_series.index[-1] - equity_series.index[0]).days
        years = days / 365.25
        if years == 0: years = 0.001
        cagr = (final_value / initial_value) ** (1 / years) - 1
        
        # 最大回撤 (MaxDD)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_dd = drawdown.min() # 負值
        
        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return cagr, max_dd, calmar

# ==========================================
# 3. 優化演算法類別 (Ant Colony Optimization)
# ==========================================
class AntColonyOptimizer:
    def __init__(self, data_p, data_q, n_ants=10, n_iterations=3, alpha=1.0, evaporation_rate=0.1): # 預設迭代次數少一點供測試
        self.data_p = data_p
        self.data_q = data_q
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha # 費洛蒙重要性
        self.evaporation_rate = evaporation_rate
        
        # 參數範圍定義 (離散化)
        self.param_ranges = {
            'N_Check': range(3,11),    # 檢查天數
            'M_RSI': range(5, 30),      # RSI 週期
            'K': range(1,2,3),          # 次數門檻
            'S_H': range(2, 6),        # 持股檔數
            'V_Bar': [ 2, 3, 4], # 成交金額門檻 (千萬)
            'Y_Check': range(3, 7)     # 出場檢查天數
        }
        
        # 初始化費洛蒙 (均勻分佈)
        self.pheromones = {}
        for param, values in self.param_ranges.items():
            self.pheromones[param] = {v: 1.0 for v in values}
            
        self.best_result = {
            'cagr': -float('inf'),
            'params': None,
            'calmar': 0
        }

    def select_params(self):
        """
        根據費洛蒙機率選擇一組參數
        """
        selected_params = {}
        for param, values in self.param_ranges.items():
            probs = []
            for v in values:
                probs.append(self.pheromones[param][v] ** self.alpha)
            
            total = sum(probs)
            norm_probs = [p / total for p in probs]
            
            selected_value = np.random.choice(values, p=norm_probs)
            selected_params[param] = selected_value
            
        return selected_params

    def update_pheromones(self, ant_results):
        """
        更新費洛蒙
        ant_results: list of (params, score)
        """
        # 蒸發
        for param in self.pheromones:
            for v in self.pheromones[param]:
                self.pheromones[param][v] *= (1 - self.evaporation_rate)
        
        # 增強
        for params, score in ant_results:
            if score > 0: # 只對正報酬增強
                deposit = score # 可以根據分數調整增強幅度
                for key, value in params.items():
                    self.pheromones[key][value] += deposit

    def run(self):
        print(f"開始執行螞蟻演算法優化 (Generations: {self.n_iterations}, Ants: {self.n_ants})...")
        
        for generation in range(self.n_iterations):
            ant_results = []
            gen_best_cagr = -float('inf')
            gen_best_params = None
            gen_best_calmar = 0
            
            for ant in range(self.n_ants):
                params = self.select_params()
                
                # 邏輯約束檢查: N_Check 必須 >= K, 等等
                if params['N_Check'] < params['K']:
                    params['N_Check'] = params['K'] + 2 # 簡單修正
                
                strategy = Strategy(self.data_p, self.data_q, params)
                _, equity_curve, _, _ = strategy.run_backtest()
                cagr, _, calmar = strategy.evaluate_performance(equity_curve['Equity'])
                
                ant_results.append((params, cagr))
                
                if cagr > gen_best_cagr:
                    gen_best_cagr = cagr
                    gen_best_params = params
                    gen_best_calmar = calmar
            
            # 更新全局最佳
            if gen_best_cagr > self.best_result['cagr']:
                self.best_result['cagr'] = gen_best_cagr
                self.best_result['params'] = gen_best_params
                self.best_result['calmar'] = gen_best_calmar
                
            # 更新費洛蒙
            self.update_pheromones(ant_results)
            
            print(f"Gen {generation+1}/{self.n_iterations} | Best CAGR: {gen_best_cagr:.2%} | Calmar: {gen_best_calmar:.2f} | Params: {gen_best_params}")
            
        print("\n優化完成!")
        print(f"全局最佳 CAGR: {self.best_result['cagr']:.2%}")
        print(f"最佳參數: {self.best_result['params']}")
        
        return self.best_result['params']

# ==========================================
# 4. 主程式
# ==========================================
def main():
    # 1. 讀取資料
    if not os.path.exists(DATA_FILE):
        print(f"錯誤: 找不到檔案 {DATA_FILE}")
        return

    data_p, data_q = load_data(DATA_FILE)
    
    # 2. 執行優化 (或設定固定參數)
    # 若要快速測試，可減少 n_ants 和 n_iterations
    optimizer = AntColonyOptimizer(data_p, data_q, n_ants=1000, n_iterations=5, alpha=1.2)
    best_params = optimizer.run()
    
    # 3. 使用最佳參數執行最終回測
    print("\n使用最佳參數執行最終回測...")
    strategy = Strategy(data_p, data_q, best_params)
    trades_df, equity_df, candidates_df, daily_record_df = strategy.run_backtest()
    
    # 4. 產出報告
    cagr, max_dd, calmar = strategy.evaluate_performance(equity_df['Equity'])
    
    # 彙總資訊
    summary_data = {
        'Metric': ['CAGR', 'Max Drawdown', 'Calmar Ratio', 'Win Rate', 'Total Trades', 'Final Equity'],
        'Value': [
            f"{cagr:.2%}",
            f"{max_dd:.2%}",
            f"{calmar:.2f}",
            # 勝率計算: 獲利交易筆數 / 總交易筆數
            f"{len(trades_df[trades_df['Amount'] > 0]) / len(trades_df):.2%}" if len(trades_df) > 0 else "0%",
            len(trades_df),
            f"{equity_df['Equity'].iloc[-1]:,.0f}"
        ]
    }
    # 補上最佳參數
    for k, v in best_params.items():
        summary_data['Metric'].append(f"Param: {k}")
        summary_data['Value'].append(v)
        
    summary_df = pd.DataFrame(summary_data)
    
    print("\n--- 策略績效總結 ---")
    print(summary_df)
    
    # 寫入 Excel
    print(f"\n正在寫入結果至 {RESULT_FILE} ...")
    with pd.ExcelWriter(RESULT_FILE, engine='xlsxwriter') as writer:
        trades_df.to_excel(writer, sheet_name='Trades')
        equity_df.to_excel(writer, sheet_name='Equity_Curve')
        # 每日持股稍微處理一下顯示格式
        # daily_record_df.to_excel(writer, sheet_name='Daily_Record')
        daily_record_df.to_excel(writer, sheet_name='Daily_Record') 
        candidates_df.to_excel(writer, sheet_name='Daily_Candidate')
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
    # 5. 繪製權益曲線圖
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
    
    # 回撤陰影區域
    cummax = equity_df['Equity'].cummax()
    drawdown = (equity_df['Equity'] - cummax) / cummax
    
    # 建立雙軸畫回撤 (可選) 或直接畫權益
    # 這裡依照需求畫權益曲線圖 (含回撤陰影表示虧損區段)
    plt.fill_between(equity_df.index, equity_df['Equity'], cummax, color='red', alpha=0.1, label='Drawdown Area')
    
    plt.title(f'Equity Curve (CAGR: {cagr:.2%}, MaxDD: {max_dd:.2%})')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('equity_curve.png') # 存檔
    # plt.show() # 如果是互動環境可顯示
    
    print(f"執行完成! 結果已儲存至 {RESULT_FILE} 及 equity_curve.png")

if __name__ == '__main__':
    main()
