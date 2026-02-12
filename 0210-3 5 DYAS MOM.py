
import pandas as pd
import numpy as np
import random
import os
import sys

# -----------------------------------------------------------------------------
# Global Configuration (可調整)
# -----------------------------------------------------------------------------
DATA_FILENAME = 'cleaned_stock_data1.xlsx'  # 資料來源：cleaned_stock_data1.xlsx 或 cleaned_stock_data2.xlsx
INITIAL_CAPITAL = 20_000_000                # 初始資金 2,000 萬
MAX_HOLDINGS = 5                            # 最大持倉檔數
COST_TAX = 0.001                            # 交易稅 0.1% (賣出時)
COST_SLIPPAGE = 0.003                       # 滑價 0.3% (買入與賣出皆計算)
ACO_GENERATIONS = 10                        # 螞蟻演算法世代數 (可調整，建議 10-50)
ACO_ANTS = 5                               # 每世代螞蟻數 (可調整，建議 10-50)
ACO_ALPHA = 1.0                             # 費洛蒙重要性

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
def load_data(filename):
    """讀取 Excel 資料並回傳 DataFrame"""
    print(f"Loading data from {filename}...")
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        sys.exit(1)
        
    xls = pd.ExcelFile(filename)
    df_p = pd.read_excel(xls, 'P', index_col=0) # Adjusted Close
    df_q = pd.read_excel(xls, 'Q', index_col=0) # Volume (張)
    df_f = pd.read_excel(xls, 'F', index_col=0) # Foreign Investment
    
    # Ensure Datetime Index
    df_p.index = pd.to_datetime(df_p.index)
    df_q.index = pd.to_datetime(df_q.index)
    df_f.index = pd.to_datetime(df_f.index)
    
    # Fill NA (Forward Fill then Backward Fill)
    df_p = df_p.ffill().bfill()
    df_q = df_q.fillna(0)
    df_f = df_f.fillna(0)
    
    return df_p, df_q, df_f

# -----------------------------------------------------------------------------
# Strategy Class
# -----------------------------------------------------------------------------
class Strategy:
    """
    策略邏輯：
    1. 篩選流動性：成交金額 (Price * Volume * 1000) >= 40,000,000 (買進金額 400萬 <= 1/10 成交金額)
    2. 趨勢濾網：收盤價 > MA_Short & 收盤價 > MA_Long
    3. 進場排序：N日動能 (Momentum)
    4. 出場：持有 N 日後再平衡 (Rebalance)
    """
    def __init__(self, df_p, df_q, params):
        self.df_p = df_p
        self.df_q = df_q
        self.params = params
        
        # Unpack Parameters
        self.mom_days = int(params['mom_days'])
        self.ma_short = int(params['ma_short'])
        self.ma_long = int(params['ma_long'])
        self.rebalance_days = int(params['rebalance_days'])
        self.stop_loss = params.get('stop_loss', 1.0) # 1.0 = 100% loss (no stop)
        
        # Pre-calculate Indicators (Vectorized for speed)
        self.momentum = df_p.pct_change(self.mom_days)
        self.ma_s_series = df_p.rolling(self.ma_short).mean()
        self.ma_l_series = df_p.rolling(self.ma_long).mean()
        
        # Calculate Turnover (Price * Volume * 1000)
        self.turnover = df_p * df_q * 1000
        
    def run_backtest(self):
        dates = self.df_p.index
        equity = INITIAL_CAPITAL
        equity_curve = []
        trades = []
        positions = {} # {stock: {'quantity': q, 'entry_price': p, 'entry_date': d, 'days_held': n}}
        
        # Iterate through dates
        # Start after max lookback
        start_idx = max(self.ma_long, self.mom_days) + 1
        
        for i in range(start_idx, len(dates)):
            curr_date = dates[i]
            prev_date = dates[i-1] # Logic uses data available at T-1 close to trade at T open usually, or T close.
            # Stock.md logic usually implies T+1 trade based on T signal. 
            # Here let's assume we calculate signal on prev_date close, and trade on curr_date Open/Close.
            # To be precise with "Rebalance every N days", we only trade on specific days.
            
            # --- 1. Update Positions (Mark to Market & Check Stop Loss/Exit) ---
            current_prices = self.df_p.loc[curr_date]
            daily_holdings = []
            
            # Identify Exits
            stocks_to_sell = []
            
            for stock, pos in positions.items():
                if stock not in current_prices.index or pd.isna(current_prices[stock]):
                    continue
                
                curr_price = current_prices[stock]
                ret = (curr_price / pos['entry_price']) - 1
                
                # Check Stop Loss
                if ret < -self.stop_loss:
                    stocks_to_sell.append((stock, 'Stop Loss', curr_price))
                    continue
                
                # Check Rebalance Period
                pos['days_held'] += 1
                if pos['days_held'] >= self.rebalance_days:
                    stocks_to_sell.append((stock, 'Rebalance', curr_price))
                    continue
                
                # Keep Holding
                daily_holdings.append({'Stock': stock, 'Qty': pos['quantity'], 'Value': pos['quantity'] * curr_price})

            # Execute Sells
            for stock, reason, price in stocks_to_sell:
                qty = positions[stock]['quantity']
                
                # Transaction Cost: Tax + Slippage
                sell_val = qty * price
                cost = sell_val * (COST_TAX + COST_SLIPPAGE)
                net_pnl = sell_val - cost - (qty * positions[stock]['entry_price'] * (1 + COST_SLIPPAGE)) # Approx PnL
                
                equity += (sell_val - cost)
                
                trades.append({
                    'Date': curr_date,
                    'Action': 'Sell',
                    'Stock': stock,
                    'Price': price,
                    'Qty': qty,
                    'Reason': reason,
                    'PnL': net_pnl,
                    'Return': net_pnl / (qty * positions[stock]['entry_price'])
                })
                del positions[stock]

            # --- 2. Check for Entries (Only if slots available) ---
            # Buying Logic: Fill up to MAX_HOLDINGS
            # Only buy on "Rebalance Days" or any day? 
            # If "Rebalance every 5 days", usually implies entire portfolio is reviewed.
            # But here let's allow filling empty slots daily if funds allow, OR adhere strictly to rebalance schedule.
            # Given "Optimal" was strict 5-day holds, let's treat every day as a potential entry if we have cash/slots.
            
            open_slots = MAX_HOLDINGS - len(positions)
            
            if open_slots > 0:
                # Target Position Size
                # stock.md: "進場S_B：總資金 / S_H檔" -> 20M / 5 = 4M fixed? 
                # Or Dynamic Equity / 5?
                # "買進單1檔股票金額固定（400萬元）" -> Fixed 4M.
                target_pos_size = 4_000_000
                
                if equity < target_pos_size: # Not enough cash
                    pass # Or buy what we can? Let's stick to fixed 4m if possible.
                
                # --- Get Candidates ---
                # Based on PREVIOUS DAY data (Signal)
                p_prev = self.df_p.loc[prev_date]
                mom_prev = self.momentum.loc[prev_date]
                ma_s_prev = self.ma_s_series.loc[prev_date]
                ma_l_prev = self.ma_l_series.loc[prev_date]
                turnover_prev = self.turnover.loc[prev_date]
                
                # 1. Liquidity Filter
                # Buy Amount (4M) <= 1/10 Turnover => Turnover >= 40M
                candidates = turnover_prev[turnover_prev >= 40_000_000].index
                
                # 2. Trend Filter (Price > MA_S > MA_L or just Price > Both)
                # Let's use Price > MA_S and Price > MA_L
                whitelist = []
                for stock in candidates:
                    price = p_prev.get(stock, np.nan)
                    if pd.isna(price): continue
                    
                    if price > ma_s_prev.get(stock, 0) and price > ma_l_prev.get(stock, 0):
                        whitelist.append(stock)
                
                # 3. Momentum Rank
                # Get Momentum values for whitelist
                cands_mom = mom_prev.loc[whitelist].sort_values(ascending=False)
                
                # Select Top N
                buy_list = []
                for stock in cands_mom.index:
                    if len(buy_list) >= open_slots:
                        break
                    if stock not in positions:
                        buy_list.append(stock)
                
                # Execute Buys
                # Buy at *Current Day* price (we use Open or Close? Let's use Close to simple daily data, with slippage)
                curr_prices_buy = self.df_p.loc[curr_date]
                
                for stock in buy_list:
                    if stock not in curr_prices_buy.index or pd.isna(curr_prices_buy[stock]):
                        continue
                    
                    buy_price = curr_prices_buy[stock]
                    # Apply Slippage on Buy
                    cost_basis = buy_price * (1 + COST_SLIPPAGE)
                    
                    # Calculate Qty
                    qty = int(target_pos_size / cost_basis)
                    
                    if qty > 0 and equity >= target_pos_size:
                        total_cost = qty * cost_basis
                        equity -= total_cost
                        
                        positions[stock] = {
                            'quantity': qty,
                            'entry_price': cost_basis, # Includes slippage
                            'entry_date': curr_date,
                            'days_held': 0
                        }
                        
                        trades.append({
                            'Date': curr_date,
                            'Action': 'Buy',
                            'Stock': stock,
                            'Price': buy_price,
                            'Qty': qty,
                            'Reason': 'Signal',
                            'PnL': 0,
                            'Return': 0
                        })

            # Record Daily Equity
            mkt_val = sum(p['quantity'] * current_prices.get(s, 0) for s, p in positions.items())
            total_equity = equity + mkt_val
            equity_curve.append({'Date': curr_date, 'Equity': total_equity})

        return pd.DataFrame(equity_curve).set_index('Date'), pd.DataFrame(trades)

# -----------------------------------------------------------------------------
# Ant Colony Optimization (ACO) Class
# -----------------------------------------------------------------------------
class AntColonyOptimizer:
    def __init__(self, df_p, df_q):
        self.df_p = df_p
        self.df_q = df_q
        
        # Parameter Space
        self.param_grid = {
            'mom_days': [3, 5, 10, 20, 60],
            'ma_short': [5, 10, 20],
            'ma_long': [20, 40, 60],
            'rebalance_days': [3, 5, 10, 20],
            'stop_loss': [0.05, 0.10, 0.20, 1.0] # 1.0 = Off
        }
        
        # Pheromones (Initialize uniformly)
        self.pheromones = {
            k: {val: 1.0 for val in v} for k, v in self.param_grid.items()
        }
        
    def select_params(self):
        """Roulette Wheel Selection based on Pheromones"""
        params = {}
        for key, values in self.param_grid.items():
            probs = [self.pheromones[key][v] for v in values]
            total = sum(probs)
            probs = [p / total for p in probs]
            params[key] = np.random.choice(values, p=probs)
        return params
    
    def update_pheromones(self, solutions):
        """Update pheromones based on solution quality"""
        # Evaporation
        for key in self.pheromones:
            for val in self.pheromones[key]:
                self.pheromones[key][val] *= 0.9 # Evaporate 10%
                
        # Deposit
        # Sort solutions by fitness
        solutions.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Top 50% get pheromones
        top_n = len(solutions) // 2
        for i in range(top_n):
            sol = solutions[i]
            fitness = sol['fitness']
            params = sol['params']
            
            deposit = fitness * ACO_ALPHA # Strength of deposit
            for k, v in params.items():
                self.pheromones[k][v] += deposit

    def fitness_function(self, cagr, calmar):
        """Objective: Maximize weighted CAGR & Calmar"""
        if pd.isna(cagr) or pd.isna(calmar): return -999
        # Normalize roughly: CAGR 0.2 ~ 1.0, Calmar 1 ~ 5
        # Let's value Calmar highly to reduce drawdown
        return cagr * 100 + calmar * 10 

    def run(self):
        best_overall = None
        
        for gen in range(ACO_GENERATIONS):
            solutions = []
            print(f"\n--- Generation {gen+1}/{ACO_GENERATIONS} ---")
            
            for ant in range(ACO_ANTS):
                params = self.select_params()
                
                # Map float params back if needed (stop_loss is float)
                # Run Strategy
                strat = Strategy(self.df_p, self.df_q, params)
                equity, trades = strat.run_backtest()
                
                # Calc Metrics
                if len(equity) > 0:
                    start_eq = equity['Equity'].iloc[0]
                    end_eq = equity['Equity'].iloc[-1]
                    years = (equity.index[-1] - equity.index[0]).days / 365.25
                    cagr = (end_eq / start_eq) ** (1/years) - 1 if years > 0 and start_eq > 0 else 0
                    
                    peaks = equity['Equity'].cummax()
                    dd = (equity['Equity'] - peaks) / peaks
                    max_dd = dd.min()
                    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
                else:
                    cagr, calmar = 0, 0
                
                fitness = self.fitness_function(cagr, calmar)
                
                solutions.append({
                    'params': params,
                    'fitness': fitness,
                    'cagr': cagr,
                    'calmar': calmar
                })
                
                print(f"Ant {ant+1}: CAGR={cagr:.1%}, Calmar={calmar:.2f}, Params={params}")
            
            # Update Pheromones
            self.update_pheromones(solutions)
            
            # Track Best
            gen_best = max(solutions, key=lambda x: x['fitness'])
            print(f"Generation Best: CAGR={gen_best['cagr']:.1%}, Calmar={gen_best['calmar']:.2f}, Params={gen_best['params']}")
            
            if best_overall is None or gen_best['fitness'] > best_overall['fitness']:
                best_overall = gen_best
                
        print("\n=== Optimization Complete ===")
        print(f"Best Overall: CAGR={best_overall['cagr']:.1%}, Calmar={best_overall['calmar']:.2f}")
        print(f"Best Params: {best_overall['params']}")
        
        return best_overall['params']

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if manual or ACO
    #mode = 'ACO' # Default to ACO for optimization
    mode = 'manual' # Default to ACO for optimization

    df_p, df_q, df_f = load_data(DATA_FILENAME)
    
    if mode == 'ACO':
        print("Starting ACO Optimization...")
        optimizer = AntColonyOptimizer(df_p, df_q)
        best_params = optimizer.run()
        
        print("\nRunning Final Strategy with Best Params...")
        strat = Strategy(df_p, df_q, best_params)
        equity, trades = strat.run_backtest()
        
    else:
        # Manual Mode (Example)
        params = {
            'mom_days': 5,
            'ma_short': 20,
            'ma_long': 60,
            'rebalance_days': 5,
            'stop_loss': 0.08
        }
        strat = Strategy(df_p, df_q, params)
        equity, trades = strat.run_backtest()
    
    # Save Results
    with pd.ExcelWriter('strategy_results.xlsx') as writer:
        equity.to_excel(writer, sheet_name='Equity_Curve')
        trades.to_excel(writer, sheet_name='Trades')
        # Add Summary
        summary = pd.DataFrame([{
            'Final Equity': equity['Equity'].iloc[-1],
            'CAGR': (equity['Equity'].iloc[-1]/INITIAL_CAPITAL)**(1/((equity.index[-1]-equity.index[0]).days/365.25))-1,
            # Add more metrics...
        }])
        summary.to_excel(writer, sheet_name='Summary')
        
    print("Results saved to strategy_results.xlsx")
