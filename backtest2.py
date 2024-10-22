import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
from datetime import datetime

class CryptoTradingBot:
    def __init__(self, data_path, model_type='rf', n_neighbors=5, initial_balance=10000, 
                 prediction_window=24, min_price_change=0.02, risk_per_trade=0.02):
        """
        Initialize the trading bot with parameters optimized for larger moves
        prediction_window increased to 24 periods to capture bigger moves
        min_price_change increased to 0.02 (2%) to focus on significant moves
        """
        self.data_path = Path(data_path)
        self.model_type = model_type
        if model_type == 'knn':
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights='distance',
                metric='euclidean'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=200,  # Increased number of trees
                max_depth=None,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            )
        self.scaler = StandardScaler()
        self.initial_balance = initial_balance
        self.prediction_window = prediction_window
        self.min_price_change = min_price_change
        self.risk_per_trade = risk_per_trade
        
    def load_data(self):
        """Load and preprocess data from CSV file"""
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp')
        return df.reset_index(drop=True)
    
    def add_features(self, df):
        """Enhanced feature set for identifying larger moves"""
        df = df.copy()
        
        # Momentum and trend features
        for window in [3, 5, 8, 13, 21, 34]:
            # ROC for multiple timeframes
            df[f'roc_{window}'] = df['close'].pct_change(periods=window)
            
            # Exponential ROC
            df[f'exp_roc_{window}'] = df['close'].ewm(span=window).mean().pct_change()
            
            # Stochastic RSI
            rsi = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
            df[f'srsi_k_{window}'] = ta.momentum.StochasticOscillator(
                rsi, rsi, rsi, window=14).stoch()
            
            # Triple EMA system
            df[f'tema_{window}'] = ta.trend.ema_indicator(
                ta.trend.ema_indicator(
                    ta.trend.ema_indicator(df['close'], window=window),
                    window=window),
                window=window)
        
        # Volatility features
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_percent'] = df['atr'] / df['close']
        
        # Historical volatility
        for window in [5, 10, 20, 30]:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(
                window=window).std() * np.sqrt(365)
        
        # Volume analysis
        df['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_z_score'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume_std']
        
        # Price patterns
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['body_range'] = abs(df['close'] - df['open']) / df['close']
        
        # Trend strength
        for window in [14, 21]:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=window)
            df[f'adx_{window}'] = adx.adx()
            df[f'di_plus_{window}'] = adx.adx_pos()
            df[f'di_minus_{window}'] = adx.adx_neg()
        
        # Custom features for large moves
        df['price_velocity'] = df['close'].diff(5) / 5
        df['price_acceleration'] = df['price_velocity'].diff(5)
        
        # Volume price trend
        df['vpt'] = (df['volume'] * 
                    ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
        
        # Custom momentum indicators
        df['momentum_strength'] = df['close'].diff(5).rolling(window=10).sum()
        df['momentum_streak'] = (df['close'].diff() > 0).astype(int).rolling(10).sum()
        
        return df.ffill().bfill()

    def prepare_data(self, df):
        """Prepare data with focus on larger price moves"""
        # Calculate future returns for multiple timeframes
        returns = {}
        for window in [12, 24, 48]:  # Multiple prediction windows
            returns[window] = df['close'].pct_change(periods=window).shift(-window)
        
        # Create target based on maximum return across timeframes
        max_return = pd.concat([returns[w] for w in returns.keys()], axis=1).max(axis=1)
        df['target'] = (max_return > self.min_price_change).astype(int)
        
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                        'turnover', 'target']]
        
        return df[feature_cols], df['target']

    def calculate_position_size(self, balance, current_price, atr):
        """Calculate position size with more aggressive sizing"""
        risk_amount = balance * self.risk_per_trade
        stop_distance = atr * 1.5  # Reduced from 2 ATR to 1.5 ATR
        position_size = risk_amount / stop_distance
        return position_size, stop_distance

    def backtest(self):
        """Backtesting with focus on larger moves"""
        df = self.load_data()
        df = self.add_features(df)
        X, y = self.prepare_data(df)
        
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]
        df = df[valid_indices].copy()
        
        results = []
        trades = []
        balance = self.initial_balance
        position = 0
        entry_price = 0
        position_size = 0
        last_trade_time = None
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            predictions = self.model.predict(X_test_scaled)
            probabilities = self.model.predict_proba(X_test_scaled)
            
            for i in range(len(test_index)):
                current_idx = test_index[i]
                current_price = df.iloc[current_idx]['close']
                current_time = df.iloc[current_idx]['timestamp']
                current_atr = df.iloc[current_idx]['atr']
                
                if i < len(predictions):
                    prediction = predictions[i]
                    probability = probabilities[i][prediction]
                    
                    # Entry conditions for larger moves
                    if position == 0:
                        min_time_passed = True if last_trade_time is None else \
                            (current_time - last_trade_time).total_seconds() / 3600 > 2
                        
                        # Trend and momentum conditions
                        trend_strength = df.iloc[current_idx]['adx_14'] > 20
                        momentum_positive = df.iloc[current_idx]['momentum_strength'] > 0
                        volume_surge = df.iloc[current_idx]['volume_z_score'] > 1.0
                        
                        # Volatility condition
                        volatility_increasing = (
                            df.iloc[current_idx]['volatility_5'] >
                            df.iloc[current_idx]['volatility_30']
                        )
                        
                        if (prediction == 1 and 
                            probability > 0.55 and  # Lower probability threshold
                            trend_strength and
                            momentum_positive and
                            (volume_surge or volatility_increasing) and
                            min_time_passed):
                            
                            position_size, stop_distance = self.calculate_position_size(
                                balance, current_price, current_atr)
                            position = 1
                            entry_price = current_price
                            stop_loss = entry_price - stop_distance
                            take_profit = entry_price + (stop_distance * 3)  # Increased reward ratio
                            
                            trades.append({
                                'type': 'buy',
                                'entry_time': current_time,
                                'entry_price': entry_price,
                                'position_size': position_size,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'confidence': probability
                            })
                    
                    elif position == 1:
                        # Exit conditions
                        momentum_reversal = df.iloc[current_idx]['momentum_strength'] < 0
                        volume_reversal = df.iloc[current_idx]['volume_z_score'] < -1.0
                        
                        exit_signal = (
                            current_price <= stop_loss or
                            current_price >= take_profit or
                            (momentum_reversal and volume_reversal) or
                            (prediction == 0 and probability > 0.7)
                        )
                        
                        if exit_signal:
                            pnl = (current_price - entry_price) / entry_price * balance
                            balance += pnl
                            position = 0
                            last_trade_time = current_time
                            trades.append({
                                'type': 'sell',
                                'exit_time': current_time,
                                'exit_price': current_price,
                                'pnl': pnl
                            })
                
                results.append({
                    'timestamp': current_time,
                    'price': current_price,
                    'position': position,
                    'balance': balance,
                    'prediction': prediction if i < len(predictions) else None,
                    'probability': probability if i < len(predictions) else None
                })
        
        results_df = pd.DataFrame(results)
        trades_df = pd.DataFrame(trades)
        
        self.plot_backtesting_results(results_df, trades_df)

        print(f"\nTotal trades taken: {len(trades_df[trades_df['type'] == 'buy'])}")
        return results_df, trades_df, self.calculate_performance_metrics(results_df, trades_df)
    
    def calculate_performance_metrics(self, results_df, trades_df):
        """Calculate comprehensive performance metrics"""
        initial_balance = self.initial_balance
        final_balance = results_df['balance'].iloc[-1]
        
        # Basic metrics
        total_return = (final_balance - initial_balance) / initial_balance * 100
        n_trades = len(trades_df[trades_df['type'] == 'buy'])
        
        # Calculate daily returns
        results_df['daily_returns'] = results_df['balance'].pct_change()
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(results_df['balance'])
        sharpe_ratio = self.calculate_sharpe_ratio(results_df['daily_returns'])
        
        # Trading metrics
        if n_trades > 0:
            profitable_trades = trades_df[
                (trades_df['type'] == 'sell') & 
                (trades_df['pnl'] > 0)
            ]
            win_rate = len(profitable_trades) / n_trades * 100
            
            # Calculate average profit/loss
            avg_profit = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
            losing_trades = trades_df[
                (trades_df['type'] == 'sell') & 
                (trades_df['pnl'] <= 0)
            ]
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            # Calculate profit factor
            total_profit = profitable_trades['pnl'].sum() if len(profitable_trades) > 0 else 0
            total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
            
        else:
            win_rate = avg_profit = avg_loss = profit_factor = 0
        
        return {
            'Total Return (%)': total_return,
            'Number of Trades': n_trades,
            'Win Rate (%)': win_rate,
            'Average Profit': avg_profit,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Final Balance': final_balance
        }
    
    def calculate_max_drawdown(self, balance_series):
        """Calculate maximum drawdown"""
        peak = balance_series.expanding(min_periods=1).max()
        drawdown = (balance_series - peak) / peak * 100
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.01):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/365
        return np.sqrt(365) * (excess_returns.mean() / excess_returns.std())
    
    def plot_backtesting_results(self, results_df, trades_df):
            """Plot comprehensive backtesting results"""
            fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            
            # Plot 1: Price and trades
            axes[0].plot(results_df['timestamp'], results_df['price'], label='Price', alpha=0.7)
            
            buy_trades = trades_df[trades_df['type'] == 'buy']
            sell_trades = trades_df[trades_df['type'] == 'sell']
            
            axes[0].scatter(buy_trades['entry_time'], buy_trades['entry_price'],
                        marker='^', color='green', label='Buy', alpha=0.7, s=100)
            axes[0].scatter(sell_trades['exit_time'], sell_trades['exit_price'],
                        marker='v', color='red', label='Sell', alpha=0.7, s=100)
            
            # Plot stop loss and take profit levels
            for _, trade in buy_trades.iterrows():
                axes[0].hlines(y=trade['stop_loss'], 
                            xmin=trade['entry_time'], 
                            xmax=trade['entry_time'] + pd.Timedelta(hours=24),
                            colors='red', linestyles='--', alpha=0.5)
                axes[0].hlines(y=trade['take_profit'],
                            xmin=trade['entry_time'],
                            xmax=trade['entry_time'] + pd.Timedelta(hours=24),
                            colors='green', linestyles='--', alpha=0.5)
            
            axes[0].set_title('Trading Activity')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            
            # Plot 2: Portfolio value and drawdown
            portfolio_value = results_df['balance']
            peak = portfolio_value.expanding(min_periods=1).max()
            drawdown = (portfolio_value - peak) / peak * 100
            
            ax2_1 = axes[1]
            ax2_2 = ax2_1.twinx()
            
            ax2_1.plot(results_df['timestamp'], portfolio_value, 
                    label='Portfolio Value', color='blue')
            ax2_2.fill_between(results_df['timestamp'], drawdown, 0,
                            color='red', alpha=0.3, label='Drawdown')
            
            ax2_1.set_xlabel('Date')
            ax2_1.set_ylabel('Portfolio Value', color='blue')
            ax2_2.set_ylabel('Drawdown %', color='red')
            
            lines1, labels1 = ax2_1.get_legend_handles_labels()
            lines2, labels2 = ax2_2.get_legend_handles_labels()
            ax2_1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Plot 3: Trade Analysis
            trade_returns = pd.Series([t['pnl'] for t in trades_df[trades_df['type'] == 'sell'].to_dict('records')])
            trade_returns.plot(kind='hist', bins=50, ax=axes[2], alpha=0.75)
            axes[2].axvline(0, color='r', linestyle='--', alpha=0.75)
            axes[2].set_title('Distribution of Trade Returns')
            axes[2].set_xlabel('Profit/Loss')
            axes[2].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.show()
            
            # Additional visualization: Plot cumulative returns over time
            plt.figure(figsize=(15, 5))
            cumulative_returns = (results_df['balance'] / self.initial_balance - 1) * 100
            plt.plot(results_df['timestamp'], cumulative_returns)
            plt.title('Cumulative Returns Over Time')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns (%)')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    bot = CryptoTradingBot(
        data_path='data/BTCUSDT_60_data.csv',
        model_type='rf',
        initial_balance=10000,
        prediction_window=6,
        min_price_change=0.01,
        risk_per_trade=0.02
    )
    
    results_df, trades_df, metrics = bot.backtest()
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    print(f"\nTotal number of trades: {len(trades_df[trades_df['type'] == 'buy'])}")