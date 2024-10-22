import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

class CryptoTradingBot:
    def __init__(self, data_path, n_neighbors=5, initial_balance=10000, prediction_window=12, min_price_change=0.02):
        """
        Initialize the trading bot
        
        Parameters:
        data_path: Path to the CSV file containing trading data
        n_neighbors: Number of neighbors for KNN algorithm
        initial_balance: Initial balance for backtesting
        prediction_window: Number of periods to look ahead for prediction (e.g., 12 hours)
        min_price_change: Minimum price change to consider for trade signal (e.g., 0.02 for 2%)
        """
        self.data_path = Path(data_path)
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',
            metric='euclidean'
        )
        self.scaler = StandardScaler()
        self.initial_balance = initial_balance
        self.prediction_window = prediction_window
        self.min_price_change = min_price_change
        
    def load_data(self):
        """Load and preprocess data from CSV file"""
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp')
        return df.reset_index(drop=True)
    
    def add_features(self, df):
        """Add technical indicators as features"""
        df = df.copy()
        
        # Calculate VWAP
        df['vwap'] = (df['turnover'].div(df['volume'])).ffill()
        
        # Longer-term moving averages
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Price-based indicators with longer windows
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_slow'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands with longer window
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        # Trend strength indicators
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        
        # Volume-based indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Price and volume changes over multiple periods
        for period in [3, 6, 12, 24]:
            df[f'price_change_{period}'] = df['close'].pct_change(periods=period)
            df[f'volume_change_{period}'] = df['volume'].pct_change(periods=period)
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        return df.ffill().bfill()

    def prepare_data(self, df):
        """Prepare features and target for ML model"""
        # Calculate future returns for the prediction window
        future_returns = df['close'].pct_change(periods=self.prediction_window).shift(-self.prediction_window)
        
        # Create target variable (1 if price increases by min_price_change, 0 otherwise)
        df['target'] = (future_returns > self.min_price_change).astype(int)
        
        feature_columns = [
            'rsi', 'rsi_slow', 'macd', 'macd_signal', 'macd_diff',
            'bb_high', 'bb_mid', 'bb_low', 'bb_width',
            'adx', 'volume_sma', 'volume_std', 'obv',
            'price_change_3', 'price_change_6', 'price_change_12', 'price_change_24',
            'volume_change_3', 'volume_change_6', 'volume_change_12', 'volume_change_24',
            'volatility', 'atr',
            'vwap'
        ]
        
        return df[feature_columns], df['target']

    def backtest(self):
        """Perform backtesting of the trading strategy"""
        df = self.load_data()
        df = self.add_features(df)
        X, y = self.prepare_data(df)
        
        # Remove rows with NaN values
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]
        df = df[valid_indices].copy()
        
        # Initialize results
        results = []
        balance = self.initial_balance
        position = 0
        entry_price = 0
        trades = []
        
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = self.model.predict(X_test_scaled)
            probabilities = self.model.predict_proba(X_test_scaled)
            
            # Trading simulation with stricter conditions
            for i in range(len(test_index)):
                current_price = df.iloc[test_index[i]]['close']
                current_time = df.iloc[test_index[i]]['timestamp']
                
                if i < len(predictions):
                    prediction = predictions[i]
                    probability = probabilities[i][prediction]
                    
                    # Trading logic with stricter conditions
                    if position == 0:  # No position
                        if (prediction == 1 and 
                            probability > 0.75 and  # Increased confidence threshold
                            df.iloc[test_index[i]]['adx'] > 25):  # Strong trend condition
                            position = 1
                            entry_price = current_price
                            trades.append({
                                'type': 'buy',
                                'entry_time': current_time,
                                'entry_price': entry_price,
                                'confidence': probability
                            })
                    elif position == 1:  # Long position
                        # Exit conditions
                        if (prediction == 0 or 
                            (current_price - entry_price) / entry_price < -0.02 or  # Stop loss
                            (current_price - entry_price) / entry_price > 0.04):    # Take profit
                            pnl = (current_price - entry_price) / entry_price * balance
                            balance += pnl
                            position = 0
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

        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(results_df, trades_df)
        
        # Plot results
        self.plot_backtesting_results(results_df, trades_df)
        
        return results_df, trades_df, performance_metrics
    
    def calculate_performance_metrics(self, results_df, trades_df):
        """Calculate various performance metrics"""
        initial_balance = self.initial_balance
        final_balance = results_df['balance'].iloc[-1]
        
        # Basic metrics
        total_return = (final_balance - initial_balance) / initial_balance * 100
        n_trades = len(trades_df) // 2  # Divide by 2 because each complete trade has entry and exit
        
        # Calculate daily returns
        results_df['daily_returns'] = results_df['balance'].pct_change()
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(results_df['balance'])
        sharpe_ratio = self.calculate_sharpe_ratio(results_df['daily_returns'])
        
        # Win rate
        profitable_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(profitable_trades) / n_trades * 100 if n_trades > 0 else 0
        
        return {
            'Total Return (%)': total_return,
            'Number of Trades': n_trades,
            'Win Rate (%)': win_rate,
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
        excess_returns = returns - risk_free_rate/365  # Daily risk-free rate
        return np.sqrt(365) * (excess_returns.mean() / excess_returns.std())
    
    def plot_backtesting_results(self, results_df, trades_df):
        """Plot backtesting results"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Price and trades
        plt.subplot(2, 1, 1)
        plt.plot(results_df['timestamp'], results_df['price'], label='Price', alpha=0.7)
        
        # Plot buy and sell points
        buy_trades = trades_df[trades_df['type'] == 'buy']
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        plt.scatter(buy_trades['entry_time'], buy_trades['entry_price'], 
                   marker='^', color='green', label='Buy', alpha=0.7)
        plt.scatter(sell_trades['exit_time'], sell_trades['exit_price'], 
                   marker='v', color='red', label='Sell', alpha=0.7)
        
        plt.title('Backtesting Results')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        # Plot 2: Portfolio value
        plt.subplot(2, 1, 2)
        plt.plot(results_df['timestamp'], results_df['balance'], label='Portfolio Value')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize bot with path to data
    bot = CryptoTradingBot(
        data_path='data/BTCUSDT_60_data.csv',
        n_neighbors=5,
        initial_balance=10000,
        prediction_window=6,  # Predict 12 hours ahead
        min_price_change=0.01  # Look for 2% moves
    )
    
    # Run backtest
    results_df, trades_df, metrics = bot.backtest()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")