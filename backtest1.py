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
    def __init__(self, data_path, n_neighbors=5, initial_balance=10000):
        """
        Initialize the trading bot
        
        Parameters:
        data_path: Path to the CSV file containing trading data
        n_neighbors: Number of neighbors for KNN algorithm
        initial_balance: Initial balance for backtesting
        """
        self.data_path = Path(data_path)
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',
            metric='euclidean'
        )
        self.scaler = StandardScaler()
        self.initial_balance = initial_balance
        
    def load_data(self):
        """Load and preprocess data from CSV file"""
        # Read CSV file
        df = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def add_features(self, df):
        """Add technical indicators as features"""
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Calculate VWAP first - using proper ffill method
        df['vwap'] = (df['turnover'].div(df['volume'])).ffill()
        
        # Price-based indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['macd_diff'] = ta.trend.MACD(df['close']).macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        # Volume-based indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        
        # Additional momentum indicators
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['stoch_d'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
        
        # Price and volume changes
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Rolling metrics
        df['volatility'] = df['close'].rolling(window=20).std()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Handle any remaining NaN values
        df = df.ffill().bfill()
        
        return df

    def backtest(self, train_size=0.7):
        """
        Perform backtesting of the trading strategy
        
        Parameters:
        train_size: Proportion of data to use for training
        
        Returns:
        DataFrame with backtesting results and performance metrics
        """
        # Load and prepare data
        df = self.load_data()
        df = self.add_features(df)
        
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_high', 'bb_mid', 'bb_low', 'bb_width',
            'stoch_k', 'stoch_d',
            'price_change', 'volume_change',
            'volatility', 'price_range',
            'vwap', 'volume_sma', 'volume_std'
        ]
        
        # Initialize backtesting results
        results = []
        balance = self.initial_balance
        position = 0  # 0: No position, 1: Long, -1: Short
        entry_price = 0
        trades = []
        
        # Create time series split for walk-forward optimization
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Walk-forward backtesting
        for train_index, test_index in tscv.split(df):
            # Split data
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
            
            # Prepare and scale features
            X_train = train_df[feature_columns]
            y_train = (train_df['close'].shift(-1) > train_df['close']).astype(int)[:-1]
            X_train = X_train[:-1]
            
            X_test = test_df[feature_columns]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions on test set
            predictions = self.model.predict(X_test_scaled)
            probabilities = self.model.predict_proba(X_test_scaled)
            
            # Simulate trading
            for i in range(len(test_df)):
                current_price = test_df.iloc[i]['close']
                current_time = test_df.iloc[i]['timestamp']
                
                if i < len(predictions):
                    prediction = predictions[i]
                    probability = probabilities[i][prediction]
                    
                    # Trading logic
                    if position == 0:  # No position
                        if prediction == 1 and probability > 0.65:  # Buy signal
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
                        if prediction == 0 or (current_price - entry_price) / entry_price < -0.02:  # Stop loss at 2%
                            pnl = (current_price - entry_price) / entry_price * balance
                            balance += pnl
                            position = 0
                            trades.append({
                                'type': 'sell',
                                'exit_time': current_time,
                                'exit_price': current_price,
                                'pnl': pnl
                            })
                
                # Record daily results
                results.append({
                    'timestamp': current_time,
                    'price': current_price,
                    'position': position,
                    'balance': balance,
                    'prediction': prediction if i < len(predictions) else None,
                    'probability': probability if i < len(predictions) else None
                })
        
        # Convert results to DataFrame
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
        data_path='data/SOLUSDT_60_data.csv',
        n_neighbors=5,
        initial_balance=10000
    )
    
    # Run backtest
    results_df, trades_df, metrics = bot.backtest()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    # Print sample of trades
    print("\nSample of Trades:")
    print(trades_df.head())