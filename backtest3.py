import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
from datetime import datetime, timedelta
import logging
import json
import warnings
warnings.filterwarnings('ignore')

class FeatureEngine:
    """Separate class for feature engineering to improve modularity"""
    
    @staticmethod
    def add_technical_features(df):
        """Add comprehensive technical indicators"""
        df = df.copy()
        
        # Price action features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volatility indicators
        for window in [5, 10, 20, 30, 50]:
            df[f'volatility_{window}'] = df['log_return'].rolling(window=window).std() * np.sqrt(365)
            df[f'bollinger_up_{window}'] = (
                df['close'].rolling(window=window).mean() + 
                df['close'].rolling(window=window).std() * 2
            )
            df[f'bollinger_down_{window}'] = (
                df['close'].rolling(window=window).mean() - 
                df['close'].rolling(window=window).std() * 2
            )
        
        # Momentum indicators
        for window in [3, 5, 8, 13, 21, 34, 55]:
            # RSI and variations
            rsi = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
            df[f'rsi_{window}'] = rsi
            df[f'rsi_ma_{window}'] = rsi.rolling(window=window).mean()
            
            # MACD variations
            macd = ta.trend.MACD(df['close'], window_slow=window*2, window_fast=window)
            df[f'macd_{window}'] = macd.macd()
            df[f'macd_signal_{window}'] = macd.macd_signal()
            df[f'macd_diff_{window}'] = macd.macd_diff()
            
            # Momentum
            df[f'roc_{window}'] = df['close'].pct_change(periods=window)
            df[f'mom_{window}'] = df['close'].diff(window)
            
        # Volume analysis
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_z_score'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume_std']
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Advanced indicators
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        df['dpo'] = ta.trend.DPOIndicator(df['close']).dpo()
        df['kst'] = ta.trend.KSTIndicator(df['close']).kst()
        df['ichimoku_a'] = ta.trend.IchimokuIndicator(df['high'], df['low']).ichimoku_a()
        df['ichimoku_b'] = ta.trend.IchimokuIndicator(df['high'], df['low']).ichimoku_b()
        
        return df.ffill().bfill()

class RiskManager:
    """Separate class for risk management"""
    
    def __init__(self, initial_balance, max_risk_per_trade=0.02, max_trades=5):
        self.initial_balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_trades = max_trades
        self.open_trades = []
        
    def calculate_position_size(self, balance, current_price, volatility, confidence):
        """Dynamic position sizing based on volatility and model confidence"""
        risk_amount = balance * self.max_risk_per_trade
        
        # Adjust position size based on volatility
        volatility_factor = 1 - (volatility / 100)  # Reduce size for high volatility
        
        # Adjust position size based on model confidence
        confidence_factor = confidence / 100
        
        # Calculate final position size
        position_size = (risk_amount * volatility_factor * confidence_factor) / current_price
        
        return position_size
    
    def can_open_trade(self):
        """Check if new trade can be opened based on risk limits"""
        return len(self.open_trades) < self.max_trades
    
    def add_trade(self, trade):
        self.open_trades.append(trade)
    
    def remove_trade(self, trade_id):
        """Remove a trade from open trades list by trade ID"""
        self.open_trades = [t for t in self.open_trades if t['id'] != trade_id]

class CryptoTradingBot:
    def __init__(self, data_path, model_type='ensemble', initial_balance=10000,
                 prediction_window=24, min_price_change=0.02, risk_per_trade=0.02,
                 max_trades=5):
        """
        Enhanced trading bot with ensemble modeling and advanced risk management
        
        Parameters:
        -----------
        data_path : str or Path
            Path to the data file
        model_type : str
            Type of model to use ('rf', 'gbm', 'ensemble')
        initial_balance : float
            Initial trading balance
        prediction_window : int
            Number of periods to look ahead for predictions
        min_price_change : float
            Minimum price change to consider for trading
        risk_per_trade : float
            Maximum risk per trade as percentage of balance
        max_trades : int
            Maximum number of concurrent trades
        """
        self.data_path = Path(data_path)
        self.model_type = model_type
        self.models = self._initialize_models()
        self.feature_engine = FeatureEngine()
        self.risk_manager = RiskManager(initial_balance, risk_per_trade, max_trades)
        
        self.scaler = RobustScaler()  # More robust to outliers
        self.initial_balance = initial_balance
        self.prediction_window = prediction_window
        self.min_price_change = min_price_change
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the trading bot"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_models(self):
        """Initialize machine learning models"""
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        }
        
        if self.model_type == 'ensemble':
            return models
        return {self.model_type: models[self.model_type]}
    
    def load_data(self):
        """Load and validate data from CSV file"""
        try:
            df = pd.read_csv(self.data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp')
            
            # Validate data
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df.reset_index(drop=True)
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_data(self, df):
        """Prepare data for model training"""
        # Calculate future returns for multiple timeframes
        returns = {}
        for window in [12, 24, 48]:
            returns[window] = df['close'].pct_change(periods=window).shift(-window)
        
        # Create target based on maximum return across timeframes
        max_return = pd.concat([returns[w] for w in returns.keys()], axis=1).max(axis=1)
        df['target'] = (max_return > self.min_price_change).astype(int)
        
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                        'turnover', 'target']]
        
        return df[feature_cols], df['target']
    
    def get_model_predictions(self, X_scaled):
        """Get ensemble predictions from all models"""
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
            probabilities[name] = model.predict_proba(X_scaled)
        
        # Ensemble decision
        if len(self.models) > 1:
            final_pred = np.mean([pred for pred in predictions.values()], axis=0) > 0.5
            final_prob = np.mean([prob[:, 1] for prob in probabilities.values()], axis=0)
            return final_pred, final_prob
        
        return list(predictions.values())[0], list(probabilities.values())[0][:, 1]
    
    def backtest(self):
        """Enhanced backtesting with comprehensive analysis"""
        try:
            df = self.load_data()
            df = self.feature_engine.add_technical_features(df)
            X, y = self.prepare_data(df)
            
            valid_indices = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_indices]
            y = y[valid_indices]
            df = df[valid_indices].copy()
            
            results = []
            trades = []
            balance = self.initial_balance
            
            tscv = TimeSeriesSplit(n_splits=5)
            
            for train_index, test_index in tscv.split(X):
                # Training
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train all models
                for name, model in self.models.items():
                    model.fit(X_train_scaled, y_train)
                    self.logger.info(f"Model {name} training completed")
                    
                    # Log model performance
                    train_pred = model.predict(X_train_scaled)
                    self.logger.info(f"\nModel {name} Training Report:")
                    self.logger.info(classification_report(y_train, train_pred))
                
                # Get predictions
                predictions, probabilities = self.get_model_predictions(X_test_scaled)
                
                # Trading simulation
                position = 0
                entry_price = 0
                position_size = 0
                last_trade_time = None
                
                for i in range(len(test_index)):
                    current_idx = test_index[i]
                    current_price = df.iloc[current_idx]['close']
                    current_time = df.iloc[current_idx]['timestamp']
                    current_volatility = df.iloc[current_idx]['volatility_20']
                    
                    # Entry logic
                    if position == 0 and i < len(predictions):
                        prediction = predictions[i]
                        probability = probabilities[i]
                        
                        entry_conditions = (
                            prediction == 1 and
                            probability > 0.65 and
                            self.risk_manager.can_open_trade() and
                            (last_trade_time is None or 
                             (current_time - last_trade_time).total_seconds() / 3600 > 2)
                        )
                        
                        if entry_conditions:
                            position_size = self.risk_manager.calculate_position_size(
                                balance, current_price, current_volatility, probability * 100)
                            position = 1
                            entry_price = current_price
                            stop_loss = entry_price * 0.98  # 2% stop loss
                            take_profit = entry_price * 1.04  # 4% take profit
                            
                            trade = {
                                'id': len(trades),
                                'type': 'buy',
                                'entry_time': current_time,
                                'entry_price': entry_price,
                                'position_size': position_size,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'confidence': probability
                            }
                            trades.append(trade)
                            self.risk_manager.add_trade(trade)
                    
                    # Exit logic
                    elif position == 1:
                        exit_conditions = (
                            current_price <= stop_loss or
                            current_price >= take_profit or
                            (i < len(predictions) and predictions[i] == 0 and 
                             probabilities[i] > 0.7)
                        )
                        
                        if exit_conditions:
                            pnl = (current_price - entry_price) / entry_price * balance
                            balance += pnl
                            position = 0
                            last_trade_time = current_time
                            
                            trade = {
                                'type': 'sell',
                                'exit_time': current_time,
                                'exit_price': current_price,
                                'pnl': pnl
                            }
                            trades.append(trade)
                            self.risk_manager.remove_trade(trades[-2]['id'])
                    
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
            
            # Save results
            self.save_results(results_df, trades_df)

            # Calculate and return performance metrics
            metrics = self.calculate_performance_metrics(results_df, trades_df)
            return results_df, trades_df, metrics
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}")
            raise
    
    def calculate_performance_metrics(self, results_df, trades_df):
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        try:
            # Basic metrics
            initial_balance = self.initial_balance
            final_balance = results_df['balance'].iloc[-1]
            total_return = (final_balance - initial_balance) / initial_balance * 100
            
            # Trading metrics
            buy_trades = trades_df[trades_df['type'] == 'buy']
            sell_trades = trades_df[trades_df['type'] == 'sell']
            n_trades = len(buy_trades)
            
            if n_trades > 0:
                # Profitability metrics
                profitable_trades = sell_trades[sell_trades['pnl'] > 0]
                win_rate = len(profitable_trades) / n_trades * 100
                
                avg_profit = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
                losing_trades = sell_trades[sell_trades['pnl'] <= 0]
                avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
                
                # Risk metrics
                total_profit = profitable_trades['pnl'].sum() if len(profitable_trades) > 0 else 0
                total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
                profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
                
                # Calculate trade duration
                if 'entry_time' in buy_trades.columns and 'exit_time' in sell_trades.columns:
                    # Make sure we have matching pairs of trades
                    min_trades = min(len(buy_trades), len(sell_trades))
                    duration = pd.to_datetime(sell_trades['exit_time'].iloc[:min_trades].values) - \
                              pd.to_datetime(buy_trades['entry_time'].iloc[:min_trades].values)
                    avg_trade_duration = duration.mean()
                else:
                    avg_trade_duration = pd.Timedelta(0)
                
                # Advanced metrics
                results_df['daily_returns'] = results_df['balance'].pct_change()
                sharpe_ratio = self.calculate_sharpe_ratio(results_df['daily_returns'])
                sortino_ratio = self.calculate_sortino_ratio(results_df['daily_returns'])
                max_drawdown = self.calculate_max_drawdown(results_df['balance'])
                
                # Kelly criterion
                win_probability = win_rate / 100
                avg_win_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
                kelly_percentage = (win_probability - ((1 - win_probability) / avg_win_loss_ratio)) * 100
                
                metrics.update({
                    'Total Return (%)': total_return,
                    'Number of Trades': n_trades,
                    'Win Rate (%)': win_rate,
                    'Average Profit': avg_profit,
                    'Average Loss': avg_loss,
                    'Profit Factor': profit_factor,
                    'Max Drawdown (%)': max_drawdown,
                    'Sharpe Ratio': sharpe_ratio,
                    'Sortino Ratio': sortino_ratio,
                    'Kelly Percentage': kelly_percentage,
                    'Average Trade Duration': avg_trade_duration,
                    'Final Balance': final_balance
                })
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.01):
        """Calculate annualized Sharpe ratio"""
        excess_returns = returns - risk_free_rate/365
        return np.sqrt(365) * (excess_returns.mean() / excess_returns.std())
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.01):
        """Calculate Sortino ratio using only downside deviation"""
        excess_returns = returns - risk_free_rate/365
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(365) * (excess_returns.mean() / downside_std) if downside_std != 0 else np.inf
    
    def calculate_max_drawdown(self, balance_series):
        """Calculate maximum drawdown percentage"""
        peak = balance_series.expanding(min_periods=1).max()
        drawdown = (balance_series - peak) / peak * 100
        return drawdown.min()
    
    def plot_backtesting_results(self, results_df, trades_df, save_dir=None):
        """Generate comprehensive visualization of trading results"""
        # Use a valid matplotlib style
        plt.style.use('default')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 2)
        
        # 1. Price and Trading Activity
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_and_trades(ax1, results_df, trades_df)
        
        # 2. Portfolio Value and Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_portfolio_and_drawdown(ax2, results_df)
        
        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax3, trades_df)
        
        # 4. Monthly Returns Heatmap
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_monthly_returns_heatmap(ax4, results_df)
        
        # 5. Trade Duration Analysis
        ax5 = fig.add_subplot(gs[3, 0])
        self._plot_trade_duration_analysis(ax5, trades_df)
        
        # 6. Win Rate by Hour
        ax6 = fig.add_subplot(gs[3, 1])
        self._plot_win_rate_by_hour(ax6, trades_df)
        
        # 7. Cumulative Returns
        ax7 = fig.add_subplot(gs[4, :])
        self._plot_cumulative_returns(ax7, results_df)
        
        # 8. Performance Metrics Table
        ax8 = fig.add_subplot(gs[5, :])
        self._plot_performance_metrics(ax8, results_df, trades_df)
        
        plt.tight_layout()
        # Save to the specified directory if provided
        if save_dir:
            save_path = Path(save_dir) / 'trading_results.png'
        else:
            save_path = 'trading_results.png'
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_price_and_trades(self, ax, results_df, trades_df):
        """Plot price action with entry/exit points"""
        ax.plot(results_df['timestamp'], results_df['price'], label='Price', alpha=0.7)
        
        buy_trades = trades_df[trades_df['type'] == 'buy']
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        ax.scatter(buy_trades['entry_time'], buy_trades['entry_price'],
                  marker='^', color='green', label='Buy', alpha=0.7, s=100)
        ax.scatter(sell_trades['exit_time'], sell_trades['exit_price'],
                  marker='v', color='red', label='Sell', alpha=0.7, s=100)
        
        ax.set_title('Price Action and Trading Activity')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        
    def _plot_portfolio_and_drawdown(self, ax, results_df):
        """Plot portfolio value and drawdown"""
        portfolio_value = results_df['balance']
        peak = portfolio_value.expanding(min_periods=1).max()
        drawdown = (portfolio_value - peak) / peak * 100
        
        ax2 = ax.twinx()
        
        ax.plot(results_df['timestamp'], portfolio_value, 
                label='Portfolio Value', color='blue')
        ax2.fill_between(results_df['timestamp'], drawdown, 0,
                        color='red', alpha=0.3, label='Drawdown')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value', color='blue')
        ax2.set_ylabel('Drawdown %', color='red')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_returns_distribution(self, ax, trades_df):
        """Plot distribution of trade returns"""
        if 'pnl' in trades_df.columns:
            sell_trades = trades_df[trades_df['type'] == 'sell']
            if len(sell_trades) > 0:
                ax.hist(sell_trades['pnl'], bins=50, alpha=0.75)
                ax.axvline(0, color='r', linestyle='--', alpha=0.75)
                ax.set_title('Distribution of Trade Returns')
                ax.set_xlabel('Profit/Loss')
                ax.set_ylabel('Frequency')
    
    def _plot_monthly_returns_heatmap(self, ax, results_df):
        """Plot monthly returns heatmap"""
        try:
            results_df['monthly_returns'] = results_df['balance'].pct_change()
            monthly_returns = results_df.set_index('timestamp')['monthly_returns'].resample('M').sum()
            
            # Create month/year matrix
            years = monthly_returns.index.year.unique()
            months = range(1, 13)
            data = np.zeros((12, len(years)))
            
            for i, year in enumerate(years):
                for j, month in enumerate(months):
                    mask = (monthly_returns.index.year == year) & (monthly_returns.index.month == month)
                    if mask.any():
                        data[j-1, i] = monthly_returns[mask].iloc[0]
            
            im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
            plt.colorbar(im, ax=ax)
            
            # Set labels
            ax.set_yticks(range(12))
            ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels(years)
            
            ax.set_title('Monthly Returns Heatmap')
        except Exception as e:
            self.logger.error(f"Error plotting monthly returns heatmap: {str(e)}")
            ax.text(0.5, 0.5, 'Unable to plot monthly returns', 
                   ha='center', va='center')
    
    def _plot_trade_duration_analysis(self, ax, trades_df):
        """Plot trade duration analysis"""
        if 'duration' in trades_df.columns:
            trades_df['duration_hours'] = trades_df['duration'].dt.total_seconds() / 3600
            sns.boxplot(y=trades_df['duration_hours'], ax=ax)
            ax.set_title('Trade Duration Distribution')
            ax.set_ylabel('Duration (hours)')
    
    def _plot_win_rate_by_hour(self, ax, trades_df):
        """Plot win rate by hour of day"""
        if 'exit_time' in trades_df.columns:
            trades_df['hour'] = pd.to_datetime(trades_df['exit_time']).dt.hour
            trades_df['profitable'] = trades_df['pnl'] > 0
            
            hourly_stats = trades_df.groupby('hour')['profitable'].agg(['mean', 'count'])
            hourly_stats['mean'].plot(kind='bar', ax=ax)
            ax.set_title('Win Rate by Hour of Day')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Win Rate')
    
    def _plot_cumulative_returns(self, ax, results_df):
        """Plot cumulative returns"""
        cumulative_returns = (results_df['balance'] / self.initial_balance - 1) * 100
        ax.plot(results_df['timestamp'], cumulative_returns)
        ax.set_title('Cumulative Returns Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns (%)')
        ax.grid(True)
    
    def _plot_performance_metrics(self, ax, results_df, trades_df):
        """Plot performance metrics as a table"""
        metrics = self.calculate_performance_metrics(results_df, trades_df)
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [[k, f"{v:.2f}" if isinstance(v, (int, float)) else str(v)] 
                     for k, v in metrics.items()]
        ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                cellLoc='center', loc='center')
        ax.set_title('Performance Metrics Summary')
    
    def save_results(self, results_df, trades_df):
        """Save trading results and analysis to files"""
        try:
            # Create timestamp for the current run
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create main results directory if it doesn't exist
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # Create subdirectory for this run
            run_dir = results_dir / timestamp
            run_dir.mkdir(exist_ok=True)
            
            # Save data files
            results_df.to_csv(run_dir / 'results.csv', index=False)
            trades_df.to_csv(run_dir / 'trades.csv', index=False)
            
            # Calculate and save performance metrics
            metrics = self.calculate_performance_metrics(results_df, trades_df)
            with open(run_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4, default=str)
                
            # Save configuration
            config = {
                'model_type': self.model_type,
                'initial_balance': self.initial_balance,
                'prediction_window': self.prediction_window,
                'min_price_change': self.min_price_change,
                'risk_per_trade': self.risk_manager.max_risk_per_trade,
                'max_trades': self.risk_manager.max_trades,
                'data_path': str(self.data_path)
            }
            with open(run_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=4)
            
            # Generate and save the plots directly to the run directory
            self.plot_backtesting_results(results_df, trades_df, save_dir=run_dir)
            
            # Create a summary file
            summary = {
                'timestamp': timestamp,
                'total_trades': len(trades_df[trades_df['type'] == 'buy']),
                'final_balance': metrics.get('Final Balance', 0),
                'total_return': metrics.get('Total Return (%)', 0),
                'win_rate': metrics.get('Win Rate (%)', 0),
                'sharpe_ratio': metrics.get('Sharpe Ratio', 0)
            }
            
            # Append to master summary file
            summary_file = results_dir / 'master_summary.csv'
            if summary_file.exists():
                summary_df = pd.read_csv(summary_file)
                summary_df = pd.concat([summary_df, pd.DataFrame([summary])], ignore_index=True)
            else:
                summary_df = pd.DataFrame([summary])
            summary_df.to_csv(summary_file, index=False)
            
            self.logger.info(f"Results saved in: {run_dir}")
            self.logger.info("Files saved:")
            self.logger.info(f"- {run_dir / 'results.csv'}")
            self.logger.info(f"- {run_dir / 'trades.csv'}")
            self.logger.info(f"- {run_dir / 'metrics.json'}")
            self.logger.info(f"- {run_dir / 'config.json'}")
            self.logger.info(f"- {run_dir / 'trading_results.png'}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

if __name__ == "__main__":
    # Configuration
    config = {
        'data_path': 'data/SOLUSDT_60_data.csv',
        'model_type': 'ensemble',
        'initial_balance': 10000,
        'prediction_window': 24,
        'min_price_change': 0.02,
        'risk_per_trade': 0.02,
        'max_trades': 5
    }
    
    # Initialize and run bot
    bot = CryptoTradingBot(**config)
    
    try:
        results_df, trades_df, metrics = bot.backtest()
        
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: {value}")
        
        print(f"\nResults saved to files with timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    except Exception as e:
        print(f"Error running trading bot: {str(e)}")