import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
warnings.filterwarnings('ignore')

class TradingSignalNet(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.2):
        super(TradingSignalNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads=8, 
            batch_first=True,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 3)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        final_hidden = attended[:, -1, :]
        
        signal = self.classifier(final_hidden)
        confidence = self.confidence(final_hidden)
        
        return signal, confidence

class VolatilityAdaptiveSystem:
    def __init__(self):
        self.volatility_lookback = 20
        self.optimization_window = 100
        self.min_pivot_period = 5
        self.max_pivot_period = 25
        self.min_channel_width = 1
        self.max_channel_width = 8
        
    def calculate_market_volatility(self, df):
        returns = df['Close'].pct_change().dropna()
        
        hist_vol = returns.rolling(self.volatility_lookback).std() * np.sqrt(252)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(self.volatility_lookback).mean()
        atr_vol = atr / df['Close']
        
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        bb_width = (2 * std) / sma
        
        volatility_metrics = pd.DataFrame({
            'hist_vol': hist_vol,
            'atr_vol': atr_vol,
            'bb_width': bb_width
        }).ffill()
        
        normalized_metrics = volatility_metrics.apply(lambda x: (x - x.rolling(50).mean()) / x.rolling(50).std())
        composite_vol = normalized_metrics.mean(axis=1)
        
        return composite_vol.fillna(0)
    
    def adaptive_pivot_period(self, volatility_score):
        if volatility_score > 1.5:
            return self.min_pivot_period
        elif volatility_score > 0.5:
            return int(self.min_pivot_period + (self.max_pivot_period - self.min_pivot_period) * 0.3)
        elif volatility_score < -0.5:
            return self.max_pivot_period
        else:
            return int((self.min_pivot_period + self.max_pivot_period) / 2)
    
    def adaptive_channel_width(self, volatility_score, atr_ratio):
        base_width = 3
        
        vol_adjustment = volatility_score * 1
        
        atr_adjustment = (atr_ratio - 1) * 2
        
        adjusted_width = base_width + vol_adjustment + atr_adjustment
        
        return np.clip(adjusted_width, self.min_channel_width, self.max_channel_width)

class TradingDataset:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def prepare_training_data(self, df_list, tickers):
        all_sequences = []
        all_labels = []
        
        for df, ticker in zip(df_list, tickers):
            print(f"Processing {ticker} for training data...")
            sequences, labels = self.create_sequences_with_labels(df)
            if len(sequences) > 0:
                all_sequences.extend(sequences)
                all_labels.extend(labels)
        
        if len(all_sequences) == 0:
            return None, None
        
        X = torch.FloatTensor(np.array(all_sequences))
        y = torch.LongTensor(np.array(all_labels))
        
        print(f"Training data shape: {X.shape}, Labels shape: {y.shape}")
        return X, y
    
    def create_sequences_with_labels(self, df):
        df = self.add_technical_indicators(df)

        feature_cols = [
            'Close', 'Volume', 'macd', 'signal', 'hist', 
            'rsi', 'bb_upper', 'bb_lower', 'atr', 'volatility',
            'volume_ratio', 'price_change', 'high_low_ratio', 
            'close_position', 'pivot_signal'
        ]
        
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0)
        
        df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
        df['label'] = 1
        df.loc[df['future_return'] > 0.02, 'label'] = 0
        df.loc[df['future_return'] < -0.02, 'label'] = 2
        
        feature_data = df[feature_cols].values
        if len(feature_data) > 50:
            scaled_features = self.scaler.fit_transform(feature_data)
        else:
            return [], []
        
        sequences = []
        labels = []
        
        for i in range(self.sequence_length, len(scaled_features) - 5):
            sequence = scaled_features[i-self.sequence_length:i]
            label = df['label'].iloc[i]
            
            if not np.isnan(label):
                sequences.append(sequence)
                labels.append(int(label))
        
        return sequences, labels
    
    def add_technical_indicators(self, df):
        df = df.copy()

        df['rsi'] = self.calculate_rsi(df['Close'])

        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['Close'])

        df['atr'] = self.calculate_atr(df)

        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']

        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

        df['volatility'] = df['price_change'].rolling(20).std()

        df['pivot_signal'] = df['pivot_value'].fillna(0)
        df['pivot_signal'] = (df['pivot_signal'] != 0).astype(int)
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()

class TradingModelTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   epochs=100, batch_size=32, learning_rate=0.001):

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                signal_output, confidence_output = self.model(data)
                
                loss = criterion(signal_output, target)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(signal_output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        signal_output, confidence_output = self.model(data)
                        
                        loss = criterion(signal_output, target)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(signal_output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * val_correct / val_total
                
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), 'best_trading_model.pth')
                
                self.training_history['train_loss'].append(avg_train_loss)
                self.training_history['train_acc'].append(train_accuracy)
                self.training_history['val_loss'].append(avg_val_loss)
                self.training_history['val_acc'].append(val_accuracy)
                
                if epoch % 50 == 0 or epoch == epochs - 1:
                    print(f'Epoch [{epoch+1}/{epochs}]')
                    print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
                    print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
                    print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                    print('-' * 50)
            else:
                self.training_history['train_loss'].append(avg_train_loss)
                self.training_history['train_acc'].append(train_accuracy)
                
                if epoch % 50 == 0 or epoch == epochs - 1:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        
        if val_loader is not None and os.path.exists('best_trading_model.pth'):
            self.model.load_state_dict(torch.load('best_trading_model.pth'))
            print("Loaded best model from training")
        
        print(f"Training completed! Ran for full {epochs} epochs.")
        return self.training_history
    
    def plot_training_history(self):
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.training_history['train_acc'], label='Train Accuracy')
        if self.training_history['val_acc']:
            ax2.plot(self.training_history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def get_stock_data(ticker, period="1y", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def calculate_pivot_points(df, pivot_period=10, pivot_source='Wick'):
    df = df.copy()
    
    if pivot_source == 'Wick':
        high_source = df['High']
        low_source = df['Low']
    else:
        df['max_candle'] = df[['Open', 'Close']].max(axis=1)
        df['min_candle'] = df[['Open', 'Close']].min(axis=1)
        high_source = df['max_candle']
        low_source = df['min_candle']
    
    df['pivot_high'] = np.nan
    for i in range(pivot_period, len(df) - pivot_period):
        is_pivot = True
        for j in range(1, pivot_period+1):
            if high_source.iloc[i] <= high_source.iloc[i-j] or high_source.iloc[i] <= high_source.iloc[i+j]:
                is_pivot = False
                break
        if is_pivot:
            df.iloc[i, df.columns.get_loc('pivot_high')] = high_source.iloc[i]
    
    df['pivot_low'] = np.nan
    for i in range(pivot_period, len(df) - pivot_period):
        is_pivot = True
        for j in range(1, pivot_period+1):
            if low_source.iloc[i] >= low_source.iloc[i-j] or low_source.iloc[i] >= low_source.iloc[i+j]:
                is_pivot = False
                break
        if is_pivot:
            df.iloc[i, df.columns.get_loc('pivot_low')] = low_source.iloc[i]
    
    df['pivot_value'] = np.nan
    mask_high = ~df['pivot_high'].isna()
    mask_low = ~df['pivot_low'].isna()
    df.loc[mask_high, 'pivot_value'] = df.loc[mask_high, 'pivot_high']
    df.loc[mask_low, 'pivot_value'] = df.loc[mask_low, 'pivot_low']
    
    return df

def calculate_sequence(df):
    df = df.copy()
    
    df['buy_sequence'] = 0
    df['sell_sequence'] = 0
    df['buy_count'] = 0
    df['sell_count'] = 0
    
    last_buy_sequence_down = 0
    last_sell_sequence_down = 0
    
    for i in range(4, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-4]:
            df.loc[df.index[i], 'buy_sequence'] = df.loc[df.index[i-1], 'buy_sequence'] + 1
        else:
            df.loc[df.index[i], 'buy_sequence'] = 0
        
        if df['Close'].iloc[i] < df['Close'].iloc[i-4]:
            df.loc[df.index[i], 'sell_sequence'] = df.loc[df.index[i-1], 'sell_sequence'] + 1
        else:
            df.loc[df.index[i], 'sell_sequence'] = 0
        
        if i > 0:
            if df['buy_sequence'].iloc[i] < df['buy_sequence'].iloc[i-1]:
                last_buy_sequence_down = df['buy_sequence'].iloc[i]
            
            if df['sell_sequence'].iloc[i] < df['sell_sequence'].iloc[i-1]:
                last_sell_sequence_down = df['sell_sequence'].iloc[i]
        
        df.loc[df.index[i], 'buy_count'] = df['buy_sequence'].iloc[i] - last_buy_sequence_down
        df.loc[df.index[i], 'sell_count'] = df['sell_sequence'].iloc[i] - last_sell_sequence_down
    
    return df

def calculate_wma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_macd(df, fast_length=12, slow_length=26, signal_length=9):
    df = df.copy()
    df['fast_wma'] = calculate_wma(df['Close'], fast_length)
    df['slow_wma'] = calculate_wma(df['Close'], slow_length)
    df['macd'] = df['fast_wma'] - df['slow_wma']
    df['signal'] = calculate_wma(df['macd'], signal_length)
    df['hist'] = df['macd'] - df['signal']
    return df

def calculate_support_resistance(pivot_values, channel_width_pct, period_highest, period_lowest, max_sr=5, min_strength=1):
    if len(pivot_values) == 0:
        return []
    
    pivot_values = [x for x in pivot_values if not np.isnan(x)]
    
    if len(pivot_values) == 0:
        return []
    
    channel_range = (period_highest - period_lowest) * channel_width_pct / 100
    
    sr_levels = []
    
    for idx in range(len(pivot_values)):
        lower = pivot_values[idx]
        upper = lower
        point_count = 0
        
        for i in range(len(pivot_values)):
            current_point = pivot_values[i]
            price_range = upper - current_point if current_point <= lower else current_point - lower
            
            if price_range <= channel_range:
                if current_point <= upper:
                    lower = min(lower, current_point)
                else:
                    upper = max(upper, current_point)
                point_count += 1
        
        if point_count >= min_strength:
            mid_level = round((upper + lower) / 2, 2)
            if not any(abs(level[0] - mid_level) < channel_range/10 for level in sr_levels):
                sr_levels.append((mid_level, point_count))

    sr_levels.sort(key=lambda x: x[1], reverse=True)
    return sr_levels[:max_sr]

class EnhancedAdaptiveTradingSystem:
    def __init__(self, model_path=None):
        self.volatility_system = VolatilityAdaptiveSystem()
        self.model = TradingSignalNet(input_size=15, hidden_size=64)
        self.dataset = TradingDataset()
        self.trainer = TradingModelTrainer(self.model)
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.is_trained = True
            print(f"Loaded pre-trained model from {model_path}")
    
    def calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def train_on_multiple_stocks(self, tickers, epochs=100, train_split=0.8):
        print(f"Collecting data for training from {len(tickers)} stocks...")
        
        df_list = []
        valid_tickers = []
        
        for ticker in tickers:
            try:
                df = get_stock_data(ticker, period="2y", interval="1d")
                if len(df) > 100:
                    df = calculate_pivot_points(df, 10)
                    df = calculate_sequence(df)
                    df = calculate_macd(df)
                    
                    df_list.append(df)
                    valid_tickers.append(ticker)
                    print(f"✓ {ticker}: {len(df)} data points")
                else:
                    print(f"✗ {ticker}: Insufficient data ({len(df)} points)")
            except Exception as e:
                print(f"✗ {ticker}: Error - {str(e)}")
        
        if len(df_list) == 0:
            print("No valid data found for training!")
            return
        
        # Prepare training data
        X, y = self.dataset.prepare_training_data(df_list, valid_tickers)
        
        if X is None or len(X) < 100:
            print("Insufficient training data!")
            return
        
        # Split data
        split_idx = int(len(X) * train_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        history = self.trainer.train_model(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=32, learning_rate=0.001
        )
        
        self.is_trained = True
        
        self.trainer.plot_training_history()
        
        torch.save(self.model.state_dict(), 'trained_trading_model.pth')
        print("Model saved as 'trained_trading_model.pth'")
        
        return history
    
    def adaptive_dca_indicator(self, ticker, period="3y", interval="1wk"):
        if not self.is_trained:
            print("Warning: Model is not trained! Training on sample data first...")
            sample_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            self.train_on_multiple_stocks(sample_tickers, epochs=100)
        
        df = get_stock_data(ticker, period, interval)
        
        if df.empty:
            return None, None, None, None, None, None, None
        
        volatility_score = self.volatility_system.calculate_market_volatility(df)
        current_vol = volatility_score.iloc[-1] if len(volatility_score) > 0 else 0
        
        atr = self.calculate_atr(df)
        atr_ratio = atr.iloc[-1] / atr.rolling(50).mean().iloc[-1] if len(atr) > 50 else 1
        
        pivot_period = self.volatility_system.adaptive_pivot_period(current_vol)
        channel_width = self.volatility_system.adaptive_channel_width(current_vol, atr_ratio)
        
        print(f"Adaptive Parameters - Volatility Score: {current_vol:.2f}")
        print(f"Pivot Period: {pivot_period}, Channel Width: {channel_width:.1f}%")
        
        df = calculate_pivot_points(df, pivot_period)
        df = calculate_sequence(df)
        df = calculate_macd(df)
        
        period_highest = df['High'].rolling(window=300, min_periods=1).max().iloc[-1]
        period_lowest = df['Low'].rolling(window=300, min_periods=1).min().iloc[-1]
        
        pivot_values = []
        for idx in df.index:
            if not pd.isna(df.loc[idx, 'pivot_value']):
                pivot_values.insert(0, df.loc[idx, 'pivot_value'])
                if len(pivot_values) > 14:
                    pivot_values.pop()
        
        sr_levels = calculate_support_resistance(
            pivot_values, channel_width, period_highest, period_lowest, 5, 1)
        
        signal, confidence, analysis = self.generate_ml_enhanced_signal(df, sr_levels, channel_width)
        
        fig, ax = self.create_adaptive_visualization(df, sr_levels, ticker, pivot_period, channel_width)
        
        return df, sr_levels, fig, ax, signal, confidence, analysis
    
    def generate_ml_enhanced_signal(self, df, sr_levels, channel_width):

        df_with_indicators = self.dataset.add_technical_indicators(df)
        
        feature_cols = [
            'Close', 'Volume', 'macd', 'signal', 'hist', 
            'rsi', 'bb_upper', 'bb_lower', 'atr', 'volatility',
            'volume_ratio', 'price_change', 'high_low_ratio', 
            'close_position', 'pivot_signal'
        ]
        
        for col in feature_cols:
            if col in df_with_indicators.columns:
                df_with_indicators[col] = df_with_indicators[col].ffill().fillna(0)
        
        if len(df_with_indicators) >= self.dataset.sequence_length:
            feature_data = df_with_indicators[feature_cols].values
            scaled_features = self.dataset.scaler.transform(feature_data)
            
            last_sequence = scaled_features[-self.dataset.sequence_length:]
            sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                signal_logits, confidence_score = self.model(sequence_tensor)
                signal_probs = torch.softmax(signal_logits, dim=1)
                ml_signal = torch.argmax(signal_probs, dim=1).item()
                ml_confidence = confidence_score.item()
            
            signals = ['buy', 'hold', 'sell']
            ml_signal_name = signals[ml_signal]
            
            traditional_signal, traditional_confidence, analysis = self.generate_traditional_signal(df, sr_levels, channel_width)
            
            if ml_signal_name == traditional_signal:
                final_signal = ml_signal_name
                final_confidence = 0.7 * ml_confidence + 0.3 * traditional_confidence
            else:
                if ml_confidence > traditional_confidence:
                    final_signal = ml_signal_name
                    final_confidence = ml_confidence * 0.8
                else:
                    final_signal = traditional_signal
                    final_confidence = traditional_confidence * 0.8
            
            analysis['ml_signal'] = ml_signal_name
            analysis['ml_confidence'] = ml_confidence
            analysis['traditional_signal'] = traditional_signal
            analysis['signal_agreement'] = ml_signal_name == traditional_signal
            
            return final_signal, final_confidence, analysis
        else:
            return self.generate_traditional_signal(df, sr_levels, channel_width)
    
    def generate_traditional_signal(self, df, sr_levels, channel_width):
        current_price = df['Close'].iloc[-1]
        macd_current = df['macd'].iloc[-1]
        signal_current = df['signal'].iloc[-1]
        hist_current = df['hist'].iloc[-1]
        hist_prev = df['hist'].iloc[-2] if len(df) > 1 else 0
        
        if sr_levels:
            distances = []
            for level_item in sr_levels:
                if isinstance(level_item, (tuple, list)) and len(level_item) >= 2:
                    level = level_item[0]
                    strength = level_item[1]
                else:
                    level = level_item
                    strength = 1
                
                distance = abs(current_price - level) / current_price
                distances.append((distance, level, strength))
            
            min_distance_info = min(distances, key=lambda x: x[0])
            min_distance = min_distance_info[0]
            nearest_level = (min_distance_info[1], min_distance_info[2])
        else:
            min_distance = 1.0
            nearest_level = (current_price, 1)
        
        proximity_threshold = channel_width / 100
        
        signal = "hold"
        confidence = 0.5
        
        if min_distance < proximity_threshold:
            if current_price < nearest_level[0] and hist_current < 0 and hist_current > hist_prev:
                signal = "buy"
                confidence = 0.7 + (nearest_level[1] * 0.05)
            elif current_price > nearest_level[0] and hist_current > 0 and hist_current < hist_prev:
                signal = "sell"
                confidence = 0.7 + (nearest_level[1] * 0.05)
        
        analysis = {
            'current_price': current_price,
            'nearest_sr_level': nearest_level[0],
            'distance_to_sr': min_distance * 100,
            'macd_signal': 'bullish' if macd_current > signal_current else 'bearish',
            'adaptive_params': {
                'pivot_period': self.volatility_system.adaptive_pivot_period(0),
                'channel_width_pct': channel_width,
                'proximity_threshold_pct': proximity_threshold * 100
            }
        }
        
        return signal, confidence, analysis
    
    def create_adaptive_visualization(self, df, sr_levels, ticker, pivot_period, channel_width):
        fig, ax = plt.subplots(figsize=(15, 8))
        
        buy_colors = {
            4: '#3ea923', 5: '#37981f', 6: '#31871c', 7: '#2b7718', 
            8: '#256615', 9: '#1f5511', 10: '#19440e'
        }
        sell_colors = {
            4: '#a71116', 5: '#950f13', 6: '#820d11', 7: '#6f0b0e', 
            8: '#59090c', 9: '#4a070a', 10: '#380607'
        }
        
        for idx, row in df.iterrows():
            i = df.index.get_loc(idx)

            if row['Close'] >= row['Open']:
                color = 'green' 
                if row['buy_count'] >= 4:
                    color = buy_colors.get(min(int(row['buy_count']), 10))
                rect = Rectangle((i, row['Open']), 0.6, row['Close'] - row['Open'], 
                                fill=True, color=color, alpha=0.8)
                ax.add_patch(rect)
            else:
                color = 'red'
                if row['sell_count'] >= 4:
                    color = sell_colors.get(min(int(row['sell_count']), 10))
                rect = Rectangle((i, row['Close']), 0.6, row['Open'] - row['Close'], 
                                fill=True, color=color, alpha=0.8)
                ax.add_patch(rect)
            
            ax.plot([i+0.3, i+0.3], [row['Low'], row['High']], color='black', linewidth=1)

        latest_close = df['Close'].iloc[-1]
        
        for level_item in sr_levels[:5]:
            if isinstance(level_item, (tuple, list)) and len(level_item) >= 2:
                level = level_item[0]
                strength = level_item[1]
            else:
                level = level_item
                strength = 1
                
            ax.axhline(y=level, color='blue', linestyle='-', linewidth=1, alpha=0.7)
            pct_change = 100 * (level - latest_close) / latest_close
            ax.text(len(df) + 1, level, f"{level:.2f} ({pct_change:.2f}%)", color='black')
        
        ax.set_title(f'{ticker} - ML-Enhanced Adaptive Pivot Analysis\n'
                     f'Adaptive Pivot Period: {pivot_period}, Channel Width: {channel_width:.1f}%')
        ax.set_ylabel('Price')
        ax.set_xlabel('Date')
        
        step = max(1, len(df) // 10)
        ax.set_xticks(range(0, len(df), step))
        ax.set_xticklabels([df.index[i].strftime('%Y-%m-%d') for i in range(0, len(df), step)], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

def run_training_and_analysis(ticker, train_first=True, epochs=100):
    system = EnhancedAdaptiveTradingSystem()
    
    if train_first:
        training_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        print("Training model on multiple stocks WITHOUT early stopping...")
        system.train_on_multiple_stocks(training_tickers, epochs=epochs)
    
    print(f"\nAnalyzing {ticker} with trained model...")
    result = system.adaptive_dca_indicator(ticker)
    
    if result[0] is not None:
        df, sr_levels, fig, ax, signal, confidence, analysis = result
        
        print(f"\n{'='*60}")
        print(f"TRAINED ML TRADING ANALYSIS - {ticker}")
        print(f"{'='*60}")
        print(f"Final Signal: {signal.upper()}")
        print(f"Final Confidence: {confidence*100:.1f}%")
        print(f"ML Signal: {analysis.get('ml_signal', 'N/A')}")
        print(f"ML Confidence: {analysis.get('ml_confidence', 0)*100:.1f}%")
        print(f"Traditional Signal: {analysis.get('traditional_signal', 'N/A')}")
        print(f"Signals Agree: {analysis.get('signal_agreement', False)}")
        print(f"Current Price: ${analysis['current_price']:.2f}")
        print(f"Nearest S/R Level: ${analysis['nearest_sr_level']:.2f}")
        print(f"Distance to S/R: {analysis['distance_to_sr']:.1f}%")
        print(f"MACD Signal: {analysis['macd_signal']}")
        
        plt.show()
        return result
    else:
        print("Failed to retrieve data")
        return None

if __name__ == "__main__":
    result = run_training_and_analysis("AAPL", train_first=True, epochs=100)
