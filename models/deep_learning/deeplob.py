"""
DeepLOB: Deep Learning Model for Limit Order Book.

Implementation of the DeepLOB architecture from your research:
- CNN layers for spatial feature extraction from LOB
- Inception modules for multi-scale temporal patterns
- LSTM layers for temporal dependencies
- Predicts mid-price direction: Up, Down, Stationary

References:
- "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
- Your NotebookLLM HFT research (53+ sources)

Note: This is a research/backtesting implementation.
Production HFT would use optimized C++/CUDA implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import deep learning frameworks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. DeepLOB requires: pip install torch")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LOBData:
    """
    Limit Order Book data structure for DeepLOB.
    
    Shape: (T, levels * 4)
    - T: number of time steps
    - levels: number of price levels (e.g., 10)
    - 4 features per level: bid_price, bid_size, ask_price, ask_size
    """
    timestamps: np.ndarray
    bid_prices: np.ndarray  # Shape: (T, levels)
    bid_sizes: np.ndarray   # Shape: (T, levels)
    ask_prices: np.ndarray  # Shape: (T, levels)
    ask_sizes: np.ndarray   # Shape: (T, levels)
    mid_prices: np.ndarray  # Shape: (T,)
    
    @property
    def n_levels(self) -> int:
        return self.bid_prices.shape[1]
    
    @property
    def n_samples(self) -> int:
        return len(self.timestamps)
    
    def to_features(self) -> np.ndarray:
        """
        Convert to feature matrix for DeepLOB.
        
        Returns:
            Shape: (T, 4 * levels)
        """
        return np.concatenate([
            self.bid_prices,
            self.bid_sizes,
            self.ask_prices,
            self.ask_sizes,
        ], axis=1)
    
    def get_labels(
        self,
        horizon: int = 10,
        threshold: float = 0.0002,  # 2 basis points
    ) -> np.ndarray:
        """
        Generate labels for mid-price direction prediction.
        
        Args:
            horizon: Number of steps ahead to predict
            threshold: Minimum change to classify as Up/Down
            
        Returns:
            Labels: 0 = Down, 1 = Stationary, 2 = Up
        """
        future_mid = np.roll(self.mid_prices, -horizon)
        
        # Calculate percentage change
        pct_change = (future_mid - self.mid_prices) / (self.mid_prices + 1e-10)
        
        # Classify
        labels = np.ones(len(pct_change), dtype=int)  # Default: Stationary (1)
        labels[pct_change > threshold] = 2  # Up
        labels[pct_change < -threshold] = 0  # Down
        
        # Last 'horizon' labels are invalid
        labels[-horizon:] = 1
        
        return labels


# =============================================================================
# LOB DATA GENERATOR (FROM OHLCV PROXY)
# =============================================================================

class LOBDataGenerator:
    """
    Generate synthetic LOB data from OHLCV data.
    
    In production, you'd use real Level 2/3 market data feeds.
    This provides a proxy for research and backtesting.
    """
    
    def __init__(
        self,
        n_levels: int = 10,
        base_spread_bps: float = 5,  # 5 basis points
        tick_size: float = 0.01,
    ):
        self.n_levels = n_levels
        self.base_spread_bps = base_spread_bps
        self.tick_size = tick_size
    
    def generate_from_ohlcv(
        self,
        df: pd.DataFrame,
    ) -> LOBData:
        """
        Generate synthetic LOB data from OHLCV dataframe.
        """
        n_samples = len(df)
        
        bid_prices = np.zeros((n_samples, self.n_levels))
        bid_sizes = np.zeros((n_samples, self.n_levels))
        ask_prices = np.zeros((n_samples, self.n_levels))
        ask_sizes = np.zeros((n_samples, self.n_levels))
        mid_prices = np.zeros(n_samples)
        
        for i in range(n_samples):
            row = df.iloc[i]
            close = row['close']
            high = row['high']
            low = row['low']
            volume = row['volume']
            open_p = row['open']
            
            # Estimate spread from high-low range
            range_pct = (high - low) / close
            spread = max(close * self.base_spread_bps / 10000, close * range_pct * 0.1)
            
            # Mid price
            mid = (high + low + close) / 3
            mid_prices[i] = mid
            
            # Best bid/ask
            best_bid = mid - spread / 2
            best_ask = mid + spread / 2
            
            # Range position indicates buying/selling pressure
            if high != low:
                range_pos = (close - low) / (high - low)
            else:
                range_pos = 0.5
            
            # Generate price levels
            for level in range(self.n_levels):
                tick_offset = (level + 1) * self.tick_size
                
                bid_prices[i, level] = best_bid - tick_offset
                ask_prices[i, level] = best_ask + tick_offset
            
            # Generate sizes (volume distributed across levels)
            avg_size = volume / (2 * self.n_levels)
            
            # Buying pressure = larger bid sizes
            bid_multiplier = 0.5 + range_pos
            ask_multiplier = 1.5 - range_pos
            
            for level in range(self.n_levels):
                # Exponential decay in size away from best price
                decay = np.exp(-level * 0.3)
                bid_sizes[i, level] = avg_size * bid_multiplier * decay
                ask_sizes[i, level] = avg_size * ask_multiplier * decay
        
        return LOBData(
            timestamps=df.index.values if hasattr(df.index, 'values') else np.arange(n_samples),
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            mid_prices=mid_prices,
        )


# =============================================================================
# PYTORCH DATASET
# =============================================================================

if TORCH_AVAILABLE:
    
    class LOBDataset(Dataset):
        """
        PyTorch Dataset for LOB data.
        
        Prepares sequences for DeepLOB model.
        """
        
        def __init__(
            self,
            lob_data: LOBData,
            sequence_length: int = 100,
            horizon: int = 10,
            threshold: float = 0.0002,
        ):
            """
            Args:
                lob_data: LOBData object
                sequence_length: Number of time steps per sample (T)
                horizon: Prediction horizon
                threshold: Classification threshold
            """
            self.features = lob_data.to_features()  # (N, 4*levels)
            self.labels = lob_data.get_labels(horizon, threshold)
            self.sequence_length = sequence_length
            
            # Normalize features
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            
            # Valid indices
            self.n_samples = len(self.features) - sequence_length - horizon
        
        def __len__(self) -> int:
            return max(0, self.n_samples)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Get a single sample.
            
            Returns:
                x: Shape (T, n_features) = (sequence_length, 4*levels)
                y: Scalar label (0, 1, or 2)
            """
            start_idx = idx
            end_idx = idx + self.sequence_length
            
            x = self.features[start_idx:end_idx]
            y = self.labels[end_idx]
            
            return torch.FloatTensor(x), torch.LongTensor([y])[0]


# =============================================================================
# DEEPLOB MODEL
# =============================================================================

if TORCH_AVAILABLE:
    
    class InceptionModule(nn.Module):
        """
        Inception Module for multi-scale feature extraction.
        
        Wraps convolutions with different filter sizes to capture
        patterns at multiple time scales (like different moving averages).
        """
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
        ):
            super().__init__()
            
            # 1x1 convolution
            self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
            
            # 1x3 convolution (short-term patterns)
            self.conv1x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 3), padding=(0, 1))
            
            # 1x5 convolution (medium-term patterns)
            self.conv1x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 5), padding=(0, 2))
            
            # Max pooling branch
            self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1))
            self.conv_pool = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
            
            self.bn = nn.BatchNorm2d(out_channels)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Shape (batch, channels, height, width)
            """
            branch1 = self.conv1x1(x)
            branch2 = self.conv1x3(x)
            branch3 = self.conv1x5(x)
            branch4 = self.conv_pool(self.pool(x))
            
            out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
            return F.relu(self.bn(out))
    
    
    class DeepLOB(nn.Module):
        """
        DeepLOB: Deep Convolutional Neural Network for LOB.
        
        Architecture (from your research):
        1. CNN layers - spatial feature extraction across LOB levels
        2. Inception modules - multi-scale temporal patterns
        3. LSTM layers - long-term temporal dependencies
        4. Fully connected - classification
        
        Input: (batch, T, n_features) where n_features = 4 * n_levels
        Output: (batch, 3) - probabilities for Down, Stationary, Up
        """
        
        def __init__(
            self,
            n_features: int = 40,  # 4 * 10 levels
            hidden_dim: int = 64,
            lstm_hidden: int = 64,
            n_lstm_layers: int = 2,
            n_classes: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            
            self.n_features = n_features
            self.hidden_dim = hidden_dim
            
            # Reshape: (batch, T, features) -> (batch, 1, T, features)
            # Treat as 2D "image" with height=T, width=features
            
            # Initial CNN layers for spatial features (across LOB levels)
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2))
            self.bn1 = nn.BatchNorm2d(32)
            
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(4, 1))
            self.bn2 = nn.BatchNorm2d(32)
            
            self.conv3 = nn.Conv2d(32, 32, kernel_size=(4, 1))
            self.bn3 = nn.BatchNorm2d(32)
            
            # Inception modules for multi-scale features
            self.inception1 = InceptionModule(32, 64)
            self.inception2 = InceptionModule(64, 64)
            
            # Calculate LSTM input size dynamically
            # This depends on the convolution operations
            self._lstm_input_size = None  # Will be set during first forward pass
            
            # LSTM for temporal dependencies
            self.lstm = nn.LSTM(
                input_size=64,  # Will be adjusted
                hidden_size=lstm_hidden,
                num_layers=n_lstm_layers,
                batch_first=True,
                dropout=dropout if n_lstm_layers > 1 else 0,
                bidirectional=False,
            )
            
            # Fully connected layers
            self.fc1 = nn.Linear(lstm_hidden, 32)
            self.fc2 = nn.Linear(32, n_classes)
            
            self.dropout = nn.Dropout(dropout)
            
            # For dynamic shape calculation
            self._initialized = False
        
        def _init_lstm_size(self, x: torch.Tensor):
            """Initialize LSTM input size based on actual feature dimensions."""
            if self._initialized:
                return
            
            with torch.no_grad():
                # Forward through CNN layers
                x = x.unsqueeze(1)  # (batch, 1, T, features)
                x = F.leaky_relu(self.bn1(self.conv1(x)))
                x = F.leaky_relu(self.bn2(self.conv2(x)))
                x = F.leaky_relu(self.bn3(self.conv3(x)))
                x = self.inception1(x)
                x = self.inception2(x)
                
                # Shape: (batch, channels, height, width)
                # For LSTM: (batch, seq_len, features)
                batch, channels, height, width = x.shape
                
                # Update LSTM
                self.lstm = nn.LSTM(
                    input_size=channels * width,
                    hidden_size=self.lstm.hidden_size,
                    num_layers=self.lstm.num_layers,
                    batch_first=True,
                    dropout=self.lstm.dropout,
                ).to(x.device)
                
                self._initialized = True
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor, shape (batch, T, n_features)
                
            Returns:
                Output logits, shape (batch, n_classes)
            """
            # Initialize LSTM size if needed
            self._init_lstm_size(x)
            
            # Add channel dimension: (batch, 1, T, features)
            x = x.unsqueeze(1)
            
            # CNN layers
            x = F.leaky_relu(self.bn1(self.conv1(x)))
            x = F.leaky_relu(self.bn2(self.conv2(x)))
            x = F.leaky_relu(self.bn3(self.conv3(x)))
            
            # Inception modules
            x = self.inception1(x)
            x = self.inception2(x)
            
            # Reshape for LSTM: (batch, seq_len, features)
            batch, channels, height, width = x.shape
            x = x.permute(0, 2, 1, 3)  # (batch, height, channels, width)
            x = x.reshape(batch, height, channels * width)
            
            # LSTM
            lstm_out, _ = self.lstm(x)
            
            # Use last hidden state
            x = lstm_out[:, -1, :]
            
            # Fully connected
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            
            return x
        
        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """Get class probabilities."""
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
        
        def predict(self, x: torch.Tensor) -> torch.Tensor:
            """Get predicted class."""
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)


# =============================================================================
# DEEPLOB TRAINER
# =============================================================================

class DeepLOBTrainer:
    """
    Training and evaluation utilities for DeepLOB.
    """
    
    def __init__(
        self,
        n_features: int = 40,
        hidden_dim: int = 64,
        lstm_hidden: int = 64,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 50,
        device: str = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for DeepLOB. Install with: pip install torch")
        
        self.n_features = n_features
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Model
        self.model = DeepLOB(
            n_features=n_features,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
        ).to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
    
    def train(
        self,
        train_dataset: 'LOBDataset',
        val_dataset: Optional['LOBDataset'] = None,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the DeepLOB model.
        
        Args:
            train_dataset: Training LOBDataset
            val_dataset: Validation LOBDataset (optional)
            early_stopping_patience: Epochs without improvement before stopping
            
        Returns:
            Training history dict
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            val_loss = 0
            val_acc = 0
            
            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
            
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.n_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                    + (f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if val_loader else "")
                )
        
        return self.history
    
    def evaluate(
        self,
        test_dataset: 'LOBDataset',
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Returns:
            Dict with accuracy, F1, etc.
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        logger.info(f"Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(
            all_labels, all_preds,
            target_names=['Down', 'Stationary', 'Up']
        ))
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels,
        }
    
    def predict(
        self,
        lob_data: LOBData,
        sequence_length: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new LOB data.
        
        Args:
            lob_data: LOBData object
            sequence_length: Same as training
            
        Returns:
            (predictions, probabilities)
        """
        # Prepare data
        dataset = LOBDataset(
            lob_data,
            sequence_length=sequence_length,
            horizon=1,  # Doesn't matter for prediction
        )
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                
                outputs = self.model(batch_x)
                probs = F.softmax(outputs, dim=-1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'n_features': self.n_features,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Model loaded from {path}")


# =============================================================================
# BAYESIAN DEEPLOB (UNCERTAINTY ESTIMATION)
# =============================================================================

if TORCH_AVAILABLE:
    
    class BayesianDeepLOB(DeepLOB):
        """
        Bayesian extension of DeepLOB (BDLOB).
        
        From your research:
        - Provides uncertainty measures on outputs
        - Can upsize positions based on model certainty
        - Uses MC Dropout for uncertainty estimation
        """
        
        def __init__(
            self,
            n_features: int = 40,
            hidden_dim: int = 64,
            lstm_hidden: int = 64,
            n_classes: int = 3,
            dropout: float = 0.3,  # Higher dropout for Bayesian inference
        ):
            super().__init__(
                n_features=n_features,
                hidden_dim=hidden_dim,
                lstm_hidden=lstm_hidden,
                n_classes=n_classes,
                dropout=dropout,
            )
        
        def predict_with_uncertainty(
            self,
            x: torch.Tensor,
            n_samples: int = 30,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Make predictions with uncertainty estimation using MC Dropout.
            
            Args:
                x: Input tensor
                n_samples: Number of stochastic forward passes
                
            Returns:
                (mean_prediction, std_prediction, entropy)
            """
            self.train()  # Enable dropout
            
            predictions = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    probs = self.predict_proba(x)
                    predictions.append(probs)
            
            # Stack and compute statistics
            predictions = torch.stack(predictions, dim=0)  # (n_samples, batch, n_classes)
            
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
            
            # Entropy as uncertainty measure
            entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=-1)
            
            return mean_pred, std_pred, entropy
        
        def get_confidence_adjusted_signal(
            self,
            x: torch.Tensor,
            base_position_size: float = 1.0,
            uncertainty_threshold: float = 0.5,
        ) -> Tuple[int, float, float]:
            """
            Get trading signal with position size adjusted by confidence.
            
            From your research:
            - Upsize positions when model is certain
            - Reduce/skip positions when uncertain
            
            Returns:
                (direction, position_size, confidence)
            """
            mean_pred, std_pred, entropy = self.predict_with_uncertainty(x)
            
            # Get predicted class
            predicted_class = torch.argmax(mean_pred, dim=-1).item()
            
            # Map class to direction: 0=Down=-1, 1=Stationary=0, 2=Up=1
            direction_map = {0: -1, 1: 0, 2: 1}
            direction = direction_map[predicted_class]
            
            # Confidence based on entropy (lower entropy = higher confidence)
            max_entropy = np.log(3)  # Maximum entropy for 3 classes
            confidence = 1 - (entropy.mean().item() / max_entropy)
            
            # Adjust position size
            if entropy.mean().item() > uncertainty_threshold:
                # High uncertainty - reduce position
                position_size = base_position_size * 0.25
            elif confidence > 0.8:
                # High confidence - increase position
                position_size = base_position_size * 1.5
            else:
                position_size = base_position_size * confidence
            
            return direction, min(position_size, base_position_size * 2), confidence


# =============================================================================
# DEEPLOB SIGNAL GENERATOR
# =============================================================================

class DeepLOBSignalGenerator:
    """
    Generate trading signals using DeepLOB model.
    
    Integrates with the HFT strategy framework.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_levels: int = 10,
        sequence_length: int = 100,
        use_bayesian: bool = False,
    ):
        """
        Args:
            model_path: Path to saved model (optional)
            n_levels: Number of LOB levels
            sequence_length: Input sequence length
            use_bayesian: Whether to use Bayesian uncertainty
        """
        self.n_levels = n_levels
        self.n_features = 4 * n_levels
        self.sequence_length = sequence_length
        self.use_bayesian = use_bayesian
        
        self.lob_generator = LOBDataGenerator(n_levels=n_levels)
        self.trainer = None
        self.model_trained = False
        
        if model_path and TORCH_AVAILABLE:
            self.load_model(model_path)
    
    def train_model(
        self,
        df: pd.DataFrame,
        val_split: float = 0.2,
        n_epochs: int = 30,
    ):
        """
        Train DeepLOB model on OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            val_split: Validation split fraction
            n_epochs: Number of training epochs
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for DeepLOB training")
            return
        
        # Generate LOB data
        lob_data = self.lob_generator.generate_from_ohlcv(df)
        
        # Create dataset
        full_dataset = LOBDataset(
            lob_data,
            sequence_length=self.sequence_length,
        )
        
        # Split
        split_idx = int(len(full_dataset) * (1 - val_split))
        
        # Create separate datasets (time-series split)
        train_lob = LOBData(
            timestamps=lob_data.timestamps[:split_idx],
            bid_prices=lob_data.bid_prices[:split_idx],
            bid_sizes=lob_data.bid_sizes[:split_idx],
            ask_prices=lob_data.ask_prices[:split_idx],
            ask_sizes=lob_data.ask_sizes[:split_idx],
            mid_prices=lob_data.mid_prices[:split_idx],
        )
        
        val_lob = LOBData(
            timestamps=lob_data.timestamps[split_idx:],
            bid_prices=lob_data.bid_prices[split_idx:],
            bid_sizes=lob_data.bid_sizes[split_idx:],
            ask_prices=lob_data.ask_prices[split_idx:],
            ask_sizes=lob_data.ask_sizes[split_idx:],
            mid_prices=lob_data.mid_prices[split_idx:],
        )
        
        train_dataset = LOBDataset(train_lob, sequence_length=self.sequence_length)
        val_dataset = LOBDataset(val_lob, sequence_length=self.sequence_length)
        
        # Initialize trainer
        self.trainer = DeepLOBTrainer(
            n_features=self.n_features,
            n_epochs=n_epochs,
        )
        
        # Train
        self.trainer.train(train_dataset, val_dataset)
        self.model_trained = True
        
        logger.info("DeepLOB model trained successfully")
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int = -1,
    ) -> Dict[str, Any]:
        """
        Generate trading signal from current market state.
        
        Returns:
            Dict with direction, confidence, probabilities
        """
        if not self.model_trained or self.trainer is None:
            return {
                'direction': 0,
                'confidence': 0,
                'probabilities': [0.33, 0.34, 0.33],
                'reasoning': 'Model not trained',
            }
        
        if current_idx == -1:
            current_idx = len(df) - 1
        
        if current_idx < self.sequence_length:
            return {
                'direction': 0,
                'confidence': 0,
                'probabilities': [0.33, 0.34, 0.33],
                'reasoning': 'Insufficient data for sequence',
            }
        
        # Get recent data
        start_idx = current_idx - self.sequence_length - 10
        end_idx = current_idx + 1
        recent_df = df.iloc[start_idx:end_idx]
        
        # Generate LOB data
        lob_data = self.lob_generator.generate_from_ohlcv(recent_df)
        
        # Make prediction
        predictions, probabilities = self.trainer.predict(lob_data, self.sequence_length)
        
        if len(predictions) == 0:
            return {
                'direction': 0,
                'confidence': 0,
                'probabilities': [0.33, 0.34, 0.33],
                'reasoning': 'No predictions generated',
            }
        
        # Get latest prediction
        latest_pred = predictions[-1]
        latest_probs = probabilities[-1]
        
        # Map to direction: 0=Down=-1, 1=Stationary=0, 2=Up=1
        direction_map = {0: -1, 1: 0, 2: 1}
        direction = direction_map[latest_pred]
        
        # Confidence = max probability
        confidence = float(np.max(latest_probs))
        
        # Reasoning
        class_names = ['DOWN', 'STATIONARY', 'UP']
        reasoning = f"DeepLOB: {class_names[latest_pred]} (p={confidence:.2%})"
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': latest_probs.tolist(),
            'reasoning': reasoning,
            'predicted_class': int(latest_pred),
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.trainer:
            self.trainer.save(path)
    
    def load_model(self, path: str):
        """Load trained model."""
        self.trainer = DeepLOBTrainer(n_features=self.n_features)
        self.trainer.load(path)
        self.model_trained = True


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_deeplob_features(
    df: pd.DataFrame,
    n_levels: int = 10,
) -> pd.DataFrame:
    """
    Add DeepLOB-derived features to a DataFrame.
    
    This can be used alongside other feature engineering pipelines.
    """
    generator = LOBDataGenerator(n_levels=n_levels)
    lob_data = generator.generate_from_ohlcv(df)
    
    # Add LOB features to dataframe
    df = df.copy()
    
    # Imbalance features
    for level in range(min(3, n_levels)):
        bid_size = lob_data.bid_sizes[:, level]
        ask_size = lob_data.ask_sizes[:, level]
        
        imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-10)
        df[f'lob_imbalance_L{level+1}'] = imbalance
    
    # Spread features
    spread = lob_data.ask_prices[:, 0] - lob_data.bid_prices[:, 0]
    df['lob_spread'] = spread
    df['lob_spread_bps'] = spread / lob_data.mid_prices * 10000
    
    # Depth features
    total_bid_depth = lob_data.bid_sizes.sum(axis=1)
    total_ask_depth = lob_data.ask_sizes.sum(axis=1)
    df['lob_bid_depth'] = total_bid_depth
    df['lob_ask_depth'] = total_ask_depth
    df['lob_depth_ratio'] = total_bid_depth / (total_ask_depth + 1e-10)
    
    # Mid-price momentum
    mid = lob_data.mid_prices
    df['lob_mid_price'] = mid
    df['lob_mid_return'] = pd.Series(mid).pct_change().values
    
    return df
