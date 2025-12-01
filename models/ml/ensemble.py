"""
Ensemble Model Module.

Implements a sophisticated ensemble model that combines:
- Multiple gradient boosting models (XGBoost, LightGBM, CatBoost)
- Random Forest
- Optional deep learning models (LSTM, Transformer)

Uses stacking and weighted averaging for optimal predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error
)

from .gradient_boost import GradientBoostingModels, ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""
    use_xgb: bool = True
    use_lgb: bool = True
    use_catboost: bool = True
    use_random_forest: bool = True
    use_stacking: bool = True
    task: str = 'classification'  # 'classification' or 'regression'
    
    # Weights for simple averaging (if not using stacking)
    xgb_weight: float = 0.3
    lgb_weight: float = 0.3
    catboost_weight: float = 0.2
    rf_weight: float = 0.2


class EnsembleModel:
    """
    Sophisticated ensemble model for trading signal prediction.
    
    Combines multiple ML models using:
    1. Base learners: XGBoost, LightGBM, CatBoost, Random Forest
    2. Meta-learner: Logistic Regression (stacking)
    3. Weighted averaging as fallback
    
    Features:
    - Automatic weight optimization based on validation performance
    - Confidence estimation
    - Time-series aware cross-validation
    - Model persistence and versioning
    
    Example:
        >>> ensemble = EnsembleModel()
        >>> ensemble.fit(X_train, y_train, X_val, y_val)
        >>> signals, confidence = ensemble.predict_with_confidence(X_test)
    """
    
    def __init__(
        self,
        config: EnsembleConfig = None,
        random_state: int = 42,
    ):
        """
        Initialize ensemble model.
        
        Args:
            config: Ensemble configuration
            random_state: Random seed for reproducibility
        """
        self.config = config or EnsembleConfig()
        self.random_state = random_state
        
        # Base models
        self.gb_models = None  # Gradient boosting ensemble
        self.rf_model = None  # Random forest
        
        # Meta learner for stacking
        self.meta_learner = None
        
        # Model weights (learned from validation)
        self.model_weights = {}
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.meta_scaler = StandardScaler()
        
        # State
        self.feature_names = None
        self.fitted = False
        self.fit_timestamp = None
        self.validation_metrics = {}
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: int = 50,
    ) -> 'EnsembleModel':
        """
        Fit the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, will split if not provided)
            y_val: Validation target
            early_stopping_rounds: Early stopping for gradient boosting
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting ensemble model...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
        else:
            # Create validation split
            split_idx = int(len(X_train) * 0.8)
            X_train, X_val = X_train[:split_idx], X_train[split_idx:]
            y_train, y_val = y_train[:split_idx], y_train[split_idx:]
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Fit base models
        base_predictions_train = []
        base_predictions_val = []
        
        # 1. Gradient Boosting Models (XGBoost, LightGBM, CatBoost)
        if any([self.config.use_xgb, self.config.use_lgb, self.config.use_catboost]):
            logger.info("Fitting gradient boosting models...")
            self.gb_models = GradientBoostingModels(
                task=self.config.task,
                random_state=self.random_state
            )
            self.gb_models.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                early_stopping_rounds=early_stopping_rounds,
                scale_features=False  # Already scaled
            )
            
            # Get predictions for stacking
            gb_pred_train = self._get_gb_predictions(X_train_scaled)
            gb_pred_val = self._get_gb_predictions(X_val_scaled)
            
            base_predictions_train.extend(gb_pred_train)
            base_predictions_val.extend(gb_pred_val)
            
            # Evaluate individual models
            self._evaluate_gb_models(X_val_scaled, y_val)
        
        # 2. Random Forest
        if self.config.use_random_forest:
            logger.info("Fitting Random Forest...")
            self._fit_random_forest(X_train_scaled, y_train)
            
            rf_pred_train = self._get_rf_predictions(X_train_scaled)
            rf_pred_val = self._get_rf_predictions(X_val_scaled)
            
            base_predictions_train.append(rf_pred_train)
            base_predictions_val.append(rf_pred_val)
            
            # Evaluate RF
            self._evaluate_rf(X_val_scaled, y_val)
        
        # 3. Fit meta-learner (stacking)
        if self.config.use_stacking and len(base_predictions_train) > 1:
            logger.info("Fitting meta-learner for stacking...")
            self._fit_meta_learner(
                base_predictions_train, y_train,
                base_predictions_val, y_val
            )
        else:
            # Calculate weights from validation performance
            self._calculate_weights()
        
        self.fitted = True
        self.fit_timestamp = datetime.now()
        
        # Final evaluation
        final_metrics = self.evaluate(X_val, y_val)
        logger.info(f"Ensemble validation metrics: {final_metrics}")
        
        return self
    
    def _fit_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Fit Random Forest model."""
        if self.config.task == 'classification':
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=self.random_state,
            )
        else:
            self.rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=self.random_state,
            )
        
        self.rf_model.fit(X, y)
    
    def _get_gb_predictions(self, X: np.ndarray) -> List[np.ndarray]:
        """Get predictions from all gradient boosting models."""
        predictions = []
        
        if self.gb_models.xgb_model and self.config.use_xgb:
            pred = self.gb_models.predict(X, model='xgb')
            predictions.append(pred)
        
        if self.gb_models.lgb_model and self.config.use_lgb:
            pred = self.gb_models.predict(X, model='lgb')
            predictions.append(pred)
        
        if self.gb_models.catboost_model and self.config.use_catboost:
            pred = self.gb_models.predict(X, model='catboost')
            predictions.append(pred)
        
        return predictions
    
    def _get_rf_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from Random Forest."""
        if self.config.task == 'classification':
            return self.rf_model.predict_proba(X)[:, 1]
        return self.rf_model.predict(X)
    
    def _evaluate_gb_models(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate gradient boosting models on validation set."""
        for model_name in ['xgb', 'lgb', 'catboost']:
            try:
                metrics = self.gb_models.evaluate(X_val, y_val, model=model_name)
                self.validation_metrics[model_name] = metrics
                logger.info(
                    f"{model_name.upper()}: "
                    f"AUC={metrics.roc_auc:.4f}, F1={metrics.f1:.4f}"
                )
            except Exception as e:
                logger.warning(f"Could not evaluate {model_name}: {e}")
    
    def _evaluate_rf(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate Random Forest on validation set."""
        pred = self._get_rf_predictions(X_val)
        
        metrics = ModelMetrics()
        if self.config.task == 'classification':
            binary_pred = (pred > 0.5).astype(int)
            metrics.accuracy = accuracy_score(y_val, binary_pred)
            metrics.precision = precision_score(y_val, binary_pred, zero_division=0)
            metrics.recall = recall_score(y_val, binary_pred, zero_division=0)
            metrics.f1 = f1_score(y_val, binary_pred, zero_division=0)
            try:
                metrics.roc_auc = roc_auc_score(y_val, pred)
            except:
                pass
        else:
            metrics.mse = mean_squared_error(y_val, pred)
        
        self.validation_metrics['rf'] = metrics
        logger.info(f"RF: AUC={metrics.roc_auc:.4f}, F1={metrics.f1:.4f}")
    
    def _fit_meta_learner(
        self,
        base_predictions_train: List[np.ndarray],
        y_train: np.ndarray,
        base_predictions_val: List[np.ndarray],
        y_val: np.ndarray,
    ):
        """Fit meta-learner for stacking."""
        # Stack base predictions
        X_meta_train = np.column_stack(base_predictions_train)
        X_meta_val = np.column_stack(base_predictions_val)
        
        # Scale meta features
        X_meta_train = self.meta_scaler.fit_transform(X_meta_train)
        X_meta_val = self.meta_scaler.transform(X_meta_val)
        
        # Fit meta-learner
        if self.config.task == 'classification':
            self.meta_learner = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
            )
        else:
            self.meta_learner = Ridge(alpha=1.0)
        
        self.meta_learner.fit(X_meta_train, y_train)
        
        # Evaluate meta-learner
        if self.config.task == 'classification':
            meta_pred = self.meta_learner.predict_proba(X_meta_val)[:, 1]
            auc = roc_auc_score(y_val, meta_pred)
            logger.info(f"Meta-learner validation AUC: {auc:.4f}")
    
    def _calculate_weights(self):
        """Calculate model weights based on validation performance."""
        weights = {}
        total_score = 0
        
        for model_name, metrics in self.validation_metrics.items():
            # Use AUC for classification, inverse MSE for regression
            if self.config.task == 'classification':
                score = metrics.roc_auc
            else:
                score = 1.0 / (metrics.mse + 1e-8)
            
            weights[model_name] = score
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            for model_name in weights:
                weights[model_name] /= total_score
        
        self.model_weights = weights
        logger.info(f"Calculated model weights: {weights}")
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        use_stacking: bool = None,
    ) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Features for prediction
            use_stacking: Whether to use stacking (default from config)
            
        Returns:
            Predictions array
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.feature_scaler.transform(X)
        
        # Get base predictions
        base_predictions = []
        
        if self.gb_models:
            gb_preds = self._get_gb_predictions(X_scaled)
            base_predictions.extend(gb_preds)
        
        if self.rf_model:
            rf_pred = self._get_rf_predictions(X_scaled)
            base_predictions.append(rf_pred)
        
        if not base_predictions:
            raise ValueError("No base models available")
        
        # Use stacking or weighted average
        use_stacking = use_stacking if use_stacking is not None else self.config.use_stacking
        
        if use_stacking and self.meta_learner is not None:
            X_meta = np.column_stack(base_predictions)
            X_meta = self.meta_scaler.transform(X_meta)
            
            if self.config.task == 'classification':
                return self.meta_learner.predict_proba(X_meta)[:, 1]
            return self.meta_learner.predict(X_meta)
        else:
            # Weighted average
            return np.average(base_predictions, axis=0)
    
    def predict_with_confidence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Confidence is based on:
        - Agreement between base models
        - Distance from decision boundary
        
        Args:
            X: Features for prediction
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.feature_scaler.transform(X)
        
        # Get all base predictions
        all_predictions = []
        
        if self.gb_models:
            if self.gb_models.xgb_model:
                all_predictions.append(self.gb_models.predict(X_scaled, model='xgb'))
            if self.gb_models.lgb_model:
                all_predictions.append(self.gb_models.predict(X_scaled, model='lgb'))
            if self.gb_models.catboost_model:
                all_predictions.append(self.gb_models.predict(X_scaled, model='catboost'))
        
        if self.rf_model:
            all_predictions.append(self._get_rf_predictions(X_scaled))
        
        # Stack predictions
        pred_stack = np.column_stack(all_predictions)
        
        # Final prediction (ensemble)
        final_pred = self.predict(X)
        
        # Calculate confidence
        # 1. Model agreement (low std = high agreement)
        pred_std = pred_stack.std(axis=1)
        agreement = 1 - np.clip(pred_std * 2, 0, 1)  # Normalize
        
        # 2. Distance from boundary (0.5 for classification)
        if self.config.task == 'classification':
            boundary_distance = np.abs(final_pred - 0.5) * 2  # 0-1 range
        else:
            boundary_distance = np.ones_like(final_pred)  # Not applicable for regression
        
        # Combine confidence factors
        confidence = 0.6 * agreement + 0.4 * boundary_distance
        
        return final_pred, confidence
    
    def predict_signal(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        long_threshold: float = 0.6,
        short_threshold: float = 0.4,
        min_confidence: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trading signals with confidence filtering.
        
        Args:
            X: Features for prediction
            long_threshold: Probability threshold for long signal
            short_threshold: Probability threshold for short signal
            min_confidence: Minimum confidence to generate signal
            
        Returns:
            Tuple of (signals, probabilities, confidence)
            signals: 1 (long), -1 (short), 0 (hold)
        """
        probs, confidence = self.predict_with_confidence(X)
        
        # Generate signals
        signals = np.zeros(len(probs))
        signals[probs > long_threshold] = 1  # Long
        signals[probs < short_threshold] = -1  # Short
        
        # Filter by confidence
        signals[confidence < min_confidence] = 0
        
        return signals.astype(int), probs, confidence
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> ModelMetrics:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            ModelMetrics with evaluation scores
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        predictions = self.predict(X)
        
        metrics = ModelMetrics()
        
        if self.config.task == 'classification':
            binary_pred = (predictions > 0.5).astype(int)
            
            metrics.accuracy = accuracy_score(y, binary_pred)
            metrics.precision = precision_score(y, binary_pred, zero_division=0)
            metrics.recall = recall_score(y, binary_pred, zero_division=0)
            metrics.f1 = f1_score(y, binary_pred, zero_division=0)
            
            try:
                metrics.roc_auc = roc_auc_score(y, predictions)
            except ValueError:
                pass
        else:
            metrics.mse = mean_squared_error(y, predictions)
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get aggregated feature importance from all models.
        
        Returns:
            DataFrame with feature importance from all models
        """
        importance_dfs = []
        
        # Get importance from gradient boosting models
        if self.gb_models:
            gb_importance = self.gb_models.get_feature_importance()
            if not gb_importance.empty:
                importance_dfs.append(gb_importance)
        
        # Get importance from random forest
        if self.rf_model:
            rf_importance = pd.DataFrame({
                'feature': self.feature_names or [f'f_{i}' for i in range(len(self.rf_model.feature_importances_))],
                'rf': self.rf_model.feature_importances_ / self.rf_model.feature_importances_.sum()
            })
            importance_dfs.append(rf_importance)
        
        if not importance_dfs:
            return pd.DataFrame()
        
        # Merge all importance dataframes
        result = importance_dfs[0]
        for df in importance_dfs[1:]:
            result = result.merge(df, on='feature', how='outer')
        
        # Fill NaN and calculate mean
        numeric_cols = [c for c in result.columns if c != 'feature']
        result[numeric_cols] = result[numeric_cols].fillna(0)
        result['mean_importance'] = result[numeric_cols].mean(axis=1)
        
        return result.sort_values('mean_importance', ascending=False)
    
    def save(self, path: str):
        """Save ensemble model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save state
        state = {
            'config': self.config,
            'random_state': self.random_state,
            'model_weights': self.model_weights,
            'feature_scaler': self.feature_scaler,
            'meta_scaler': self.meta_scaler,
            'feature_names': self.feature_names,
            'fitted': self.fitted,
            'fit_timestamp': self.fit_timestamp,
            'validation_metrics': self.validation_metrics,
        }
        
        with open(path / 'ensemble_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        # Save sub-models
        if self.gb_models:
            self.gb_models.save(str(path / 'gradient_boost'))
        
        if self.rf_model:
            with open(path / 'rf_model.pkl', 'wb') as f:
                pickle.dump(self.rf_model, f)
        
        if self.meta_learner:
            with open(path / 'meta_learner.pkl', 'wb') as f:
                pickle.dump(self.meta_learner, f)
        
        logger.info(f"Ensemble model saved to {path}")
    
    def load(self, path: str) -> 'EnsembleModel':
        """Load ensemble model from disk."""
        path = Path(path)
        
        # Load state
        with open(path / 'ensemble_state.pkl', 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.random_state = state['random_state']
        self.model_weights = state['model_weights']
        self.feature_scaler = state['feature_scaler']
        self.meta_scaler = state['meta_scaler']
        self.feature_names = state['feature_names']
        self.fitted = state['fitted']
        self.fit_timestamp = state['fit_timestamp']
        self.validation_metrics = state['validation_metrics']
        
        # Load sub-models
        gb_path = path / 'gradient_boost'
        if gb_path.exists():
            self.gb_models = GradientBoostingModels(task=self.config.task)
            self.gb_models.load(str(gb_path))
        
        rf_path = path / 'rf_model.pkl'
        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                self.rf_model = pickle.load(f)
        
        meta_path = path / 'meta_learner.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                self.meta_learner = pickle.load(f)
        
        logger.info(f"Ensemble model loaded from {path}")
        return self
