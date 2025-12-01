"""
Gradient Boosting Models Module.

Implements XGBoost, LightGBM, and CatBoost models for
trading signal prediction with hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

# ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    mse: float = 0.0
    mae: float = 0.0


class GradientBoostingModels:
    """
    Gradient Boosting models for trading signal prediction.
    
    Supports:
    - XGBoost: Extreme Gradient Boosting
    - LightGBM: Light Gradient Boosting Machine
    - CatBoost: Categorical Boosting
    
    Features:
    - Hyperparameter optimization with Optuna
    - Time-series cross-validation
    - Feature importance analysis
    - Model persistence
    
    Example:
        >>> models = GradientBoostingModels(task='classification')
        >>> models.fit(X_train, y_train)
        >>> predictions = models.predict(X_test)
    """
    
    def __init__(
        self,
        task: str = 'classification',
        xgb_params: Dict[str, Any] = None,
        lgb_params: Dict[str, Any] = None,
        catboost_params: Dict[str, Any] = None,
        random_state: int = 42,
    ):
        """
        Initialize gradient boosting models.
        
        Args:
            task: 'classification' for signals, 'regression' for returns
            xgb_params: XGBoost hyperparameters
            lgb_params: LightGBM hyperparameters
            catboost_params: CatBoost hyperparameters
            random_state: Random seed for reproducibility
        """
        self.task = task
        self.random_state = random_state
        
        # Default parameters
        self.xgb_params = xgb_params or self._default_xgb_params()
        self.lgb_params = lgb_params or self._default_lgb_params()
        self.catboost_params = catboost_params or self._default_catboost_params()
        
        # Models
        self.xgb_model = None
        self.lgb_model = None
        self.catboost_model = None
        
        # Scalers and feature names
        self.scaler = StandardScaler()
        self.feature_names = None
        self.fitted = False
        self.scale_features = True  # Track if scaling was used during fit
        
    def _default_xgb_params(self) -> Dict[str, Any]:
        """Default XGBoost parameters."""
        base_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        if self.task == 'classification':
            base_params['objective'] = 'binary:logistic'
            base_params['eval_metric'] = 'auc'
        else:
            base_params['objective'] = 'reg:squarederror'
            base_params['eval_metric'] = 'rmse'
        
        return base_params
    
    def _default_lgb_params(self) -> Dict[str, Any]:
        """Default LightGBM parameters."""
        base_params = {
            'n_estimators': 500,
            'max_depth': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
        }
        
        if self.task == 'classification':
            base_params['objective'] = 'binary'
            base_params['metric'] = 'auc'
        else:
            base_params['objective'] = 'regression'
            base_params['metric'] = 'rmse'
        
        return base_params
    
    def _default_catboost_params(self) -> Dict[str, Any]:
        """Default CatBoost parameters."""
        base_params = {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'random_strength': 1.0,
            'bagging_temperature': 1.0,
            'random_seed': self.random_state,
            'verbose': False,
        }
        
        if self.task == 'classification':
            base_params['loss_function'] = 'Logloss'
            base_params['eval_metric'] = 'AUC'
        else:
            base_params['loss_function'] = 'RMSE'
            base_params['eval_metric'] = 'RMSE'
        
        return base_params
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple] = None,
        early_stopping_rounds: int = 50,
        scale_features: bool = True,
    ) -> 'GradientBoostingModels':
        """
        Fit all gradient boosting models.
        
        Args:
            X: Training features
            y: Training target
            eval_set: Validation set (X_val, y_val) for early stopping
            early_stopping_rounds: Rounds for early stopping
            scale_features: Whether to scale features
            
        Returns:
            Self for method chaining
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Track scaling setting
        self.scale_features = scale_features
        
        # Scale features
        if scale_features:
            X = self.scaler.fit_transform(X)
            if eval_set:
                X_val = self.scaler.transform(eval_set[0])
                eval_set = (X_val, eval_set[1])
        
        # Fit XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost model...")
            self._fit_xgboost(X, y, eval_set, early_stopping_rounds)
        
        # Fit LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM model...")
            self._fit_lightgbm(X, y, eval_set, early_stopping_rounds)
        
        # Fit CatBoost
        if CATBOOST_AVAILABLE:
            logger.info("Training CatBoost model...")
            self._fit_catboost(X, y, eval_set, early_stopping_rounds)
        
        self.fitted = True
        return self
    
    def _fit_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple],
        early_stopping_rounds: int,
    ):
        """Fit XGBoost model."""
        # XGBoost 2.0+ requires early_stopping_rounds in constructor
        xgb_params = self.xgb_params.copy()
        if eval_set:
            xgb_params['early_stopping_rounds'] = early_stopping_rounds
        
        if self.task == 'classification':
            self.xgb_model = xgb.XGBClassifier(**xgb_params)
        else:
            self.xgb_model = xgb.XGBRegressor(**xgb_params)
        
        fit_params = {'verbose': False}
        if eval_set:
            fit_params['eval_set'] = [eval_set]
        
        self.xgb_model.fit(X, y, **fit_params)
    
    def _fit_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple],
        early_stopping_rounds: int,
    ):
        """Fit LightGBM model."""
        if self.task == 'classification':
            self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
        else:
            self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
        
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        
        if eval_set:
            self.lgb_model.fit(
                X, y,
                eval_set=[eval_set],
                callbacks=callbacks,
            )
        else:
            self.lgb_model.fit(X, y)
    
    def _fit_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple],
        early_stopping_rounds: int,
    ):
        """Fit CatBoost model."""
        if self.task == 'classification':
            self.catboost_model = CatBoostClassifier(**self.catboost_params)
        else:
            self.catboost_model = CatBoostRegressor(**self.catboost_params)
        
        if eval_set:
            self.catboost_model.fit(
                X, y,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            self.catboost_model.fit(X, y)
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        model: str = 'all',
        aggregate: str = 'mean',
    ) -> np.ndarray:
        """
        Make predictions using trained models.
        
        Args:
            X: Features for prediction
            model: Which model to use ('xgb', 'lgb', 'catboost', 'all')
            aggregate: How to combine predictions ('mean', 'vote', 'weighted')
            
        Returns:
            Predictions array
        """
        if not self.fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Only scale if scaling was used during fit
        if self.scale_features:
            X = self.scaler.transform(X)
        
        if model == 'all':
            predictions = []
            
            if self.xgb_model:
                if self.task == 'classification':
                    predictions.append(self.xgb_model.predict_proba(X)[:, 1])
                else:
                    predictions.append(self.xgb_model.predict(X))
            
            if self.lgb_model:
                if self.task == 'classification':
                    predictions.append(self.lgb_model.predict_proba(X)[:, 1])
                else:
                    predictions.append(self.lgb_model.predict(X))
            
            if self.catboost_model:
                if self.task == 'classification':
                    predictions.append(self.catboost_model.predict_proba(X)[:, 1])
                else:
                    predictions.append(self.catboost_model.predict(X))
            
            if not predictions:
                raise ValueError("No models available for prediction")
            
            if aggregate == 'mean':
                return np.mean(predictions, axis=0)
            elif aggregate == 'vote':
                binary_preds = [np.round(p) for p in predictions]
                return np.round(np.mean(binary_preds, axis=0))
            elif aggregate == 'weighted':
                # TODO: Implement weighted based on validation performance
                return np.mean(predictions, axis=0)
        
        elif model == 'xgb':
            if self.task == 'classification':
                return self.xgb_model.predict_proba(X)[:, 1]
            return self.xgb_model.predict(X)
        
        elif model == 'lgb':
            if self.task == 'classification':
                return self.lgb_model.predict_proba(X)[:, 1]
            return self.lgb_model.predict(X)
        
        elif model == 'catboost':
            if self.task == 'classification':
                return self.catboost_model.predict_proba(X)[:, 1]
            return self.catboost_model.predict(X)
        
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        model: str = 'all',
    ) -> np.ndarray:
        """
        Get probability predictions for classification.
        
        Args:
            X: Features for prediction
            model: Which model to use
            
        Returns:
            Probability array (n_samples, 2)
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        probs = self.predict(X, model=model)
        return np.column_stack([1 - probs, probs])
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model: str = 'all',
    ) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels/values
            model: Which model to evaluate
            
        Returns:
            ModelMetrics with evaluation scores
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        predictions = self.predict(X, model=model)
        
        metrics = ModelMetrics()
        
        if self.task == 'classification':
            binary_preds = (predictions > 0.5).astype(int)
            
            metrics.accuracy = accuracy_score(y, binary_preds)
            metrics.precision = precision_score(y, binary_preds, zero_division=0)
            metrics.recall = recall_score(y, binary_preds, zero_division=0)
            metrics.f1 = f1_score(y, binary_preds, zero_division=0)
            
            try:
                metrics.roc_auc = roc_auc_score(y, predictions)
            except ValueError:
                metrics.roc_auc = 0.0
        else:
            metrics.mse = mean_squared_error(y, predictions)
            metrics.mae = mean_absolute_error(y, predictions)
        
        return metrics
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        gap: int = 20,
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Features
            y: Target
            n_splits: Number of CV splits
            gap: Gap between train and test to prevent leakage
            
        Returns:
            Dictionary of metrics per fold
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        
        cv_results = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'roc_auc': [], 'mse': [], 'mae': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Cross-validation fold {fold + 1}/{n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Reset and fit
            self.fitted = False
            self.xgb_model = None
            self.lgb_model = None
            self.catboost_model = None
            
            self.fit(X_train, y_train)
            metrics = self.evaluate(X_test, y_test)
            
            cv_results['accuracy'].append(metrics.accuracy)
            cv_results['precision'].append(metrics.precision)
            cv_results['recall'].append(metrics.recall)
            cv_results['f1'].append(metrics.f1)
            cv_results['roc_auc'].append(metrics.roc_auc)
            cv_results['mse'].append(metrics.mse)
            cv_results['mae'].append(metrics.mae)
        
        # Log summary
        for metric, values in cv_results.items():
            if any(v > 0 for v in values):
                logger.info(f"CV {metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
        
        return cv_results
    
    def get_feature_importance(
        self,
        model: str = 'all',
        importance_type: str = 'gain',
    ) -> pd.DataFrame:
        """
        Get feature importance from models.
        
        Args:
            model: Which model's importance to get
            importance_type: Type of importance ('gain', 'split', 'weight')
            
        Returns:
            DataFrame with feature importance
        """
        importance_dict = {}
        
        if model in ['all', 'xgb'] and self.xgb_model:
            if importance_type == 'gain':
                imp = self.xgb_model.feature_importances_
            else:
                imp = self.xgb_model.feature_importances_
            importance_dict['xgb'] = imp
        
        if model in ['all', 'lgb'] and self.lgb_model:
            if importance_type == 'gain':
                imp = self.lgb_model.booster_.feature_importance('gain')
            else:
                imp = self.lgb_model.booster_.feature_importance('split')
            importance_dict['lgb'] = imp
        
        if model in ['all', 'catboost'] and self.catboost_model:
            imp = self.catboost_model.get_feature_importance()
            importance_dict['catboost'] = imp
        
        if not importance_dict:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(importance_dict)
        
        if self.feature_names:
            df['feature'] = self.feature_names
        else:
            df['feature'] = [f'feature_{i}' for i in range(len(df))]
        
        # Normalize and average
        for col in ['xgb', 'lgb', 'catboost']:
            if col in df.columns:
                df[col] = df[col] / df[col].sum()
        
        numeric_cols = [c for c in ['xgb', 'lgb', 'catboost'] if c in df.columns]
        df['mean_importance'] = df[numeric_cols].mean(axis=1)
        
        return df.sort_values('mean_importance', ascending=False)
    
    def save(self, path: str):
        """Save models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'task': self.task,
            'xgb_params': self.xgb_params,
            'lgb_params': self.lgb_params,
            'catboost_params': self.catboost_params,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'fitted': self.fitted,
        }
        
        with open(path / 'state.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        if self.xgb_model:
            self.xgb_model.save_model(str(path / 'xgb_model.json'))
        
        if self.lgb_model:
            self.lgb_model.booster_.save_model(str(path / 'lgb_model.txt'))
        
        if self.catboost_model:
            self.catboost_model.save_model(str(path / 'catboost_model.cbm'))
        
        logger.info(f"Models saved to {path}")
    
    def load(self, path: str):
        """Load models from disk."""
        path = Path(path)
        
        with open(path / 'state.pkl', 'rb') as f:
            state = pickle.load(f)
        
        self.task = state['task']
        self.xgb_params = state['xgb_params']
        self.lgb_params = state['lgb_params']
        self.catboost_params = state['catboost_params']
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.fitted = state['fitted']
        
        xgb_path = path / 'xgb_model.json'
        if xgb_path.exists() and XGBOOST_AVAILABLE:
            if self.task == 'classification':
                self.xgb_model = xgb.XGBClassifier()
            else:
                self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(str(xgb_path))
        
        lgb_path = path / 'lgb_model.txt'
        if lgb_path.exists() and LIGHTGBM_AVAILABLE:
            if self.task == 'classification':
                self.lgb_model = lgb.LGBMClassifier()
            else:
                self.lgb_model = lgb.LGBMRegressor()
            self.lgb_model.booster_ = lgb.Booster(model_file=str(lgb_path))
        
        catboost_path = path / 'catboost_model.cbm'
        if catboost_path.exists() and CATBOOST_AVAILABLE:
            if self.task == 'classification':
                self.catboost_model = CatBoostClassifier()
            else:
                self.catboost_model = CatBoostRegressor()
            self.catboost_model.load_model(str(catboost_path))
        
        logger.info(f"Models loaded from {path}")
        return self
