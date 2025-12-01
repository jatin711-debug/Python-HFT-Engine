"""
Machine Learning Model Parameters for the Trading Engine.

This module contains hyperparameters for all ML models used in the system,
including XGBoost, LightGBM, Random Forest, and deep learning models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class XGBoostParams:
    """Hyperparameters for XGBoost model."""
    
    # Core parameters
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 3
    
    # Regularization
    gamma: float = 0.1                      # Min loss reduction for split
    reg_alpha: float = 0.1                  # L1 regularization
    reg_lambda: float = 1.0                 # L2 regularization
    
    # Sampling
    subsample: float = 0.8                  # Row sampling
    colsample_bytree: float = 0.8           # Column sampling per tree
    colsample_bylevel: float = 0.8          # Column sampling per level
    
    # Other
    objective: str = "binary:logistic"      # For classification
    eval_metric: str = "auc"
    tree_method: str = "hist"               # Fast histogram-based
    n_jobs: int = -1
    random_state: int = 42
    early_stopping_rounds: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model initialization."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bylevel': self.colsample_bylevel,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'tree_method': self.tree_method,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
        }


@dataclass
class LightGBMParams:
    """Hyperparameters for LightGBM model."""
    
    # Core parameters
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    num_leaves: int = 31                    # Main parameter for complexity
    min_child_samples: int = 20
    
    # Regularization
    reg_alpha: float = 0.1                  # L1 regularization
    reg_lambda: float = 0.1                 # L2 regularization
    min_gain_to_split: float = 0.01
    
    # Sampling
    subsample: float = 0.8                  # bagging_fraction
    subsample_freq: int = 1                 # bagging every iteration
    colsample_bytree: float = 0.8           # feature_fraction
    
    # Other
    objective: str = "binary"
    metric: str = "auc"
    boosting_type: str = "gbdt"             # Options: gbdt, dart, goss
    n_jobs: int = -1
    random_state: int = 42
    verbose: int = -1
    early_stopping_rounds: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model initialization."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'subsample': self.subsample,
            'subsample_freq': self.subsample_freq,
            'colsample_bytree': self.colsample_bytree,
            'objective': self.objective,
            'metric': self.metric,
            'boosting_type': self.boosting_type,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }


@dataclass
class RandomForestParams:
    """Hyperparameters for Random Forest model."""
    
    # Core parameters
    n_estimators: int = 300
    max_depth: int = 10
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    
    # Feature selection
    max_features: str = "sqrt"              # Options: sqrt, log2, None
    
    # Sampling
    bootstrap: bool = True
    max_samples: float = 0.8                # Subsample ratio
    
    # Other
    criterion: str = "gini"                 # Options: gini, entropy
    class_weight: str = "balanced"          # Handle imbalanced classes
    n_jobs: int = -1
    random_state: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model initialization."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'max_samples': self.max_samples,
            'criterion': self.criterion,
            'class_weight': self.class_weight,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
        }


@dataclass
class LSTMParams:
    """Hyperparameters for LSTM deep learning model."""
    
    # Architecture
    sequence_length: int = 20               # Look back window
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5              # L2 regularization
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Other
    random_state: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
        }


@dataclass
class TransformerParams:
    """Hyperparameters for Transformer model."""
    
    # Architecture
    sequence_length: int = 30
    d_model: int = 64                       # Embedding dimension
    nhead: int = 4                          # Number of attention heads
    num_encoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.2
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 1e-5
    
    # Scheduler
    warmup_steps: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Other
    random_state: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
        }


@dataclass
class EnsembleParams:
    """Parameters for ensemble model combination."""
    
    # Model weights (should sum to 1.0)
    xgboost_weight: float = 0.35
    lightgbm_weight: float = 0.30
    random_forest_weight: float = 0.20
    lstm_weight: float = 0.15
    
    # Meta-learner
    use_meta_learner: bool = True
    meta_learner_type: str = "logistic"     # Options: logistic, xgboost
    
    # Stacking settings
    use_stacking: bool = False
    stacking_cv_folds: int = 5
    
    # Voting method
    voting_method: str = "soft"             # Options: soft, hard
    
    # Confidence calibration
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"    # Options: isotonic, sigmoid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'xgboost_weight': self.xgboost_weight,
            'lightgbm_weight': self.lightgbm_weight,
            'random_forest_weight': self.random_forest_weight,
            'lstm_weight': self.lstm_weight,
            'use_meta_learner': self.use_meta_learner,
            'voting_method': self.voting_method,
        }


@dataclass
class HyperoptParams:
    """Parameters for hyperparameter optimization."""
    
    # Search method
    search_method: str = "bayesian"         # Options: grid, random, bayesian
    n_trials: int = 100
    
    # Cross-validation
    cv_folds: int = 5
    cv_method: str = "time_series"          # Options: kfold, time_series
    
    # Optimization target
    target_metric: str = "sharpe_ratio"     # Options: accuracy, auc, sharpe_ratio
    
    # Time budget
    timeout_seconds: int = 3600             # 1 hour max
    
    # Early stopping
    early_stopping_rounds: int = 20
    
    # XGBoost search space
    xgb_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500, 700],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0.1, 0.5, 1, 2],
    })
    
    # LightGBM search space
    lgb_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': [4, 6, 8, 10, 12],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300, 500],
        'num_leaves': [15, 31, 63, 127],
        'min_child_samples': [10, 20, 30, 50],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5, 1],
    })


@dataclass
class ModelParams:
    """Main model parameters container."""
    
    xgboost: XGBoostParams = field(default_factory=XGBoostParams)
    lightgbm: LightGBMParams = field(default_factory=LightGBMParams)
    random_forest: RandomForestParams = field(default_factory=RandomForestParams)
    lstm: LSTMParams = field(default_factory=LSTMParams)
    transformer: TransformerParams = field(default_factory=TransformerParams)
    ensemble: EnsembleParams = field(default_factory=EnsembleParams)
    hyperopt: HyperoptParams = field(default_factory=HyperoptParams)
    
    # Target variable settings
    target_type: str = "classification"     # Options: classification, regression
    classification_threshold: float = 0.0   # Return > 0 = positive class
    
    # Feature preprocessing
    scale_features: bool = True
    scaler_type: str = "standard"           # Options: standard, minmax, robust
    
    # Feature selection
    use_feature_selection: bool = True
    feature_selection_method: str = "importance"  # Options: importance, rfe, boruta
    max_features_ratio: float = 0.8         # Keep top 80% of features
    
    # Class imbalance handling
    handle_imbalance: bool = True
    imbalance_method: str = "smote"         # Options: smote, undersample, class_weight


# Global model parameters instance
model_params = ModelParams()
