"""Model implementations for view prediction (Phase 2 - 2.3).

Provides a base class and concrete implementations for predicting
YouTube Shorts views from transcript embeddings.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb

from ..utils import setup_logger


class ViewPredictor(ABC):
    """Base class for view prediction models.
    
    All models should inherit from this and implement fit/predict.
    """

    def __init__(self, **kwargs) -> None:
        self.logger = setup_logger(self.__class__.__name__)
        self.model: Optional[Any] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the model on embeddings X and target y.
        
        Args:
            X: Training embeddings (N, D)
            y: Training targets (N,)
            X_val: Optional validation embeddings for early stopping
            y_val: Optional validation targets for early stopping
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict views for embeddings X.
        
        Args:
            X: Embeddings (N, D)
            
        Returns:
            Predictions (N,)
        """
        pass

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model on given data.
        
        Args:
            X: Embeddings (N, D)
            y_true: True targets (N,)
            
        Returns:
            Dictionary of metric name to value
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X)
        
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "spearman": spearmanr(y_true, y_pred)[0],
        }
        
        return metrics

    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        raise NotImplementedError("Subclass must implement save()")

    def load(self, path: Path | str) -> None:
        """Load model from disk."""
        raise NotImplementedError("Subclass must implement load()")


class LinearRegressor(ViewPredictor):
    """Ridge or Lasso regression baseline for view prediction.
    
    Simple linear model serving as a baseline. Good for establishing
    a lower bound on performance and understanding feature importance.
    """

    def __init__(self, alpha: float = 1.0, model_type: str = "ridge", **kwargs) -> None:
        """Initialize linear regressor.
        
        Args:
            alpha: Regularization strength
            model_type: 'ridge' or 'lasso'
            **kwargs: Additional sklearn parameters
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.model_type = model_type.lower()
        
        if self.model_type == "ridge":
            self.model = Ridge(alpha=alpha, **kwargs)
        elif self.model_type == "lasso":
            self.model = Lasso(alpha=alpha, **kwargs)
        else:
            raise ValueError(f"model_type must be 'ridge' or 'lasso', got {model_type}")
        
        self.logger.info(f"Initialized {self.model_type.capitalize()} regressor with alpha={alpha}")

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit linear model."""
        self.logger.info(f"Fitting {self.model_type} on {X.shape[0]} samples")
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Log training performance
        train_metrics = self.evaluate(X, y)
        self.logger.info(f"Training metrics: {train_metrics}")
        
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.logger.info(f"Validation metrics: {val_metrics}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with linear model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)


class GradientBoostingRegressor(ViewPredictor):
    """XGBoost or LightGBM regressor for view prediction.
    
    Tree-based ensemble model with support for early stopping and
    feature importance analysis.
    """

    def __init__(
        self,
        framework: str = "xgboost",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 10,
        **kwargs
    ) -> None:
        """Initialize gradient boosting regressor.
        
        Args:
            framework: 'xgboost' or 'lightgbm'
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (eta)
            max_depth: Maximum tree depth
            subsample: Row sampling ratio
            colsample_bytree: Column sampling ratio
            early_stopping_rounds: Stop if no improvement for N rounds
            **kwargs: Additional framework-specific parameters
        """
        super().__init__(**kwargs)
        self.framework = framework.lower()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.extra_params = kwargs
        
        if self.framework not in ["xgboost", "lightgbm"]:
            raise ValueError(f"framework must be 'xgboost' or 'lightgbm', got {framework}")
        
        self.logger.info(f"Initialized {framework.upper()} with n_estimators={n_estimators}, lr={learning_rate}")

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit gradient boosting model."""
        self.logger.info(f"Fitting {self.framework.upper()} on {X.shape[0]} samples")
        
        if self.framework == "xgboost":
            self._fit_xgboost(X, y, X_val, y_val)
        else:
            self._fit_lightgbm(X, y, X_val, y_val)
        
        self.is_fitted = True
        
        # Log training performance
        train_metrics = self.evaluate(X, y)
        self.logger.info(f"Training metrics: {train_metrics}")
        
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.logger.info(f"Validation metrics: {val_metrics}")

    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> None:
        """Fit XGBoost model using sklearn API (more stable than native API)."""
        # Use sklearn API to avoid segfault issues on macOS
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="reg:squarederror",
            tree_method="exact",  # Safer for small datasets
            n_jobs=1,  # Single-threaded to prevent OpenMP issues
            random_state=42,
            **self.extra_params,
        )
        
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            self.model.fit(X, y)

    def _fit_lightgbm(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> None:
        """Fit LightGBM model."""
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "verbosity": -1,
            **self.extra_params,
        }
        
        dtrain = lgb.Dataset(X, label=y)
        valid_sets = [dtrain]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("val")
        
        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds)] if X_val is not None else None,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with gradient boosting model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary mapping feature index to importance score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if self.framework == "xgboost":
            # sklearn API uses feature_importances_ attribute
            importance = self.model.feature_importances_
            return {i: float(v) for i, v in enumerate(importance)}
        else:
            importance = self.model.feature_importance(importance_type="gain")
            return {i: float(v) for i, v in enumerate(importance)}
