"""Model training orchestration with experiment tracking (Phase 2 - 2.4).

Provides a unified interface for training view prediction models
with comprehensive MLflow logging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sklearn.model_selection import KFold
from .models import ViewPredictor
from .evaluator import ModelEvaluator
from .run_logger import RunLogger
from ..utils import setup_logger


class ModelTrainer:
    """Orchestrate model training with experiment tracking.
    
    Uses simple RunLogger for:
    - Hyperparameter logging
    - Metric tracking
    - Model artifact storage
    - Clean, human-readable run directories
    """

    def __init__(self, runs_dir: Path | str = "runs") -> None:
        """Initialize trainer with run logging.
        
        Args:
            runs_dir: Base directory for all training runs
        """
        self.logger = setup_logger(__name__)
        self.run_logger = RunLogger(runs_dir)

    def train(
        self,
        model: ViewPredictor,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        *,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        save_model: bool = True,
        train_video_ids: Optional[list] = None,
        val_video_ids: Optional[list] = None,
        test_video_ids: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Train a model with run tracking.
        
        Args:
            model: Model instance to train
            X_train: Training embeddings (N, D)
            y_train: Training targets (N,)
            X_val: Validation embeddings
            y_val: Validation targets
            X_test: Optional test embeddings
            y_test: Optional test targets
            run_name: Optional name for this training run
            config: Optional configuration dict to log
            save_model: Whether to save model artifact
            
        Returns:
            Dictionary with training results and metrics
        """
        self.logger.info(f"Starting training run: {run_name or 'unnamed'}")
        self.logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Start run
        run_dir = self.run_logger.start_run(run_name)
        
        try:
            # Log dataset info
            self.run_logger.log_params({
                "train_size": X_train.shape[0],
                "val_size": X_val.shape[0],
                "embedding_dim": X_train.shape[1],
            })
            if X_test is not None:
                self.run_logger.log_params({"test_size": X_test.shape[0]})
            
            # Log model hyperparameters
            self._log_model_params(model)
            
            # Log configuration
            if config:
                self.run_logger.log_config(config)
            
            # Save video IDs for data reproducibility
            if train_video_ids is not None:
                import json
                data_splits = {
                    "train_video_ids": train_video_ids,
                    "val_video_ids": val_video_ids or [],
                    "test_video_ids": test_video_ids or [],
                }
                splits_path = run_dir / "data_splits.json"
                with open(splits_path, "w") as f:
                    json.dump(data_splits, f, indent=2)
                self.logger.info(f"Saved data splits to {splits_path}")
            
            # Train model
            self.logger.info("Fitting model...")
            model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate on all splits
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            evaluator = ModelEvaluator(save_dir=plots_dir)
            
            # Training metrics
            train_metrics = model.evaluate(X_train, y_train)
            for name, value in train_metrics.items():
                self.run_logger.log_metric(f"train_{name}", value)
            
            # Validation metrics
            val_metrics = model.evaluate(X_val, y_val)
            for name, value in val_metrics.items():
                self.run_logger.log_metric(f"val_{name}", value)
            
            # Test metrics (if provided)
            test_metrics = None
            if X_test is not None and y_test is not None:
                test_metrics = evaluator.evaluate(
                    y_test,
                    model.predict(X_test),
                    split_name="test",
                    generate_plots=True,
                )
                for name, value in test_metrics.items():
                    self.run_logger.log_metric(f"test_{name}", value)
            
            # Save model
            if save_model:
                self.run_logger.save_model(model)
            
            # Finalize run
            self.run_logger.end_run()
            
            results = {
                "run_dir": str(run_dir),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "model": model,
            }
            
            self.logger.info(f"Training complete: {run_dir}")
            self.logger.info(f"Val R²: {val_metrics['r2']:.4f}, Spearman: {val_metrics['spearman']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.run_logger.end_run()
            raise

    def train_with_cv(
        self,
        model_factory: callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        *,
        n_folds: int = 5,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        save_model: bool = True,
        train_video_ids: Optional[list] = None,
        test_video_ids: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Train using K-Fold Cross-Validation followed by Final Fit.
        
        Args:
            model_factory: Function that returns a fresh model instance
            X_train: Training embeddings (N, D)
            y_train: Training targets (N,)
            X_test: Held-out test embeddings
            y_test: Held-out test targets
            n_folds: Number of CV folds
            run_name: Run name
            config: Config dict
            save_model: Whether to save final model
            
        Returns:
            Results dictionary
        """
        self.logger.info(f"Starting CV training run: {run_name or 'unnamed'}")
        self.logger.info(f"Train: {X_train.shape}, Test: {X_test.shape if X_test is not None else 'None'}")
        
        run_dir = self.run_logger.start_run(run_name)
        
        try:
            # Log params
            self.run_logger.log_params({
                "train_size": X_train.shape[0],
                "embedding_dim": X_train.shape[1],
                "n_folds": n_folds,
                "strategy": "cross_validation_final_fit"
            })
            if X_test is not None:
                self.run_logger.log_params({"test_size": X_test.shape[0]})
            
            if config:
                self.run_logger.log_config(config)
                
            # 1. Cross-Validation Loop
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_metrics = []
            
            self.logger.info(f"Starting {n_folds}-Fold Cross-Validation...")
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                self.logger.info(f"Fold {fold+1}/{n_folds}")
                
                # Split fold data
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Create fresh model
                fold_model = model_factory()
                
                # Train
                fold_model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
                
                # Evaluate
                metrics = fold_model.evaluate(X_fold_val, y_fold_val)
                cv_metrics.append(metrics)
                
                # Log fold metrics
                for name, value in metrics.items():
                    self.run_logger.log_metric(f"fold_{fold+1}_{name}", value)
            
            # Aggregate CV metrics
            avg_metrics = {}
            for key in cv_metrics[0].keys():
                values = [m[key] for m in cv_metrics]
                avg = np.mean(values)
                std = np.std(values)
                avg_metrics[key] = avg
                self.run_logger.log_metric(f"cv_mean_{key}", avg)
                self.run_logger.log_metric(f"cv_std_{key}", std)
                self.logger.info(f"CV Mean {key}: {avg:.4f} (+/- {std:.4f})")
            
            # 2. Final Fit
            self.logger.info("Performing Final Fit on all training data...")
            final_model = model_factory()
            
            # We don't have a validation set for early stopping here, 
            # so we rely on the hyperparameters found/set previously.
            # Some models (XGBoost) might need a validation set for early stopping.
            # We can either:
            # a) Use no early stopping
            # b) Use a small slice of train as val just for early stopping (but that wastes data)
            # c) Use the average best_iteration from CV (advanced)
            # For now, we'll pass X_test as validation if available, OR just fit without early stopping
            # BUT: We strictly should NOT use X_test for early stopping to avoid leakage.
            # So we fit without early stopping or use the full set.
            final_model.fit(X_train, y_train)
            
            # 3. Final Evaluation on Test Set
            test_metrics = None
            if X_test is not None and y_test is not None:
                self.logger.info("Evaluating Final Model on Test Set...")
                plots_dir = run_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                evaluator = ModelEvaluator(save_dir=plots_dir)
                
                test_metrics = evaluator.evaluate(
                    y_test,
                    final_model.predict(X_test),
                    split_name="test",
                    generate_plots=True,
                )
                for name, value in test_metrics.items():
                    self.run_logger.log_metric(f"test_{name}", value)
            
            # Save Final Model
            if save_model:
                self.run_logger.save_model(final_model)
            
            self.run_logger.end_run()
            
            results = {
                "run_dir": str(run_dir),
                "cv_metrics": avg_metrics,
                "test_metrics": test_metrics,
                "model": final_model
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"CV Training failed: {e}")
            self.run_logger.end_run()
            raise

    def _log_model_params(self, model: ViewPredictor) -> None:
        """Log model-specific hyperparameters.
        
        Args:
            model: Model instance
        """
        # Log class name
        self.run_logger.log_params({"model_class": model.__class__.__name__})
        
        # Try to extract hyperparameters from model attributes
        if hasattr(model, "__dict__"):
            params_to_log = {}
            for key, value in model.__dict__.items():
                # Skip logger, model object, and internal state
                if key in ["logger", "model", "is_fitted", "extra_params"]:
                    continue
                # Only log simple types
                if isinstance(value, (int, float, str, bool)):
                    params_to_log[key] = value
            
            if params_to_log:
                self.run_logger.log_params(params_to_log)
