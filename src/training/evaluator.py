"""Model evaluation utilities (Phase 2 - 2.5).

Provides comprehensive evaluation metrics and visualizations
for view prediction models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from scipy.stats import spearmanr

from ..utils import setup_logger


class ModelEvaluator:
    """Evaluate and visualize model performance.
    
    Computes standard regression metrics plus domain-specific metrics
    like top-K accuracy (are high predictions actually high?).
    """

    def __init__(self, save_dir: Optional[Path | str] = None) -> None:
        """Initialize evaluator.
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        self.logger = setup_logger(__name__)
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        split_name: str = "test",
        generate_plots: bool = True,
    ) -> Dict[str, float]:
        """Evaluate predictions against ground truth.
        
        Args:
            y_true: True targets (N,)
            y_pred: Predicted targets (N,)
            split_name: Name of the split being evaluated (for logging/plots)
            generate_plots: Whether to generate visualization plots
            
        Returns:
            Dictionary of metric name to value
        """
        self.logger.info(f"Evaluating {split_name} split with {len(y_true)} samples")
        
        metrics = self._compute_metrics(y_true, y_pred)
        
        # Log metrics
        for name, value in metrics.items():
            self.logger.info(f"{split_name} {name}: {value:.4f}")
        
        # Generate visualizations
        if generate_plots:
            self._plot_predictions_vs_actual(
                y_true, y_pred, split_name=split_name, metrics=metrics
            )
            self._plot_residuals(y_true, y_pred, split_name=split_name)
        
        return metrics

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all evaluation metrics.
        
        Args:
            y_true: True targets
            y_pred: Predicted targets
            
        Returns:
            Dictionary of metric name to value
        """
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "spearman": spearmanr(y_true, y_pred)[0],
        }
        
        # Top-K accuracy: do we correctly identify high performers?
        for k in [5, 10]:
            if len(y_true) >= k:
                metrics[f"top{k}_accuracy"] = self._compute_top_k_accuracy(
                    y_true, y_pred, k=k
                )
        
        # NDCG@5
        if len(y_true) > 1:
            # ndcg_score expects shape (n_samples, n_labels)
            # We treat the whole set as one sample with many labels (items)
            metrics["ndcg@5"] = ndcg_score([y_true], [y_pred], k=5)
        
        return metrics

    def _compute_top_k_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 5
    ) -> float:
        """Compute top-K accuracy.
        
        Measures how well the model identifies the top K highest-view segments.
        Returns the overlap (Jaccard similarity) between predicted and actual top-K.
        
        Args:
            y_true: True targets
            y_pred: Predicted targets
            k: Number of top items to consider
            
        Returns:
            Overlap score between 0 and 1
        """
        if len(y_true) < k:
            return 0.0
        
        # Get indices of top K actual and predicted
        top_k_true = set(np.argsort(y_true)[-k:])
        top_k_pred = set(np.argsort(y_pred)[-k:])
        
        # Compute overlap (Jaccard similarity)
        intersection = len(top_k_true & top_k_pred)
        union = len(top_k_true | top_k_pred)
        
        return intersection / union if union > 0 else 0.0

    def _plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        split_name: str = "test",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Create scatter plot of predictions vs actual values.
        
        Args:
            y_true: True targets
            y_pred: Predicted targets
            split_name: Name of split for title
            metrics: Optional metrics to display on plot
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        plt.xlabel('Actual Views', fontsize=12)
        plt.ylabel('Predicted Views', fontsize=12)
        plt.title(f'Predictions vs Actual ({split_name.capitalize()})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics text box
        if metrics:
            textstr = '\n'.join([
                f"R² = {metrics.get('r2', 0):.3f}",
                f"RMSE = {metrics.get('rmse', 0):.3f}",
                f"MAE = {metrics.get('mae', 0):.3f}",
                f"Spearman = {metrics.get('spearman', 0):.3f}"
            ])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(
                0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=props
            )
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / f"predictions_vs_actual_{split_name}.png"
            plt.savefig(save_path, dpi=150)
            self.logger.info(f"Saved plot to {save_path}")
        
        plt.close()

    def _plot_residuals(
        self, y_true: np.ndarray, y_pred: np.ndarray, *, split_name: str = "test"
    ) -> None:
        """Create residual plot to diagnose model bias.
        
        Args:
            y_true: True targets
            y_pred: Predicted targets
            split_name: Name of split for title
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs predicted
        ax1.scatter(y_pred, residuals, alpha=0.5, s=50)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Views', fontsize=12)
        ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        ax1.set_title(f'Residual Plot ({split_name.capitalize()})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Residual histogram
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Residual Distribution ({split_name.capitalize()})', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / f"residuals_{split_name}.png"
            plt.savefig(save_path, dpi=150)
            self.logger.info(f"Saved plot to {save_path}")
        
        plt.close()

    def compare_models(
        self,
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        *,
        split_name: str = "test",
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on the same test set.
        
        Args:
            y_true: True targets
            predictions_dict: Dictionary mapping model name to predictions
            split_name: Name of split for logging
            
        Returns:
            Dictionary mapping model name to its metrics
        """
        self.logger.info(f"Comparing {len(predictions_dict)} models on {split_name} split")
        
        results = {}
        for model_name, y_pred in predictions_dict.items():
            metrics = self._compute_metrics(y_true, y_pred)
            results[model_name] = metrics
            self.logger.info(f"{model_name}: R²={metrics['r2']:.4f}, Spearman={metrics['spearman']:.4f}")
        
        # Create comparison plot
        self._plot_model_comparison(results, split_name=split_name)
        
        return results

    def _plot_model_comparison(
        self, results: Dict[str, Dict[str, float]], *, split_name: str = "test"
    ) -> None:
        """Create bar chart comparing models across metrics.
        
        Args:
            results: Dictionary mapping model name to metrics
            split_name: Name of split for title
        """
        if not results:
            return
        
        metrics_to_plot = ["r2", "spearman", "mae"]
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, metrics_to_plot):
            values = [results[m].get(metric_name, 0) for m in model_names]
            ax.bar(model_names, values, alpha=0.7, edgecolor='black')
            ax.set_ylabel(metric_name.upper(), fontsize=12)
            ax.set_title(f'{metric_name.upper()} Comparison', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'Model Comparison ({split_name.capitalize()})', fontsize=16)
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / f"model_comparison_{split_name}.png"
            plt.savefig(save_path, dpi=150)
            self.logger.info(f"Saved comparison plot to {save_path}")
        
        plt.close()
