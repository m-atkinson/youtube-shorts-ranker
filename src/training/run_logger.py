"""Simple run logging for model training (replaces MLflow complexity).

Creates clean, human-readable run directories:
runs/
  <run-name>/
    config.yaml
    metrics.json
    plots/
    model.pkl
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..utils import setup_logger


class RunLogger:
    """Simple experiment tracking with clean directory structure."""

    def __init__(self, runs_dir: Path | str = "runs"):
        """Initialize run logger.
        
        Args:
            runs_dir: Base directory for all runs
        """
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(__name__)
        
        self.current_run_dir: Optional[Path] = None
        self.run_name: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.params: Dict[str, Any] = {}

    def start_run(self, run_name: Optional[str] = None) -> Path:
        """Start a new run with unique directory.
        
        Args:
            run_name: Name for this run (will add timestamp if not unique)
            
        Returns:
            Path to run directory
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure unique directory name
        base_name = run_name
        run_dir = self.runs_dir / base_name
        counter = 1
        while run_dir.exists():
            run_dir = self.runs_dir / f"{base_name}_{counter}"
            counter += 1
        
        run_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_dir = run_dir
        self.run_name = run_dir.name
        self.metrics = {}
        self.params = {}
        
        self.logger.info(f"Started run: {self.run_name}")
        return run_dir

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters and configuration.
        
        Args:
            params: Dictionary of parameters to log
        """
        self.params.update(params)

    def log_metric(self, name: str, value: float) -> None:
        """Log a single metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        self.metrics.update(metrics)

    def log_config(self, config: Dict[str, Any], filename: str = "config.yaml") -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            filename: Name of config file
        """
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        config_path = self.current_run_dir / filename
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Saved config to {config_path}")

    def log_artifact(self, file_path: Path | str, subdir: Optional[str] = None) -> None:
        """Copy an artifact (plot, file, etc.) to run directory.
        
        Args:
            file_path: Path to file to copy
            subdir: Optional subdirectory within run dir
        """
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.warning(f"Artifact not found: {file_path}")
            return
        
        # Determine destination
        if subdir:
            dest_dir = self.current_run_dir / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)
        else:
            dest_dir = self.current_run_dir
        
        dest_path = dest_dir / file_path.name
        
        # Copy file
        import shutil
        shutil.copy2(file_path, dest_path)
        self.logger.info(f"Logged artifact: {dest_path}")

    def save_model(self, model: Any, filename: str = "model.pkl") -> None:
        """Save model to run directory.
        
        Args:
            model: Model object to save
            filename: Name of model file
        """
        if self.current_run_dir is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        model_path = self.current_run_dir / filename
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        self.logger.info(f"Saved model to {model_path}")

    def end_run(self) -> None:
        """Finalize run by saving metrics and params."""
        if self.current_run_dir is None:
            self.logger.warning("No active run to end.")
            return
        
        # Save all metrics
        metrics_path = self.current_run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save all params
        params_path = self.current_run_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(self.params, f, indent=2)
        
        # Create summary
        summary = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "params": self.params,
        }
        
        summary_path = self.current_run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Run complete: {self.current_run_dir}")
        self.current_run_dir = None
        self.run_name = None

    def get_run_dir(self) -> Optional[Path]:
        """Get current run directory."""
        return self.current_run_dir
