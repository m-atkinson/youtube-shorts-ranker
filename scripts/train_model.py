"""Training script for YouTube Shorts view prediction models (Phase 2 - 2.6).

Orchestrates end-to-end training:
- Load dataset and generate embeddings
- Initialize model from config
- Train with MLflow tracking
- Evaluate and save results

Usage:
    python scripts/train_model.py --config config/training.yaml
    python scripts/train_model.py --model xgboost --embedding all-MiniLM-L6-v2
"""
import os
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import ShortsDataset, ModelTrainer
from src.training.models import LinearRegressor, GradientBoostingRegressor
from src.utils import setup_logger

# Disable parallelism to prevent segfaults on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logger = setup_logger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def create_model_from_config(config: dict):
    """Create model instance from configuration.
    
    Args:
        config: Configuration dict with 'model' section
        
    Returns:
        Model instance
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "xgboost").lower()
    hyperparams = model_cfg.get("hyperparameters", {})
    
    if model_type == "linear" or model_type == "ridge":
        model = LinearRegressor(
            alpha=hyperparams.get("alpha", 1.0),
            model_type="ridge",
        )
    elif model_type == "lasso":
        model = LinearRegressor(
            alpha=hyperparams.get("alpha", 1.0),
            model_type="lasso",
        )
    elif model_type == "xgboost":
        model = GradientBoostingRegressor(
            framework="xgboost",
            n_estimators=hyperparams.get("n_estimators", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            max_depth=hyperparams.get("max_depth", 6),
            subsample=hyperparams.get("subsample", 0.8),
            colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
            early_stopping_rounds=config.get("training", {}).get("early_stopping_rounds", 10),
        )
    elif model_type == "lightgbm":
        model = GradientBoostingRegressor(
            framework="lightgbm",
            n_estimators=hyperparams.get("n_estimators", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            max_depth=hyperparams.get("max_depth", 6),
            subsample=hyperparams.get("subsample", 0.8),
            colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
            early_stopping_rounds=config.get("training", {}).get("early_stopping_rounds", 10),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Created {model_type} model")
    return model


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to training configuration YAML file",
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("data/database/shorts_data.db"),
    help="Path to SQLite database",
)
@click.option(
    "--embedding-model",
    type=str,
    help="Embedding model name (overrides config)",
)
@click.option(
    "--model-type",
    type=click.Choice(["linear", "ridge", "lasso", "xgboost", "lightgbm"], case_sensitive=False),
    help="Model type (overrides config)",
)
@click.option(
    "--run-name",
    type=str,
    help="MLflow run name",
)
def main(
    config: Path,
    db_path: Path,
    embedding_model: str,
    model_type: str,
    run_name: str,
):
    """Train a YouTube Shorts view prediction model."""
    logger.info("Starting model training pipeline")
    
    # Load configuration
    if config:
        cfg = load_config(config)
        logger.info(f"Loaded config from {config}")
    else:
        # Default configuration
        cfg = {
            "data": {
                "train_split": 0.85,
                "val_split": 0.0,
                "test_split": 0.15,
            },
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "cache_dir": "data/processed/embeddings",
            },
            "model": {
                "type": "xgboost",
                "hyperparameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                },
            },
            "training": {
                "early_stopping_rounds": 10,
            },
        }
        logger.info("Using default configuration")
    
    # Override config with CLI args if provided
    if embedding_model:
        cfg["embedding"]["model_name"] = embedding_model
    if model_type:
        cfg["model"]["type"] = model_type
    
    # Prepare dataset
    logger.info("Loading and preparing dataset...")
    data_cfg = cfg.get("data", {})
    emb_cfg = cfg.get("embedding", {})
    
    dataset = ShortsDataset(
        db_path=db_path,
        embedding_model=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
        cache_dir=Path(emb_cfg.get("cache_dir", "data/processed/embeddings")),
        train_split=data_cfg.get("train_split", 0.7),
        val_split=data_cfg.get("val_split", 0.15),
        test_split=data_cfg.get("test_split", 0.15),
        target_transform=data_cfg.get("target_transform", "log1p"),
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.get_splits()
    train_ids, val_ids, test_ids = dataset.get_split_video_ids()
    logger.info(f"Dataset loaded: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    # Create model
    model = create_model_from_config(cfg)
    
    # Initialize trainer
    trainer = ModelTrainer(runs_dir="runs")
    
    # Define model factory for CV
    def model_factory():
        return create_model_from_config(cfg)
    
    # Train with CV
    results = trainer.train_with_cv(
        model_factory=model_factory,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_folds=5,
        run_name=run_name,
        config=cfg,
        train_video_ids=train_ids,
        test_video_ids=test_ids,
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Run directory: {results['run_dir']}")
    logger.info(f"CV Metrics (Mean):")
    for metric, value in results["cv_metrics"].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    if results["test_metrics"]:
        logger.info(f"Test Metrics:")
        for metric, value in results["test_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
