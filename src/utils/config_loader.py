"""Configuration loading and validation utilities."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()


import os

class YouTubeConfig(BaseModel):
    """YouTube API configuration."""
    api_key: str = Field(
        default_factory=lambda: os.environ.get("YOUTUBE_API_KEY", ""),
        description="YouTube Data API v3 key"
    )
    channel_id: str = Field(..., description="Target channel ID")
    max_results: int = Field(default=50, description="Max results per API call")
    max_duration_seconds: int = Field(default=60, description="Max video duration for Shorts")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("YouTube API key must be provided in config or YOUTUBE_API_KEY env var")
        return v


class DataPipelineConfig(BaseModel):
    """Data pipeline configuration."""
    youtube: YouTubeConfig
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_dir: Path = Field(default=Path("data/processed"), description="Processed data directory")
    database_path: Path = Field(default=Path("data/database/shorts_data.db"), description="SQLite database path")
    
    # Download settings
    video_format: str = Field(default="mp4", description="Video format for downloads")
    audio_format: str = Field(default="mp3", description="Audio format for extraction")
    
    # Transcription settings
    whisper_model: str = Field(default="base", description="Whisper model size")
    transcription_language: str = Field(default="en", description="Transcription language")
    
    # Processing settings
    batch_size: int = Field(default=10, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Max parallel workers")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Initial retry delay in seconds")
    
    @field_validator('data_dir', 'raw_dir', 'processed_dir', 'database_path', mode='before')
    @classmethod
    def convert_to_path(cls, v):
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    train_split: float = Field(default=0.7, description="Training set ratio")
    val_split: float = Field(default=0.15, description="Validation set ratio")
    test_split: float = Field(default=0.15, description="Test set ratio")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    
    # Model settings
    model_type: str = Field(default="xgboost", description="Model type (xgboost, lightgbm, etc.)")
    target_variable: str = Field(default="view_count", description="Target variable to predict")
    
    # MLflow settings
    experiment_name: str = Field(default="youtube-shorts-predictor", description="MLflow experiment name")
    tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI")


class InferenceConfig(BaseModel):
    """Inference configuration."""
    model_path: Path = Field(..., description="Path to trained model")
    top_k: int = Field(default=5, description="Number of top segments to return")
    min_segment_length: int = Field(default=30, description="Minimum segment length in seconds")
    max_segment_length: int = Field(default=60, description="Maximum segment length in seconds")
    overlap: int = Field(default=15, description="Overlap between segments in seconds")


def load_config(config_path: Path, config_class: type[BaseModel]) -> BaseModel:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        config_class: Pydantic model class for validation
    
    Returns:
        Validated configuration object
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_class(**config_dict)


def save_config(config: BaseModel, config_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object
        config_path: Path to save YAML file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
