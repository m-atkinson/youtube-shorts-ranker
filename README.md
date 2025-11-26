# YouTube Shorts Predictor
## Goal
Extract five transcript segments from a long-form video that are most likely to generate the highest YouTubeShorts views, using a model trained on the channel’s historical Shorts performance. Built with modularity in mind to test different training methods. 


## Features
- **Data Pipeline**:
    - **Scrape Metadata & Download Videos**: Fetches video metadata (views, likes, comments) from a target YouTube channel and downloads videos using yt-dlp.
    - **Transcribe**: Generates transcripts using OpenAI's Whisper model.
- **Model Training**:
    - Train ridge regression model to predict log norm view counts.
- **Inference**:
    - Predict the best segments from new transcripts.


## Background
- Tried tools like Opus Clips found that they did not perform well for a client's YouTube channel. 
- Have got this to a place where it is performing well for the client.
- As time permits will experiment with synthetic data and other channels' content to improve the model.

## Performance
I used Dwarkesh Patel's channel (not my client) for sample data (https://www.youtube.com/@DwarkeshPatel).
* Ridge Regression
    * Sample size: 150 Shorts
    * Spearman correlation: 0.82
    * NDCG@5: 0.95

## Quick Start

### Prerequisites
- Python 3.10 or higher
- YouTube Data API v3 key
- Gemini API key
- Target YouTube channel ID
- ffmpeg (required by Whisper for audio processing)
- Install dependencies `requirements.txt`

### Pipeline
- Setup database `python scripts/setup_database.py`
- Add your channel id to `config/data_pipeline.yaml` 
- Add your Gemini (for embedding) and YouTube (for metadata and video download) API keys to your `.env`
- Download and process training data `python scripts/run_data_pipeline.py --max-videos 50`

### Training
- Delete downloaded videos `rm -rf data/raw/shorts_videos`
- Train model `python scripts/train_model.py --config config/training_gemini.yaml --run-name dp-150`

### Inference
- Run inference `python scripts/predict_segments.py --input data/inference/inputs/ilya-sutskever.txt --model runs/dp-150/model.pkl --output data/inference/outputs/dp-150-predictions.json --segmentation-method sentence --target-duration 60 --embedding-model models/text-embedding-004` 

## Project Structure

```
youtube-shorts-ranker/
├── config/                 # Config files
│   ├── data_pipeline.yaml  # Pipeline configuration
│   └── training_gemini.yaml# Training configuration
├── data/                   # Data storage
│   ├── database/           # SQLite database
│   ├── inference/          # Inference inputs and outputs
│   ├── processed/          # Embeddings
│   └── raw/                # Raw downloads & intermediate files
├── scripts/                # Executable scripts
│   ├── predict_segments.py # Inference script
│   ├── reembed_data.py     # Script to regenerate embeddings to switch embedding model
│   ├── run_data_pipeline.py# Main data collection pipeline
│   ├── setup_database.py   # Database initialization
│   ├── train_model.py      # Model training script
│   └── view_run_data.py    # Helper to inspect database
├── src/                    # Source code
│   ├── data_pipeline/      # Scraper, downloader, transcriber logic
│   ├── inference/          # Prediction and segmentation logic
│   ├── training/           # Dataset and model training logic
│   └── utils/              # Utility functions
├── tests/                  # Unit tests
├── .env.example            # Example environment variables
├── requirements.txt        # Python dependencies
```

## License

This project is licensed under the MIT License.
