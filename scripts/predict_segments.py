"""CLI script for predicting top shorts segments from a transcript.

Usage:
    python scripts/predict_segments.py --input <path_to_transcript> --model <path_to_model>
"""
import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.transcript_loader import TranscriptLoader
from src.inference.segmenter import SlidingWindowSegmenter, SentenceWindowSegmenter
from src.inference.predictor import SegmentPredictor
from src.training.embedder import TranscriptEmbedder
from src.utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Predict top YouTube Shorts segments")
    parser.add_argument("--input", required=True, help="Path to input transcript (.txt or .docx)")
    parser.add_argument("--model", default="runs/v5/model.pkl", help="Path to trained model")
    parser.add_argument("--output", default="predictions.json", help="Output JSON file path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of segments to return")
    parser.add_argument("--segmentation-method", choices=["sliding_window", "sentence"], default="sliding_window", help="Segmentation strategy")
    parser.add_argument("--step-size", type=int, default=25, help="Sliding window step size in words (for sliding_window)")
    parser.add_argument("--target-duration", type=int, default=60, help="Target segment duration in seconds (for sentence)")
    parser.add_argument("--overlap-threshold", type=float, default=0.2, help="Max overlap ratio for NMS")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")   # Should this be default all-mpnet-base-v2 because 768 vs 384?
    parser.add_argument("--device", default="cpu", help="Device for embedding model (cpu/cuda)")
    
    args = parser.parse_args()
    
    logger = setup_logger("predict_segments")
    
    # 1. Load Transcript
    logger.info(f"Loading transcript from {args.input}")
    loader = TranscriptLoader()
    try:
        transcript_text = loader.load(args.input)
    except Exception as e:
        logger.error(f"Failed to load transcript: {e}")
        sys.exit(1)
        
    # 2. Segment
    logger.info(f"Segmenting transcript using {args.segmentation_method} method...")
    
    if args.segmentation_method == "sentence":
        segmenter = SentenceWindowSegmenter(target_duration=args.target_duration)
    else:
        segmenter = SlidingWindowSegmenter(step_size=args.step_size)
        
    segments = segmenter.segment(transcript_text)
    
    if not segments:
        logger.error("No segments generated. Transcript might be too short.")
        sys.exit(1)
        
    # 3. Initialize Model & Embedder
    logger.info("Initializing models...")
    embedder = TranscriptEmbedder(model_name=args.embedding_model, device=args.device)
    predictor = SegmentPredictor(model_path=args.model, embedder=embedder)
    
    # 4. Predict & Rank
    logger.info("Predicting top segments...")
    top_segments = predictor.predict_top_segments(
        segments, 
        top_k=args.top_k,
        overlap_threshold=args.overlap_threshold
    )
    
    # 5. Output
    output_data = {
        "input_file": args.input,
        "model_path": args.model,
        "parameters": {
            "segmentation_method": args.segmentation_method,
            "step_size": args.step_size if args.segmentation_method == "sliding_window" else None,
            "target_duration": args.target_duration if args.segmentation_method == "sentence" else None,
            "overlap_threshold": args.overlap_threshold
        },
        "top_segments": top_segments
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
        
    logger.info(f"Saved top {len(top_segments)} predictions to {args.output}")
    
    # Print preview
    print("\nTop Segments:")
    for seg in top_segments:
        print(f"Rank {seg['rank']} (Score: {seg['score']:.4f}):")
        print(f"Duration: ~{seg['estimated_duration']}s")
        print(f"Text: {seg['text'][:100]}...")
        print("-" * 40)

if __name__ == "__main__":
    main()
