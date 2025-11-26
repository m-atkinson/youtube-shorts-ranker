"""View the actual training/validation/test data used in a run.

Usage:
    python scripts/view_run_data.py runs/with-data-tracking
    python scripts/view_run_data.py runs/with-data-tracking --split train
    python scripts/view_run_data.py runs/with-data-tracking --split test --output csv
"""
import json
import sqlite3
import sys
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test", "all"], case_sensitive=False),
    default="all",
    help="Which split to view",
)
@click.option(
    "--output",
    type=click.Choice(["table", "csv", "json"], case_sensitive=False),
    default="table",
    help="Output format",
)
@click.option(
    "--db-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/database/shorts_data.db"),
    help="Path to database",
)
def main(run_dir: Path, split: str, output: str, db_path: Path):
    """View the training/validation/test data used in a specific run."""
    
    # Load data splits
    splits_file = run_dir / "data_splits.json"
    if not splits_file.exists():
        click.echo(f"Error: {splits_file} not found", err=True)
        click.echo("This run may not have data split tracking enabled.", err=True)
        sys.exit(1)
    
    with open(splits_file) as f:
        splits_data = json.load(f)
    
    # Determine which video IDs to fetch
    if split == "all":
        video_ids = (
            splits_data["train_video_ids"]
            + splits_data["val_video_ids"]
            + splits_data["test_video_ids"]
        )
        split_labels = (
            ["train"] * len(splits_data["train_video_ids"])
            + ["val"] * len(splits_data["val_video_ids"])
            + ["test"] * len(splits_data["test_video_ids"])
        )
    else:
        video_ids = splits_data[f"{split}_video_ids"]
        split_labels = [split] * len(video_ids)
    
    if not video_ids:
        click.echo(f"No video IDs found for split: {split}")
        sys.exit(0)
    
    # Fetch data from database
    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join("?" * len(video_ids))
        query = f"""
            SELECT
                sm.video_id,
                sm.title,
                sm.description,
                sm.publish_date,
                sm.duration_seconds,
                sm.view_count,
                t.transcript_text
            FROM shorts_metadata sm
            JOIN transcripts t ON sm.video_id = t.video_id
            WHERE sm.video_id IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=video_ids)
    
    # Add split label
    video_id_to_split = dict(zip(video_ids, split_labels))
    df["split"] = df["video_id"].map(video_id_to_split)
    
    # Reorder columns
    cols = ["split", "video_id", "title", "view_count", "duration_seconds", "publish_date", "description", "transcript_text"]
    df = df[cols]
    
    # Output
    if output == "table":
        # Truncate long text for display
        df_display = df.copy()
        df_display["transcript_text"] = df_display["transcript_text"].str[:100] + "..."
        df_display["description"] = df_display["description"].str[:50] + "..."
        click.echo(df_display.to_string(index=False))
    elif output == "csv":
        click.echo(df.to_csv(index=False))
    elif output == "json":
        click.echo(df.to_json(orient="records", indent=2))
    
    # Summary
    click.echo(f"\n{'='*60}", err=True)
    click.echo(f"Run: {run_dir.name}", err=True)
    click.echo(f"Total videos: {len(df)}", err=True)
    if split == "all":
        click.echo(f"  Train: {len(splits_data['train_video_ids'])}", err=True)
        click.echo(f"  Val: {len(splits_data['val_video_ids'])}", err=True)
        click.echo(f"  Test: {len(splits_data['test_video_ids'])}", err=True)
    click.echo(f"{'='*60}", err=True)


if __name__ == "__main__":
    main()
