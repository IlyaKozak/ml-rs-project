from pathlib import Path

import click
from sklearn.metrics import accuracy_score

from .pipeline import create_pipeline
from .data import get_dataset

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--random-state",
    default=42,
    type=int
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True)
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool
)
def train(
    dataset_path: Path, 
    random_state: int, 
    test_split_ratio: float,
    use_scaler: bool
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
      dataset_path, 
      random_state, 
      test_split_ratio
    )

    pipeline = create_pipeline(use_scaler, random_state)
    pipeline.fit(features_train, target_train)

    accuracy = accuracy_score(target_val, pipeline.predict(features_val))
    click.echo(f"Accuracy: {accuracy}")
