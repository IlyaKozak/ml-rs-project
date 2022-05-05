from pathlib import Path
from joblib import dump

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
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path)
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
@click.option(
    "--max-iter",
    default=100,
    type=int
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int, 
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
      dataset_path, 
      random_state, 
      test_split_ratio
    )

    pipeline = create_pipeline(use_scaler, max_iter, random_state)
    pipeline.fit(features_train, target_train)

    accuracy = accuracy_score(target_val, pipeline.predict(features_val))
    click.echo(f"Accuracy: {accuracy}")

    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}")
