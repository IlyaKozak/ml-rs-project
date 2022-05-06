from pathlib import Path
from joblib import dump

import click
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

from .pipeline import create_pipeline
from .data import get_dataset, get_split_dataset

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
@click.option(
    "--logreg-c",
    default=1.0,
    type=float
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int, 
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float
) -> None:
    features, target = get_dataset(
        dataset_path
    )

    features_train, features_val, target_train, target_val = get_split_dataset(
        dataset_path,
        random_state,
        test_split_ratio
    )

    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)

        scoring = {
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "mcc": make_scorer(matthews_corrcoef)
        }

        scores = cross_validate(
            pipeline, 
            features, 
            target, 
            scoring=scoring
        )

        pipeline.fit(features, target)

        accuracy = pd.Series(scores["test_accuracy"]).mean()
        click.echo(f"Accuracy: {accuracy}")
        f1 = pd.Series(scores["test_f1"]).mean()
        click.echo(f"F1 score: {f1}")
        mcc = pd.Series(scores["test_mcc"]).mean()
        click.echo(f"Matthews correlation coefficient (MCC): {mcc}")

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("mcc", mcc)

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}")
