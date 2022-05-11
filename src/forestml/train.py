from pathlib import Path
from joblib import dump
import warnings

import click
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
@click.option(
    "--model",
    default="logreg",
    type=click.Choice(["logreg", "knn", "rfc"])
)
@click.option(
    "--max-depth",
    default=None,
    type=int
)
@click.option(
    "--n-estimators",
    default=100,
    type=int
)
@click.option(
    "--use-psa",
    default=False,
    type=bool
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int, 
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    model: str,
    max_depth: int,
    n_estimators: int,
    use_psa: bool
) -> None:
    warnings.filterwarnings("ignore")

    features, target = get_dataset(
        dataset_path
    )

    features_train, features_val, target_train, target_val = get_split_dataset(
        dataset_path,
        random_state,
        test_split_ratio
    )

    with mlflow.start_run():
        # define search space
        param_logreg = {}
        clf_logreg = LogisticRegression(random_state=random_state)
        param_logreg["classifier"] = [clf_logreg]
        param_logreg["classifier__C"] = [0.7, 0.85, 1]
        param_logreg["classifier__max_iter"] = [100, 200, 500]

        pipeline = Pipeline([("classifier", clf_logreg)])

        param_rfc = {}
        clf_rfc = RandomForestClassifier(random_state=random_state)
        param_rfc["classifier"] = [clf_rfc]
        param_rfc["classifier__max_depth"] = [5, 10, None]
        param_rfc["classifier__n_estimators"] = [75, 100, 150]

        params = [param_logreg, param_rfc]

        cv_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)
        outer_results = []

        for train_ix, test_ix in cv_outer.split(features):
            X_train, X_test = features.iloc[train_ix, :], features.iloc[test_ix, :]
            y_train, y_test = target[train_ix], target[test_ix]

            cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)

            # define search
            search = GridSearchCV(pipeline, params, scoring="accuracy", cv=cv_inner, refit=True)

            # execute search
            result = search.fit(X_train, y_train)
            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            # evaluate model on the hold out dataset
            yhat = best_model.predict(X_test)
            # evaluate the model
            acc = accuracy_score(y_test, yhat)
            # store the result
            outer_results.append(acc)
            # report progress
            mlflow.log_param("_model", result.best_params_)
            mlflow.log_metric("accuracy", acc)
            click.echo('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        # summarize the estimated performance of the model
        click.echo('Accuracy: %.3f (%.3f)' % (pd.Series(outer_results).mean(), pd.Series(outer_results).std()))

        pipeline = create_pipeline(
            model="rfc",
            use_scaler=use_scaler,
            max_depth=None,
            n_estimators=100,
            random_state=random_state
        )

        pipeline.fit(features, target)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        click.echo(f"Accuracy: {accuracy}")
        f1 = f1_score(y_test, y_pred)
        click.echo(f"F1 score: {f1}")
        mcc = matthews_corrcoef(y_test, y_pred)
        click.echo(f"Matthews correlation coefficient (MCC): {mcc}")

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("mcc", mcc)

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}")
