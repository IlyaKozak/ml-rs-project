This repo uses [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv) dataset.

## Usage
This package allows you to train model to predict an integer classification for the forest cover type.
1. Clone this repository to your machine.
2. Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. Go to the commit before nested cross-validation (go to the commit 32e25fb48437250bc226450ac9230f2c96e1020d). To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

## Development

TODO:
* error cases test(s)
* valid case test
* mypy (to type annotate the code)
* nox (to combine steps of testing and linting into a single command)
* github actions (to run tests and linters)
