from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_pipeline(
    model: str,
    use_scaler: bool,
    max_iter: int,
    logrec_C: float,
    max_depth: int,
    n_estimators: int,
    use_psa: bool,
    random_state: int,
) -> Pipeline:
    pipeline_steps = []

    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if use_psa:
        pipeline_steps.append(("pca", PCA()))

    if model == "knn":
        clf = KNeighborsClassifier()
    elif model == "rfc":
        clf = RandomForestClassifier(
            random_state=random_state, max_depth=max_depth, n_estimators=n_estimators
        )
    else:
        clf = LogisticRegression(
            random_state=random_state, max_iter=max_iter, C=logrec_C
        )

    pipeline_steps.append(("classifier", clf))

    return Pipeline(steps=pipeline_steps)
