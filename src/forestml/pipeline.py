from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(random_state: int) -> Pipeline:
    scaler = StandardScaler()

    clf = LogisticRegression(random_state=random_state)

    pipeline = Pipeline(steps=[("scaler", scaler), ("classifier", clf)])

    return pipeline
    