from rain.nodes.studd.src.batch import STUDD
from sklearn.ensemble import RandomForestClassifier as RF
from river.drift import PageHinkley as PHT
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd

def run_studd(data):
    label_column = data.columns[-1]
    unique_labels = data[label_column].unique()
    binaries = set(unique_labels) <= {0, 1}

    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    window = int(data.shape[0] / 10)  #700
    delta = 0.002

    model = STUDD(X=features, y=labels, n_train=window)
    model.initial_fit(model=RF(), std_model=RF())
    labels, predictions = model.drift_detection_std(
        datastream_=model.datastream,
        model_=model.base_model,
        std_model_=model.student_model,
        n_train_=model.n_train,
        n_samples=window,
        delta=delta / 2,
        upd_model=True,
        upd_std_model=True,
        detector=PHT
    )

    if binaries == True:
        return pd.DataFrame({
            'algorithm': 'Studd',
            'f1_score': round(float(f1_score(labels, predictions, average='binary')), 3),
            'accuracy': round(float(accuracy_score(labels, predictions)), 3),
            'precision': round(float(precision_score(labels, predictions, average='binary')), 3),
            'recall': round(float(recall_score(labels, predictions, average='binary')), 3)
        })
    else:
        return pd.DataFrame({
            'algorithm': 'Studd',
            'f1_score': round(float(f1_score(labels, predictions, average='macro')), 3),
            'precision': round(float(precision_score(labels, predictions, average='macro')), 3),
            'recall': round(float(recall_score(labels, predictions, average='macro')), 3)
        })