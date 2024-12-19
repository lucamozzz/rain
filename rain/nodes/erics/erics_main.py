import pandas as pd
from rain.nodes.erics.src.erics import ERICS
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def run_erics(data):
    print('erics')

    label_column = data.columns[-1]
    unique_labels = data[label_column].unique()
    binaries = set(unique_labels) <= {0, 1}

    erics = ERICS(n_param=len(data.columns) - 1,
              window_mvg_average=50,
              window_drift_detect=50,
              beta=0.001,
              base_model='probit')

    labels = []
    predictions = []

    for index, row in data.iterrows():
        x = row.iloc[:-1].values
        y = row.iloc[-1]

        global_drift, partial_drift = erics.check_drift(x, y)

        labels.append(y)
        predictions.append(1 if global_drift or partial_drift else 0)

    if binaries == True:
        return pd.DataFrame({
            'algorithm': 'Erics',
            'f1_score': round(float(f1_score(labels, predictions, average='binary')), 3),
            'accuracy': round(float(accuracy_score(labels, predictions)), 3),
            'precision': round(float(precision_score(labels, predictions, average='binary')), 3),
            'recall': round(float(recall_score(labels, predictions, average='binary')), 3)
        })
    else:
        return pd.DataFrame({
            'algorithm': 'Erics',
            'f1_score': round(float(f1_score(labels, predictions, average='macro')), 3),
            'precision': round(float(precision_score(labels, predictions, average='macro')), 3),
            'recall': round(float(recall_score(labels, predictions, average='macro')), 3)
        })