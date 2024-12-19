import pandas as pd
import torch
from rain.nodes.klcpd.src.model import KL_CPD
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def run_klcpd(data):
    label_column = data.columns[-1]
    unique_labels = data[label_column].unique()
    binaries = set(unique_labels) <= {0, 1}
    
    labels = data.iloc[:, -1].values
    features = data.iloc[:, :-1].values

    model = KL_CPD(features.shape[1]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.fit(features)
    predictions = (model.predict(features) > 0.5).astype(int)

    if binaries == True:
        return pd.DataFrame({
            'algorithm': 'Klcpd',
            'f1_score': round(float(f1_score(labels, predictions, average='binary')), 3),
            'accuracy': round(float(accuracy_score(labels, predictions)), 3),
            'precision': round(float(precision_score(labels, predictions, average='binary')), 3),
            'recall': round(float(recall_score(labels, predictions, average='binary')), 3)
        })
    else:
        return pd.DataFrame({
            'algorithm': 'Klcpd',
            'f1_score': round(float(f1_score(labels, predictions, average='macro')), 3),
            'precision': round(float(precision_score(labels, predictions, average='macro')), 3),
            'recall': round(float(recall_score(labels, predictions, average='macro')), 3)
        })