from rain.nodes.nndvi.src.nndvi import NNDVI
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd

def run_nndvi(data):
    print('NNDVI')

    label_column = data.columns[-1]
    unique_labels = data[label_column].unique()
    binaries = set(unique_labels) <= {0, 1}

    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1].to_numpy() 

    batch_size = int(data.shape[0] / 10) # 50
    initial = batch_size * 2 # 100

    det = NNDVI(k_nn=30, sampling_times=500, alpha=0.05)
    det.set_reference(features[:initial])  

    status = []  
    for data_batch in range(initial, len(features), batch_size):
        batch = features[data_batch: data_batch + batch_size]
        if len(batch) > 1:
            det.update(batch) 
            status.append(det.drift_state)

    predictions = [[1] * batch_size if status[i] is not None else [0] * batch_size for i in range(len(status))]
    predictions = np.r_[[0] * initial, np.hstack(predictions)] 
    predictions = predictions[:len(labels)]

    if binaries == True:
        return pd.DataFrame({
            'algorithm': 'Nndvi',
            'f1_score': round(float(f1_score(labels, predictions, average='binary')), 3),
            'accuracy': round(float(accuracy_score(labels, predictions)), 3),
            'precision': round(float(precision_score(labels, predictions, average='binary')), 3),
            'recall': round(float(recall_score(labels, predictions, average='binary')), 3)
        })
    else:
        return pd.DataFrame({
            'algorithm': 'Nndvi',
            'f1_score': round(float(f1_score(labels, predictions, average='macro')), 3),
            'precision': round(float(precision_score(labels, predictions, average='macro')), 3),
            'recall': round(float(recall_score(labels, predictions, average='macro')), 3)
        })