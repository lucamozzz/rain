import os
import random
from typing import Dict, List
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
from rain.nodes.dynamo.src.pyclee.forgetting import ForgettingMethod
from rain.nodes.dynamo.src.dynamo import DynAmo
from rain.nodes.dynamo.src.init import create_config, run_dyclee


seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def run_dynamo(data):
    print('DYNAMO')

    label_column = data.columns[-1]
    unique_labels = data[label_column].unique()
    binaries = set(unique_labels) <= {0, 1}

    labels = data.iloc[:, -1]
    features = data.iloc[:, :-1]
    signal = features

    cfg = create_config(int(data.shape[0]), 5000)

    hyperbox_fraction: float  = cfg['hyperbox_fraction']
    forgetting_method: ForgettingMethod = cfg['forgetting_instance']
    dynamo: DynAmo = cfg['dynamo']

    start_offset = cfg['start_offset']
    labels = labels.values[start_offset:]

    new_signal: pd.DataFrame = run_dyclee(
        signal.values[start_offset:],
        hyperbox_fraction=hyperbox_fraction,
        forgetting_method=forgetting_method
    )

    dynamo.signal = new_signal
    predictions: List[int] = dynamo.run()

    if binaries == True:
        return pd.DataFrame({
            'algorithm': 'Dynamo',
            'f1_score': round(float(f1_score(labels, predictions, average='binary')), 3),
            'accuracy': round(float(accuracy_score(labels, predictions)), 3),
            'precision': round(float(precision_score(labels, predictions, average='binary')), 3),
            'recall': round(float(recall_score(labels, predictions, average='binary')), 3)
        })
    else:
        return pd.DataFrame({
            'algorithm': 'Dynamo',
            'f1_score': round(float(f1_score(labels, predictions, average='macro')), 3),
            'precision': round(float(precision_score(labels, predictions, average='macro')), 3),
            'recall': round(float(recall_score(labels, predictions, average='macro')), 3)
        })