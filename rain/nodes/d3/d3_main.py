from rain.nodes.d3.src.d3 import D3
from river.tree import HoeffdingTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np

def run_d3(data):
    print('D3')

    label_column = data.columns[-1]
    unique_labels = data[label_column].unique()
    binaries = set(unique_labels) <= {0, 1}

    features = data.iloc[:, :-1].to_dict(orient='records')
    labels = data.iloc[:, -1].values
    final_labels = []
    predictions = []

    w = int(data.shape[0] / 10)
    rho = 0.5
    auc = 0.7

    model = D3(w, rho, len(features[0]), auc)

    stream_clf = HoeffdingTreeClassifier()
    feature_split, labels_split = features[:int(w * rho)], labels[:int(w * rho)]
    
    for x, y in zip(feature_split, labels_split):
        stream_clf.learn_one(x, y)

    for i in range(len(feature_split), len(features)):
        feature_tmp = features[i]
        final_label_tmp = labels[i]

        feature_np = np.array(list(feature_tmp.values())).reshape(1, -1)

        if model.isEmpty():
            model.addInstance(feature_np, np.array([final_label_tmp]))
            prediction_tmp = stream_clf.predict_one(feature_tmp)
            predictions.append(prediction_tmp)
            final_labels.append(final_label_tmp)
            stream_clf.learn_one(feature_tmp, final_label_tmp)
        else:
            if model.driftCheck(auc):
                stream_clf = HoeffdingTreeClassifier()
                current_data = model.getCurrentData()
                current_labels = model.getCurrentLabels()

                for x, y in zip(current_data, current_labels):
                    x_dict = {f'feature_{i}': value for i, value in enumerate(x.flatten())}
                    stream_clf.learn_one(x_dict, y)
                
                prediction_tmp = stream_clf.predict_one(feature_tmp)
                predictions.append(prediction_tmp)
                final_labels.append(final_label_tmp)
                stream_clf.learn_one(feature_tmp, final_label_tmp)
                model.addInstance(feature_np, np.array([final_label_tmp]))
            else:
                prediction_tmp = stream_clf.predict_one(feature_tmp)
                predictions.append(prediction_tmp)
                final_labels.append(final_label_tmp)
                stream_clf.learn_one(feature_tmp, final_label_tmp)
                model.addInstance(feature_np, np.array([final_label_tmp]))

    if binaries == True:
        return pd.DataFrame({
            'algorithm': 'D3',
            'f1_score': round(float(f1_score(labels, predictions, average='binary')), 3),
            'accuracy': round(float(accuracy_score(labels, predictions)), 3),
            'precision': round(float(precision_score(labels, predictions, average='binary')), 3),
            'recall': round(float(recall_score(labels, predictions, average='binary')), 3)
        })
    else:
        return pd.DataFrame({
            'algorithm': 'D3',
            'f1_score': round(float(f1_score(labels, predictions, average='macro')), 3),
            'precision': round(float(precision_score(labels, predictions, average='macro')), 3),
            'recall': round(float(recall_score(labels, predictions, average='macro')), 3)
        })