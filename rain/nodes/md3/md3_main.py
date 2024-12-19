import pandas as pd
from rain.nodes.md3.src.md3 import MD3
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

def run_md3(data):
    print('MD3')

    label_column = data.columns[-1]
    unique_labels = data[label_column].unique()
    binaries = set(unique_labels) <= {0, 1}

    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    predictions = []

    clf = SVC(kernel="linear")
    clf.fit(features, labels)
    md3 = MD3(clf=clf)
    md3.set_reference(features.join(labels), target_name=labels.name)

    for i in range(len(features)):
        md3.update(features.iloc[i:i+1])
        predictions.append(1 if md3.drift_state == "drift" else 0)

    if binaries == True:
        return pd.DataFrame({
            'algorithm': 'Md3',
            'f1_score': round(float(f1_score(labels, predictions, average='binary')), 3),
            'accuracy': round(float(accuracy_score(labels, predictions)), 3),
            'precision': round(float(precision_score(labels, predictions, average='binary')), 3),
            'recall': round(float(recall_score(labels, predictions, average='binary')), 3)
        })
    else:
        return pd.DataFrame({
            'algorithm': 'Md3',
            'f1_score': round(float(f1_score(labels, predictions, average='macro')), 3),
            'precision': round(float(precision_score(labels, predictions, average='macro')), 3),
            'recall': round(float(recall_score(labels, predictions, average='macro')), 3)
        })