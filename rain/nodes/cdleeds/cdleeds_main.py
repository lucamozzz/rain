import numpy as np
from river import tree
from rain.nodes.cdleeds.src.cdleeds import CDLEEDS
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd

def run_cdleeds(data): 
    label_column = data.columns[-1]
    unique_labels = data[label_column].unique()
    binaries = set(unique_labels) <= {0, 1}

    features = data.iloc[:, :-1].to_dict(orient='records')
    labels = data.iloc[:, -1].values
    final_labels = []
    predictions = []

    detector = CDLEEDS()
    model = tree.HoeffdingTreeClassifier()

    portion = int(len(data) / 10)
    features_init = features[:portion]
    labels_init = labels[:portion]

    unique_labels = set(labels_init)
    if len(unique_labels) < 2:
        missing_class = 1 if 0 in unique_labels else 0
        synthetic_feature = {k: v for k, v in features_init[0].items()}
        labels_init = np.append(labels_init, missing_class)
        features_init.append(synthetic_feature)

    for x, y in zip(features_init, labels_init):
        model.learn_one(x, y)

    baseline_features = pd.DataFrame(features_init).mean().to_dict()
    probabilities = model.predict_proba_one(baseline_features)
    if len(probabilities) < 2:
        probabilities = {0: probabilities.get(0, 0.0), 1: probabilities.get(1, 0.0)}
    detector.set_baseline(np.array(list(probabilities.values())).reshape(1, -1))

    for feature in features_init:
        feature_array = np.array(list(feature.values())).reshape(1, -1)
        detector.add_to_monitored_sample(feature_array)

    for i in range(portion, len(features)):
        feature = features[i]
        label = labels[i]

        feature_array = np.array(list(feature.values())).reshape(1, -1)

        probabilities_dict = {0: 0.0, 1: 0.0}
        tmp = model.predict_proba_one(feature)
        probabilities_dict.update(tmp)
        probabilities = np.array(list(probabilities_dict.values())).reshape(1, -1)
        
        detector.partial_fit(feature_array, probabilities)

        detector.detect_local_change()

        baseline = 0.001 * feature_array + (1 - 0.001) * np.array(list(baseline_features.values())).reshape(1, -1)
        detector.set_baseline(np.array(list(model.predict_proba_one(dict(zip(baseline_features.keys(), baseline.flatten()))).values())).reshape(1, -1))

        model.learn_one(feature, label)

        prediction = model.predict_one(feature)
        predictions.append(prediction)
        final_labels.append(label)

    if binaries == True:
        return {
            'algorithm': 'Cdleeds',
            'dataset': data.file_name,
            'f1_score': round(float(f1_score(final_labels, predictions, average='binary')), 3),
            'accuracy': round(float(accuracy_score(final_labels, predictions)), 3),
            'precision': round(float(precision_score(final_labels, predictions, average='binary')), 3),
            'recall': round(float(recall_score(final_labels, predictions, average='binary')), 3)
        }
    else:
        return {
            'algorithm': 'Cdleeds',
            'dataset': data.file_name,
            'f1_score': round(float(f1_score(final_labels, predictions, average='macro')), 3),
            'precision': round(float(precision_score(final_labels, predictions, average='macro')), 3),
            'recall': round(float(recall_score(final_labels, predictions, average='macro')), 3)
        }