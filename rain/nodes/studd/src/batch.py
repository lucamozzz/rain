from river.stream import iter_pandas
from river.drift import PageHinkley as PHT
import copy
import numpy as np
import pandas as pd

class STUDD:

    def __init__(self, X, y, n_train):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        self.datastream = iter_pandas(X_df, y_series)
        self.n_train = n_train
        self.W = n_train
        self.base_model = None
        self.student_model = None
        self.init_training_data = None

    def initial_fit(self, model, std_model):
        X_tr, y_tr = [], []
        for i, (xi, yi) in enumerate(self.datastream):
            X_tr.append(list(xi.values()))
            y_tr.append(yi)
            if i >= self.n_train - 1:
                break

        X_tr = np.array(X_tr)
        y_tr = np.array(y_tr)

        model.fit(X_tr, y_tr)
        yhat_tr = model.predict(X_tr)
        std_model.fit(X_tr, yhat_tr)

        self.base_model = model
        self.student_model = std_model
        self.init_training_data = dict({"X": X_tr, "y": y_tr, "y_hat": yhat_tr})


    @staticmethod
    def drift_detection_std(datastream_, model_, std_model_, n_train_, delta, n_samples, upd_model=False,
                            upd_std_model=True, detector=PHT):

        base_model = copy.deepcopy(model_)
        student_model = copy.deepcopy(std_model_)
        n_train = copy.deepcopy(n_train_)

        std_detector = detector(delta=delta)
        std_alarms = []

        iter = n_train
        n_updates = 0
        samples_used = 0
        predictions = []
        y_buffer, labels = [], []
        X_buffer, X_hist = [], []

        while True:
            try:
                xi, yi = next(datastream_)
            except StopIteration:
                break

            X_hist.append(list(xi.values()))
            labels.append(yi)
            X_buffer.append(list(xi.values()))
            y_buffer.append(yi)

            predictions.append(base_model.predict([list(xi.values())])[0])

            value = yi - student_model.predict([list(xi.values())])[0]

            std_detector.update(value)

            if std_detector.drift_detected:
                std_alarms.append(iter)

                if upd_model:
                    X_buffer_np = np.array(X_buffer[-n_samples:])
                    y_buffer_np = np.array(y_buffer[-n_samples:])

                    base_model.fit(X_buffer_np, y_buffer_np)
                    n_updates += 1
                    samples_used += len(y_buffer_np)

            iter += 1
        
        return labels, predictions