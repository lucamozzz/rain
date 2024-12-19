import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from rain.nodes.md3.src.detector import DriftDetector


class MD3(DriftDetector):

    input_type = "stream"

    def calculate_margin_inclusion_signal(self, sample, clf):

        w = np.array(clf.coef_[0])
        intercept = np.array(clf.intercept_)
        b = intercept[0] / w[1]

        mis = np.abs(np.dot(w, sample) + b)

        if mis <= 1:
            return 1
        else:
            return 0

    def __init__(
        self,
        clf,
        margin_calculation_function=calculate_margin_inclusion_signal,
        sensitivity=2,
        k=10,
        oracle_data_length_required=None,
    ):

        super().__init__()
        self.classifier = clf
        self.margin_calculation_function = margin_calculation_function
        self.sensitivity = sensitivity
        self.k = k
        self.oracle_data_length_required = oracle_data_length_required
        self.oracle_data = None
        self.waiting_for_oracle = False

    def set_reference(self, X, y_true=None, y_pred=None, target_name=None):

        self.reference_batch_features = copy.deepcopy(
            X.loc[:, X.columns != target_name]
        )
        self.reference_batch_target = copy.deepcopy(X.loc[:, X.columns == target_name])

        self.reference_distribution = self.calculate_distribution_statistics(X)

        if self.oracle_data_length_required is None:
            self.oracle_data_length_required = self.reference_distribution["len"]

        self.forgetting_factor = (
            self.reference_distribution["len"] - 1
        ) / self.reference_distribution["len"]
        self.curr_margin_density = self.reference_distribution["md"]

    def calculate_distribution_statistics(self, data):

        duplicate_classifier = clone(self.classifier)

        # prepare the cross-validation procedure
        margin_densities = []
        accuracies = []
        cv = KFold(n_splits=self.k, random_state=42, shuffle=True)

        # perform k-fold cross validation to acquire distribution margin density and acuracy values
        for train_index, test_index in cv.split(self.reference_batch_features):
            X_train, X_test = (
                self.reference_batch_features.iloc[train_index],
                self.reference_batch_features.iloc[test_index],
            )
            y_train, y_test = (
                self.reference_batch_target.iloc[train_index],
                self.reference_batch_target.iloc[test_index],
            )

            duplicate_classifier.fit(X_train, y_train.values.ravel())

            # record margin inclusion signals for all samples in this test band
            signal_func_values = []
            for i in range(len(X_test)):
                sample_np_array = X_test.iloc[i].to_numpy()
                margin_inclusion_signal = self.margin_calculation_function(
                    self, sample_np_array, duplicate_classifier
                )
                signal_func_values.append(margin_inclusion_signal)

            # record margin density over this test band
            margin_density = sum(signal_func_values) / len(signal_func_values)
            margin_densities.append(margin_density)

            # record accuracy of prediction over this test band
            y_pred = duplicate_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # record performance - margin density
        md = np.mean(margin_densities)
        md_std = np.std(margin_densities)

        # record performance - accuracy
        acc = np.mean(accuracies)
        acc_std = np.std(accuracies)

        # return reference distribution statistics
        return {
            "len": len(data),
            "md": md,
            "md_std": md_std,
            "acc": acc,
            "acc_std": acc_std,
        }

    def update(self, X, y_true=None, y_pred=None):

        if self.waiting_for_oracle == True:
            raise ValueError(
                """give_oracle_label method must be called to provide detector with a
                labeled sample to confirm or rule out drift."""
            )

        if len(X) != 1:
            raise ValueError(
                """This method is only available for data inputs in the form of 
                a Pandas DataFrame with exactly 1 record."""
            )

        if self.drift_state == "drift":
            self.reset()

        super().update(X, y_true, y_pred)

        sample_np_array = X.to_numpy()[0]
        margin_inclusion_signal = self.margin_calculation_function(
            self, sample_np_array, self.classifier
        )
        self.curr_margin_density = (
            self.forgetting_factor * self.curr_margin_density
            + (1 - self.forgetting_factor) * margin_inclusion_signal
        )

        warning_level = np.abs(
            self.curr_margin_density - self.reference_distribution["md"]
        )
        warning_threshold = self.sensitivity * self.reference_distribution["md_std"]

        if warning_level > warning_threshold:
            self.drift_state = "warning"
            self.waiting_for_oracle = True

    def give_oracle_label(self, labeled_sample):

        if self.waiting_for_oracle != True:
            raise ValueError(
                """give_oracle_label method can be called only when a drift warning has
                been issued and drift needs to be confirmed or ruled out."""
            )

        if len(labeled_sample) != 1:
            raise ValueError(
                """This method is only available for data inputs in the form of 
                a Pandas DataFrame with exactly 1 record."""
            )

        labeled_columns = list(labeled_sample.columns)
        feature_columns = list(self.reference_batch_features.columns)
        target_column = list(self.reference_batch_target.columns)
        reference_columns = feature_columns + target_column
        if len(labeled_columns) != len(reference_columns) or set(
            labeled_columns
        ) != set(reference_columns):
            raise ValueError(
                """give_oracle_label method can be called only with a sample containing
                the same number and names of columns as the original reference distribution."""
            )

        self.drift_state = None

        if self.oracle_data is None:
            self.oracle_data = labeled_sample
        else:
            self.oracle_data = pd.concat(
                [self.oracle_data, labeled_sample], ignore_index=True
            )

        if len(self.oracle_data) == self.oracle_data_length_required:
            X_test, y_test = (
                self.oracle_data[feature_columns],
                self.oracle_data[target_column],
            )
            y_pred = self.classifier.predict(X_test)
            acc_labeled_samples = accuracy_score(y_test, y_pred)

            drift_level = self.reference_distribution["acc"] - acc_labeled_samples
            drift_threshold = self.sensitivity * self.reference_distribution["acc_std"]

            if drift_level > drift_threshold:
                self.drift_state = "drift"

            # update reference distribution
            self.set_reference(self.oracle_data, target_name=target_column[0])
            self.oracle_data = None
            self.waiting_for_oracle = False

    def reset(self):
        super().reset()
        self.curr_margin_density = self.reference_distribution["md"]
