from abc import ABC, abstractmethod
from pandas import DataFrame
import numpy as np
import copy


class StreamingDetector(ABC):

    def __init__(self, *args, **kwargs):
        self._total_samples = 0
        self._samples_since_reset = 0
        self._drift_state = None
        self._input_cols = None
        self._input_col_dim = None

    @abstractmethod
    def update(self, X, y_true, y_pred):
        self.total_samples += 1
        self.samples_since_reset += 1

    @abstractmethod
    def reset(self, *args, **kwargs):
        self.samples_since_reset = 0
        self.drift_state = None

    def _validate_X(self, X):
        if isinstance(X, DataFrame):
            # The first update with a dataframe will constrain subsequent input.
            if self._input_cols is None:
                self._input_cols = X.columns
                self._input_col_dim = len(self._input_cols)
            elif self._input_cols is not None:
                if not X.columns.equals(self._input_cols):
                    raise ValueError(
                        "Columns of new data must match with columns of prior data."
                    )
            ary = X.values
        else:
            ary = copy.copy(X)
            ary = np.array(ary)
            if len(ary.shape) <= 1:
                # only one sample should be passed, so coerce column vectors (e.g. pd.Series) to rows
                ary = ary.reshape(1, -1)
            if self._input_col_dim is None:
                # This allows starting with a dataframe, then later passing bare
                # numpy arrays. For now, assume users are not miscreants.
                self._input_col_dim = ary.shape[1]
            elif self._input_col_dim is not None:
                if ary.shape[1] != self._input_col_dim:
                    raise ValueError(
                        "Column-dimension of new data must match prior data."
                    )

        if ary.shape[0] != 1:
            raise ValueError(
                "Input for streaming detectors should contain only one observation."
            )
        return ary

    def _validate_y(self, y):
        ary = np.array(y).ravel()
        if ary.shape != (1,):
            raise ValueError(
                "Input for streaming detectors should contain only one observation."
            )
        return ary

    def _validate_input(self, X, y_true, y_pred):
        if X is not None:
            X = self._validate_X(X)
        if y_true is not None:
            y_true = self._validate_y(y_true)
        if y_pred is not None:
            y_pred = self._validate_y(y_pred)
        return X, y_true, y_pred

    @property
    def total_samples(self):
        return self._total_samples

    @total_samples.setter
    def total_samples(self, value):
        self._total_samples = value

    @property
    def samples_since_reset(self):
        return self._samples_since_reset

    @samples_since_reset.setter
    def samples_since_reset(self, value):
        self._samples_since_reset = value

    @property
    def drift_state(self):
        return self._drift_state

    @drift_state.setter
    def drift_state(self, value):
        if value not in ("drift", "warning", None):
            raise ValueError("tbd")
        else:
            self._drift_state = value


class BatchDetector(ABC):
    def __init__(self, *args, **kwargs):
        self._total_batches = 0
        self._batches_since_reset = 0
        self._drift_state = None
        self._input_cols = None
        self._input_col_dim = None

    @abstractmethod
    def update(self, X, y_true, y_pred):
        self.total_batches += 1
        self.batches_since_reset += 1

    @abstractmethod
    def set_reference(self, X, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        self.batches_since_reset = 0
        self.drift_state = None

    def _validate_X(self, X):
        if isinstance(X, DataFrame):
            # The first update with a dataframe will constrain subsequent input.
            if self._input_cols is None:
                self._input_cols = X.columns
                self._input_col_dim = len(self._input_cols)
            elif self._input_cols is not None:
                if not X.columns.equals(self._input_cols):
                    raise ValueError(
                        "Columns of new data must match with columns of prior data."
                    )
            ary = X.values
        else:
            ary = copy.copy(X)
            ary = np.array(ary)
            if len(ary.shape) <= 1:
                # Batch size of 1 will break downstream - don't allow it.
                # Attempts to coerce a row vector into a column vector.
                ary = ary.reshape(-1, 1)
            if self._input_col_dim is None:
                # This allows starting with a dataframe, then later passing bare
                # numpy arrays. For now, assume users are not miscreants.
                self._input_col_dim = ary.shape[1]
            elif self._input_col_dim is not None:
                if ary.shape[1] != self._input_col_dim:
                    raise ValueError(
                        "Column-dimension of new data must match prior data."
                    )
        if ary.shape[0] <= 1:
            raise ValueError(
                "Input for batch detectors should contain more than one observation."
            )
        return ary

    def _validate_y(self, y):
        ary = np.array(y)
        if len(ary.shape) <= 1:
            ary = ary.reshape(1, -1)
        if ary.shape[0] == 1:
            raise ValueError(
                "Input for batch detectors should contain more than one obsevation."
            )
        if ary.shape[1] != 1:
            raise ValueError("y input for detectors should contain only one column.")
        return ary

    def _validate_input(self, X, y_true, y_pred):
        if X is not None:
            X = self._validate_X(X)
        if y_true is not None:
            y_true = self._validate_y(y_true)
        if y_pred is not None:
            y_pred = self._validate_y(y_pred)
        return X, y_true, y_pred

    @property
    def total_batches(self):
        return self._total_batches

    @total_batches.setter
    def total_batches(self, value):
        self._total_batches = value

    @property
    def batches_since_reset(self):
        return self._batches_since_reset

    @batches_since_reset.setter
    def batches_since_reset(self, value):
        self._batches_since_reset = value

    @property
    def drift_state(self):
        return self._drift_state

    @drift_state.setter
    def drift_state(self, value):
        if value not in ("drift", "warning", None):
            raise ValueError("tbd")
        else:
            self._drift_state = value


class DriftDetector(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._total_updates = 0
        self._updates_since_reset = 0
        self._drift_state = None
        self._input_type = None

    @abstractmethod
    def update(self, X, y_true, y_pred):
        self.total_updates += 1
        self.updates_since_reset += 1

    @abstractmethod
    def reset(self, *args, **kwargs):
        self.updates_since_reset = (
            0  # number of elements the detector has been updated with since last reset
        )
        self.drift_state = None

    @property
    def total_updates(self):
        return self._total_updates

    @total_updates.setter
    def total_updates(self, value):
        self._total_updates = value

    @property
    def updates_since_reset(self):
        return self._updates_since_reset

    @updates_since_reset.setter
    def updates_since_reset(self, value):
        self._updates_since_reset = value

    @property
    def drift_state(self):
        return self._drift_state

    @drift_state.setter
    def drift_state(self, value):
        if value not in ("drift", "warning", None):
            raise ValueError(
                """DriftDetector._drift_state must be ``"drift"``, 
                ``"warning"``, or ``None``."""
            )
        else:
            self._drift_state = value

    @property
    @abstractmethod
    def input_type(self):
        return self._input_type
