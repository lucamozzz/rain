import numpy as np
from scipy.stats import norm

from rain.nodes.nndvi.src.detector import BatchDetector
from rain.nodes.nndvi.src.partitioners import NNSpacePartitioner


class NNDVI(BatchDetector):

    def __init__(self, k_nn: int = 30, sampling_times: int = 500, alpha: float = 0.01):
        super().__init__()
        self.k_nn = k_nn
        self.sampling_times = sampling_times
        self.alpha = alpha

    def update(self, X: np.array, y_true=None, y_pred=None):
        if self._drift_state == "drift":
            self.reset()

        X, _, _ = super()._validate_input(X, None, None)

        super().update(X=X, y_true=None, y_pred=None)
        test_batch = np.array(X)

        nnsp = NNSpacePartitioner(self.k_nn)
        nnsp.build(self.reference_batch, test_batch)
        M_nnps = nnsp.nnps_matrix
        v_ref, v_test = nnsp.v1, nnsp.v2
        d_act = NNSpacePartitioner.compute_nnps_distance(M_nnps, v_ref, v_test)

        theta_drift = self._compute_drift_threshold(
            M_nnps, v_ref, v_test, self.sampling_times, self.alpha
        )
        if d_act > theta_drift:
            self._drift_state = "drift"
            self.set_reference(test_batch)

    def set_reference(self, X, y_true=None, y_pred=None):
        X, _, _ = super()._validate_input(X, None, None)
        self.reference_batch = X

    def reset(self):
        super().reset()

    @staticmethod
    def _compute_drift_threshold(M_nnps, v_ref, v_test, sampling_times, alpha):
        # TODO - Would like to parallelize this - Anmol
        d_shuffle = []
        for _ in range(sampling_times):
            v1_shuffle = np.random.permutation(v_ref)
            v2_shuffle = 1 - v1_shuffle

            d_i_shuffle = NNSpacePartitioner.compute_nnps_distance(
                M_nnps, v1_shuffle, v2_shuffle
            )
            d_shuffle.append(d_i_shuffle)
        mu, std = norm.fit(d_shuffle)
        drift_threshold = norm.ppf(1 - alpha, mu, std)
        return drift_threshold
