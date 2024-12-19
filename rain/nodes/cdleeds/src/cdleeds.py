import copy
import itertools
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import ttest_ind, combine_pvalues
from sklearn.metrics.pairwise import rbf_kernel
from typing import Any, Optional, Tuple, List


class CDLEEDS:
    def __init__(self,
                 significance: float = 0.01,
                 gamma: float = 0.95,
                 max_node_size: int = 200,
                 max_tree_depth: int = 5,
                 max_time_stationary: int = 100):
        self.significance = significance
        self.id_generator = itertools.count()
        self.root = Node(node_id=next(self.id_generator),
                         gamma=gamma,
                         max_node_size=max_node_size,
                         max_tree_depth=max_tree_depth,
                         max_time_stationary=max_time_stationary,
                         depth=0)
        self.monitored_sample = dict()
        self._baseline = None

    def set_baseline(self, baseline: ArrayLike):
        self._baseline = baseline

    def partial_fit(self, X: ArrayLike, y_pred: ArrayLike):
        if self._baseline is None:
            raise AttributeError('No CDLEEDS baseline value has been specified. '
                                 'Use set_baseline() to set or update the baseline.')
        elif self._baseline.shape != y_pred.shape:
            raise ValueError('CDLEEDS baseline {} and y_pred {} have different shapes but should have the same.'.format(
                self._baseline.shape, y_pred.shape))
        else:
            # Start recursive update at the root node.
            self.root.update(X=X, pred_baseline_diff=y_pred - self._baseline, id_generator=self.id_generator)

    def detect_local_change(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        leaves = self._get_leaf_info(node=self.root, leaves=dict())
        leaf_ids = np.asarray(list(leaves.keys()))
        centroids = np.asarray([leaf['centroid'] for leaf in leaves.values()])
        p_values = np.asarray([leaf['p_value'] for leaf in leaves.values()])
        local_drifts = np.sum(p_values <= self.significance, axis=1) > 0

        # Check for change in the monitored sample.
        changing_monitored_samples = []
        if self.monitored_sample:
            # Get old neighborhoods of each observation in the sample.
            sample = np.asarray([s['x'] for s in self.monitored_sample.values()])
            old_neighborhoods = np.asarray([s['neighborhood'] for s in self.monitored_sample.values()])

            # Identify new neighborhood of each observation in the sample.
            sim_obs_centroids = rbf_kernel(sample, centroids)
            leaf_indices = np.argmax(sim_obs_centroids, axis=1)
            new_neighborhoods = leaf_ids[leaf_indices]

            # Update neighborhoods that have changed.
            update_keys = [list(self.monitored_sample.keys())[i] for i in np.argwhere(
                new_neighborhoods != old_neighborhoods).flatten()]
            if len(update_keys) > 0:
                update_values = [dict(zip(['x', 'neighborhood'], (x, n))) for x, n in zip(sample[new_neighborhoods != old_neighborhoods],
                                                                                          new_neighborhoods[new_neighborhoods != old_neighborhoods])]
                update_dict = dict(zip(update_keys, update_values))
                self.monitored_sample = {**self.monitored_sample, **update_dict}

            # Identify monitored observations that have been assigned to a new or temporally changing neighborhood.
            changed_neighbors = (new_neighborhoods != old_neighborhoods) & (old_neighborhoods != None)
            changing_monitored_samples = sample[changed_neighbors].tolist()
            changing_monitored_samples.extend(sample[local_drifts[leaf_indices] & ~changed_neighbors])

        return local_drifts, centroids, np.asarray(changing_monitored_samples)

    def detect_global_change(self) -> bool:
        leaves = self._get_leaf_info(node=self.root, leaves=dict())
        p_values = np.asarray([leaf['p_value'] for leaf in leaves.values()])

        combined_p_value = np.ones(p_values.shape[1])
        fdr_controlled_alpha = np.ones(p_values.shape[1]) * self.significance

        for class_col in range(p_values.shape[1]):
            p_col = p_values[:, class_col][~np.isnan(p_values[:, class_col])]
            p_col[p_col == 0] = 1e-10  # if p-value at a leaf is 0 then set it to a very small positive number instead.

            if len(p_col) > 0:
                _, combined_p_value[class_col] = combine_pvalues(p_col)  # Combine p-values with Fisher's method.
                # Compute mean FDR alpha (alpha adjusted for k independent tests)
                fdr_controlled_alpha[class_col] = self.significance * (len(p_col) + 1) / (2 * len(p_col))

        if any(combined_p_value <= fdr_controlled_alpha):
            return True
        else:
            return False

    def add_to_monitored_sample(self, X: ArrayLike):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        for x in X:
            if x.tobytes() not in self.monitored_sample:
                self.monitored_sample[x.tobytes()] = dict()
                self.monitored_sample[x.tobytes()]['x'] = x
                self.monitored_sample[x.tobytes()]['neighborhood'] = None

    def delete_from_monitored_sample(self, X: ArrayLike):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        for x in X:
            if x.tobytes() in self.monitored_sample:
                del self.monitored_sample[x.tobytes()]

    def _get_leaf_info(self, node: Any, leaves: dict) -> dict:
        if node.is_leaf:
            leaves[node.id] = dict()
            leaves[node.id]['centroid'] = copy.copy(node.centroid)
            leaves[node.id]['p_value'] = node.p_value
        else:
            for child in node.children:
                leaves = self._get_leaf_info(node=child, leaves=leaves)
        return leaves


class Node:
    def __init__(self,
                 node_id: int,
                 gamma: float,
                 max_node_size: int,
                 max_tree_depth: int,
                 max_time_stationary: int,
                 depth: int):
        self.id = node_id
        self.gamma = gamma
        self.max_node_size = max_node_size
        self.max_tree_depth = max_tree_depth
        self.max_time_stationary = max_time_stationary
        self.depth = depth

        self.is_leaf = True
        self.p_value = None
        self.age = 0
        self.X_window = None
        self.centroid = None
        self.pred_baseline_diff_window = None
        self.children = []

    def update(self, X: ArrayLike, pred_baseline_diff: ArrayLike, id_generator: Any):
        if self.p_value is None:  # Set up vector of p-values (per class)
            self.p_value = np.asarray([np.nan] * pred_baseline_diff.shape[1])

        # ----------------------------
        # Update the node statistics.
        # ----------------------------
        self.age += 1
        self.X_window = self._update_window(window=self.X_window, data=X, window_size=self.max_node_size)
        self.centroid = np.mean(self.X_window, axis=0)
        self.pred_baseline_diff_window = self._update_window(window=self.pred_baseline_diff_window,
                                                             data=pred_baseline_diff,
                                                             window_size=self.max_node_size)

        if self.is_leaf:
            # Check where the minimum similarity from the centroid is violated.
            similarities = rbf_kernel(self.X_window, self.centroid.reshape(1, -1))
            violations = similarities < self.gamma

            if np.count_nonzero(violations) > 0 and self.depth + 1 <= self.max_tree_depth:
                # ----------------------------
                # Split the leaf node.
                # ----------------------------
                self.is_leaf = False

                # Assign all observations in the sliding window to one of the two most dissimilar points (centroids).
                proximity_matrix = rbf_kernel(self.X_window, self.X_window)
                child_centroids = np.unravel_index(proximity_matrix.argmin(), proximity_matrix.shape)
                clusters = self._cluster(X=self.X_window,
                                         left_centroid=self.X_window[child_centroids[0]],
                                         right_centroid=self.X_window[child_centroids[1]])

                for child_i in range(2):
                    self.children.append(Node(node_id=next(id_generator),
                                              gamma=self.gamma,
                                              max_node_size=self.max_node_size,
                                              max_tree_depth=self.max_tree_depth,
                                              max_time_stationary=self.max_time_stationary,
                                              depth=self.depth + 1))
                    self.children[-1].age = self.age - 1  # Note that the age is incremented again in the following update() call.
                    self.children[-1].update(X=self.X_window[clusters == child_i],
                                             pred_baseline_diff=self.pred_baseline_diff_window[clusters == child_i],
                                             id_generator=id_generator)
            else:
                # ----------------------------
                # Update p-values.
                # ----------------------------
                if self.pred_baseline_diff_window.shape[0] >= 4:  # We require at least 4 observations to provide two disjoint samples to the t-test.
                    self.p_value = self._update_p_value(X=self.pred_baseline_diff_window)

        else:  # Inner Node
            # ----------------------------
            # Forward observations to children.
            # ----------------------------
            clusters = self._cluster(X=X,
                                     left_centroid=self.children[0].centroid.reshape(1, -1),
                                     right_centroid=self.children[1].centroid.reshape(1, -1))

            for child_i in np.unique(clusters):
                self.children[child_i].age = self.age - 1  # Note that the age is incremented again in the following update() call.
                self.children[child_i].update(X=X[clusters == child_i],
                                              pred_baseline_diff=pred_baseline_diff[clusters == child_i],
                                              id_generator=id_generator)

            min_child_age = min(self.children[0].age, self.children[1].age)
            if self.age - min_child_age >= self.max_time_stationary:
                # ----------------------------
                # Prune the inner node and update p-values.
                # ----------------------------
                self.is_leaf = True
                self.children = []
                if self.pred_baseline_diff_window.shape[0] >= 4:  # We require at least 4 observations to provide two disjoint samples to the t-test.
                    self.p_value = self._update_p_value(X=self.pred_baseline_diff_window)

    @staticmethod
    def _update_window(window: Optional[ArrayLike], data: ArrayLike, window_size: int) -> ArrayLike:
        if data.ndim == 1:
            data = data.reshape(1, -1)

        for data_point in data:
            data_point = data_point.reshape(1, -1)
            if window is None:
                window = data_point
            else:
                if window.shape[0] < window_size:
                    window = np.append(window, data_point, axis=0)
                else:
                    window = window[1:]
                    window = np.append(window, data_point, axis=0)
        return window

    @staticmethod
    def _cluster(X: ArrayLike, left_centroid: ArrayLike, right_centroid: ArrayLike) -> ArrayLike:
        sim_left = rbf_kernel(X, left_centroid.reshape(1, -1))
        sim_right = rbf_kernel(X, right_centroid.reshape(1, -1))
        sim = np.append(sim_left.reshape(-1, 1), sim_right.reshape(-1, 1), axis=1)
        return np.argmax(sim, axis=1)

    @staticmethod
    def _update_p_value(X: ArrayLike) -> ArrayLike:
        idx = round(X.shape[0] / 2)  # We compare two-equally sized samples
        _, p_val = ttest_ind(X[:idx], X[idx:], nan_policy='raise')
        return p_val