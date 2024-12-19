import numpy as np
from sklearn.neighbors import NearestNeighbors


class NNSpacePartitioner:
    def __init__(self, k: int):
        self.k = k
        self.D = None
        self.v1 = None
        self.v2 = None
        self.nnps_matrix = None
        self.adjacency_matrix = None

    def build(self, sample1: np.array, sample2: np.array):

        data = np.vstack((sample1, sample2))
        D, inverted_indices = np.unique(data, axis=0, return_inverse=True)
        self.D = D
        v1, v2 = np.array_split(inverted_indices, 2)
        v1_onehot = np.zeros(D.shape[0])
        v2_onehot = np.zeros(D.shape[0])
        # XXX - Alternatively, v1_onehot = np.identity(adjacency_matrix.shape[0])[v1] - Anmol
        v1_onehot[v1] = 1.0
        v2_onehot[v2] = 1.0
        self.v1 = v1_onehot
        self.v2 = v2_onehot
        nn = NearestNeighbors(n_neighbors=self.k).fit(D)
        # TODO: maybe we can gain performance by performing operations using the returned
        # scipy.sparse array, as opposed to converting this way.
        M_adj = nn.kneighbors_graph(D).toarray()
        self.adjacency_matrix = M_adj
        # XXX - NearestNeighbors already adds the self-neighbors
        # TODO - check about order preservation
        P_nnps = M_adj
        weight_array = np.sum(P_nnps, axis=1).astype(int)
        Q = np.lcm.reduce(weight_array)
        m = Q / weight_array
        m = m * np.identity(len(m))
        self.nnps_matrix = np.matmul(m, P_nnps)

    @staticmethod
    def compute_nnps_distance(nnps_matrix, v1, v2):
        M_s1 = np.dot(v1, nnps_matrix)
        M_s2 = np.dot(v2, nnps_matrix)

        # These commented lines would only be relevant if there were overlap
        # between the two vectors, which there never should be for our use case.
        # Otherwise, this is always going to be the number of elements.
        # membership = np.sum(np.array([v1, v2]), axis=0)
        # membership = membership >= 1  # in case of overlap
        # denom = sum(membership)
        denom = len(v1)

        d_nnps = np.sum(np.abs(M_s1 - M_s2) / (M_s1 + M_s2))
        d_nnps /= denom
        return d_nnps
