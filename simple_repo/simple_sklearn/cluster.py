from simple_repo.parameter import KeyValueParameter, Parameters
from simple_repo.simple_sklearn.node_structure import SklearnClusterer
from sklearn.cluster import KMeans


class SimpleKMeans(SklearnClusterer):
    """A clusterer for the sklearn KMeans.

    Parameters
    ----------
    execute : list[str]
        Methods to execute with this clusterer, they can be: fit, predict, transform, score.
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    """

    def __init__(self, node_id: str, execute: list, n_clusters: int = 8):
        self.parameters = Parameters(
            n_clusters=KeyValueParameter("n_clusters", int, n_clusters)
        )
        super(SimpleKMeans, self).__init__(node_id, KMeans, execute)
