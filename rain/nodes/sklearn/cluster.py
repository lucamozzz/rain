"""
 Copyright (C) 2023 Università degli Studi di Camerino.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta, Luca Mozzoni, Vincenzo Nucci

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

import pandas

from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.sklearn.node_structure import SklearnClusterer
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


class KMeansClusterer(SklearnClusterer):
    """A clusterer for the sklearn KMeans that uses the 'sklearn.cluster.KMeans'.

    Input
    -----
    fit_dataset : pandas.DataFrame
        The dataset that will be used to perform the fit of the clusterer.
    predict_dataset : pandas.DataFrame
        The dataset that will be used to perform the predict of the clusterer.
    score_dataset : pandas.DataFrame
        The dataset that will be used to perform the scoring.
    transform_dataset : pandas.DataFrame
        The dataset that will be used to perform the transform.

    Output
    ------
    fitted_model : sklearn.base.BaseEstimator
        The model that results from the fit of the estimator.
    predictions : pandas.DataFrame
        The predictions that result from the predict.
    score_value : float
        The score value that results from the scoring.
    transformed_dataset : pandas.DataFrame
        The dataset that results from the transform.
    labels : pandas.DataFrame
        Labels of each point.
        It corresponds to the 'labels_' attribute of the sklearn KMeans.

    Parameters
    ----------
    node_id : str
        Id of the node.
    execute : [fit, predict, score, transform]
        List of strings to specify the methods to execute.
        The allowed strings are those from the _method attribute.
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    """

    _output_vars = {"labels": pandas.DataFrame}

    def __init__(self, node_id: str, execute: list, n_clusters: int = 8):
        super(KMeansClusterer, self).__init__(node_id, execute)
        self.parameters = Parameters(
            n_clusters=KeyValueParameter("n_clusters", int, n_clusters)
        )
        self._estimator_or_function = KMeans(**self.parameters.get_dict())

    def execute(self):
        super(KMeansClusterer, self).execute()
        self.labels = self.fitted_model.labels_


class KMedoidsClusterer(SklearnClusterer):
    """A clusterer for the sklearn_extra KMedoids that uses the 'sklearn_extra.cluster.KMedoids'.

    Input
    -----
    fit_dataset : pandas.DataFrame
        The dataset that will be used to perform the fit of the clusterer.
    predict_dataset : pandas.DataFrame
        The dataset that will be used to perform the predict of the clusterer.
    transform_dataset : pandas.DataFrame
        The dataset that will be used to perform the transform.

    Output
    ------
    fitted_model : sklearn.base.BaseEstimator
        The model that results from the fit of the estimator.
    predictions : pandas.DataFrame
        The predictions that result from the predict.
    transformed_dataset : pandas.DataFrame
        The dataset that results from the transform.
    labels : pandas.DataFrame
        Labels of each point.
        It corresponds to the 'labels_' attribute of the sklearn KMeans.
    medoids : pandas.DataFrame
        The indices of the medoid rows in X, if 'precomputed' == true,
        cluster centers, i.e. medoids, otherwise.

    Parameters
    ----------
    node_id : str
        Id of the node.
    execute : [fit, predict, transform]
        List of strings to specify the methods to execute.
        The allowed strings are those from the _method attribute.
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    precomputed : bool
        If set to True, the user must then feed the clusterer with a precomputed kernel matrix.
    """

    _output_vars = {"labels": pandas.DataFrame, "medoids": pandas.DataFrame}

    def __init__(self, node_id: str, execute: list, n_clusters: int = 8, precomputed: bool = False):
        super(KMedoidsClusterer, self).__init__(node_id, execute)
        if precomputed: metric = "precomputed" 
        else: metric = "euclidean"
        self.parameters = Parameters(
            n_clusters=KeyValueParameter("n_clusters", int, n_clusters),
            metric=KeyValueParameter("metric", str, metric),
        )
        self._estimator_or_function = KMedoids(**self.parameters.get_dict())

    def execute(self):
        super(KMedoidsClusterer, self).execute()
        # self.labels = self.fitted_model.labels_
        self.labels = pandas.DataFrame(self.fitted_model.labels_)
        if self.parameters.metric.value == "precomputed":
            self.medoids = pandas.DataFrame(self.fitted_model.medoid_indices_)
        else:
            self.medoids = pandas.DataFrame(self.fitted_model.cluster_centers_)
