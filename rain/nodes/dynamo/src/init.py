import os
import random
from itertools import count
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from rain.nodes.dynamo.src.pyclee.clusters import Cluster
from rain.nodes.dynamo.src.pyclee.dyclee import DyClee, DyCleeContext
from rain.nodes.dynamo.src.pyclee.forgetting import ExponentialForgettingMethod, ForgettingMethod
from rain.nodes.dynamo.src.pyclee.types import Element, Timestamp
from rain.nodes.dynamo.src.tests import (DensestHyperboxDifference, DivergenceMetric, MeanDivergence)
from rain.nodes.dynamo.src.dynamo import DynAmo
from rain.nodes.dynamo.src.trackers import (BoxSizeProductTracker, BoxSizeTracker, DifferenceBoxTracker, NormalizedBoxSizeTracker, NormalizedDifferenceBoxTracker, Tracker)
from rain.nodes.dynamo.src.consensus_functions import (Consensus, MajorityVoting)
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# Seed value
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def run_dyclee(
    signal: np.array,
    hyperbox_fraction: float=.2,
    forgetting_method: ForgettingMethod = None) -> pd.DataFrame:
    
    def run_internal(
        dy: DyClee,
        elements: Iterable[Element],
        times: Optional[Iterable[Timestamp]] = None,
        progress: bool = True) -> List[Cluster]:
        
        if progress and tqdm is not None:
            elements = tqdm(elements)
        
        if times is None:
            times = count()
        
        for element, time in zip(elements, times):
            _, clusters, _, _ = dy.step(element, time, skip_density_step=False)
            all_clusters.append(clusters)
            
        dy.density_step(time)
        
        return all_clusters

    all_clusters: list[Cluster] = list()
    centroids: list[np.ndarray] = list()
    bounds = np.column_stack((signal.min(axis=0), signal.max(axis=0)))
    
    context = DyCleeContext(
        n_features=signal.shape[1],
        hyperbox_fractions=hyperbox_fraction,
        feature_ranges=bounds,
        forgetting_method=forgetting_method)

    dy = DyClee(context)

    all_clusters = run_internal(dy, signal)
    all_clusters = list(filter(lambda item: item is not None, all_clusters))


    for step in all_clusters:
        max_dense = 0
        for cluster in step:
            micros = cluster.μclusters
            density = sum([micro.n_elements for micro in micros])

            if density >= max_dense:

                max_dense = density
                centroid = cluster.centroid

        centroids.append(centroid)

    return np.array(centroids)
    
def create_config(data_size, threshold_size) -> Dict[str, object]:
    res: Dict[str, object] = dict()

    consensus_instance = MajorityVoting()

    trackers_list: List[Tracker] = list()
    trackers = [
            BoxSizeTracker,
            BoxSizeProductTracker,
            NormalizedBoxSizeTracker,
            DifferenceBoxTracker,
            NormalizedDifferenceBoxTracker
        ]
    for tracker in trackers:
        trackers_list.append(tracker())
            

    divergence_metrics_list: List[DivergenceMetric] = list()
    divergence_metrics = [
        DensestHyperboxDifference,
        MeanDivergence
    ]
    for divergence_metric in divergence_metrics:
        divergence_metrics_list.append(divergence_metric())
    
    lookup_size = 0
    drift_detection_threshold = 0
    wnd_moving_step = 0
    limit_per_window = 0

    if data_size > threshold_size:
        lookup_size = 25
        drift_detection_threshold = 0.2666
        wnd_moving_step = 10
        limit_per_window = 30
    else:
        lookup_size = 4
        drift_detection_threshold = 0.3422
        wnd_moving_step = 4
        limit_per_window = 17

    dynamo = DynAmo(
        signal=None,
        trackers=trackers_list,
        divergence_metrics=divergence_metrics_list,
        consensus_func=consensus_instance,
        lookup_size=lookup_size,
        wnd_moving_step=wnd_moving_step,
        drift_detection_threshold=drift_detection_threshold,
        limit_per_window=limit_per_window
    )

    res['dynamo'] = dynamo
    res['start_offset'] = 0
    res['hyperbox_fraction'] = 0.2
    res['forgetting_instance'] = ExponentialForgettingMethod(.02)

    return res