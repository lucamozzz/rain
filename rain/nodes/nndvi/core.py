import numpy as np
import pandas
import rain.nodes.nndvi.nndvi_main as nndvi_main

from rain.core.parameter import Parameters, KeyValueParameter
from rain.core.base import ComputationalNode, Tags, LibTag, TypeTag

class RainfallNndvi(ComputationalNode):
    """Runs drift detection with Nndvi algorithm and returns the result.

    Input
    -----
    file : pandas.DataFrame
        The dataset to analyze.

    Output
    ------
    results : pandas.DataFrame
        The result obtained from the algorithm.

    Parameters
    ----------
    node_id : str
        Id of the node.
    """

    _input_vars = {"file": pandas.DataFrame}
    _output_vars = {"results": pandas.DataFrame}
    def __init__(
        self,
        node_id: str
    ):
        super(ComputationalNode, self).__init__(node_id)
        self.parameters = Parameters()

    def execute(self):
        self.results = nndvi_main.run_nndvi(self.file)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.BASE, TypeTag.DRIFT_DETECTOR)
        