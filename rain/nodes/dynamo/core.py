import numpy as np
import pandas
import rain.nodes.dynamo.dynamo_main as dynamo_main

from rain.core.parameter import Parameters, KeyValueParameter
from rain.core.base import ComputationalNode, Tags, LibTag, TypeTag

class RainfallDynamo(ComputationalNode):
    """Runs drift detection with Dynamo algorithm and returns the result.

    Input
    -----
    file : pandas.DataFrame
        The dataset to analyze.

    Output
    ------
    results : dict
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
        self.parameters = Parameters( )

    def execute(self):
        self.results = dynamo_main.run_dynamo(self.file)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.BASE, TypeTag.DRIFT_DETECTOR)
        