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

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.spark.node_structure import SparkInputNode


class SparkCSVLoader(SparkInputNode):
    """Loads a CSV file as a Spark DataFrame.

    Output
    ------
    dataset : DataFrame
        The loaded Spark DataFrame.

    Parameters
    ----------
    node_id : str
        Id of the node.
    path : str
        Path of the csv file.
    header : bool, default False
        Uses the first line as names of columns.
    schema : bool, default False
        Infers the input schema automatically from data. It requires one extra
        pass over the data.
    """

    _output_vars = {"dataset": DataFrame}

    def __init__(
        self, node_id: str, path: str, header: bool = False, schema: bool = False
    ):
        super(SparkCSVLoader, self).__init__(node_id)
        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path),
            header=KeyValueParameter("header", bool, header),
            schema=KeyValueParameter("inferSchema", bool, schema),
        )

    def execute(self):
        self.dataset = self.spark.read.csv(**self.parameters.get_dict())


class SparkModelLoader(SparkInputNode):
    """Loads a file as a Spark Model.

    Output
    ------
    model : PipelineModel
        The loaded Spark PipelineModel.

    Parameters
    ----------
    node_id : str
        Id of the node.
    path : str
        Path of the csv file.
    """

    _output_vars = {"model": PipelineModel}

    def __init__(self, node_id: str, path: str):
        self.parameters = Parameters(path=KeyValueParameter("path", str, path))
        super(SparkModelLoader, self).__init__(node_id)

    def execute(self):
        self.model = PipelineModel.load(self.parameters.path.value)
