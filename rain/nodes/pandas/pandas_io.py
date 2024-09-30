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

import os
import sys
import uuid
from io import StringIO
from abc import abstractmethod
from typing import Union
from datetime import datetime
from rain.core.base import InputNode, OutputNode, Tags, LibTag, TypeTag
from rain.core.parameter import KeyValueParameter, Parameters
from pymongo import MongoClient
import pandas


MONGODB_URL = os.environ.get("MONGODB_URL")
RAINFALL_DB = 'rainfall'
FILES_COLLECTION = 'files'
EXECUTIONS_COLLECTION = 'executions'
VISUALS_COLLECTION = 'visuals'
FOLDERS_COLLECTION = 'folders'


class PandasInputNode(InputNode):
    """Parent class for all the nodes that load a pandas DataFrame from some kind of source.
    """
    _output_vars = {"dataset": pandas.DataFrame}

    @abstractmethod
    def execute(self):
        pass  # pragma: no cover

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.INPUT)


class PandasOutputNode(OutputNode):
    """Parent class for all the nodes that return a pandas DataFrame toward some kind of destination.
    """
    _input_vars = {"dataset": pandas.DataFrame}

    @abstractmethod
    def execute(self):
        pass  # pragma: no cover

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.OUTPUT)


class PandasCSVLoader(PandasInputNode):
    """Loads a pandas DataFrame from a CSV file.

    Output
    ------
    dataset : pandas.DataFrame
        The loaded csv file as a pandas DataFrame.

    Parameters
    ----------
    path : str
        Of the CSV file.
    delim : str, default ','
        Delimiter symbol of the CSV file.
    index_col : str, default=None
        Column to use as the row labels of the DataFrame, given as string name

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.read_csv.html>`_ for Pandas read_csv
    documentation.
    """

    def __init__(self, node_id: str, path: str, delim: str = ",", index_col: Union[int, str] = None):
        super(PandasCSVLoader, self).__init__(node_id)
        
        if os.getenv('ISSUED_BY') is not None:
            path = '/tmp/data/' + os.getenv('ISSUED_BY') + '/' + path

        self.parameters = Parameters(
            path=KeyValueParameter("filepath_or_buffer", str, path),
            delim=KeyValueParameter("delimiter", str, delim),
            index_col=KeyValueParameter("index_col", str, index_col)
        )

        self.parameters.group_all("read_csv")

    def execute(self):
        param_dict = self.parameters.get_dict_from_group("read_csv")
        self.dataset = pandas.read_csv(**param_dict)


class PandasCSVWriter(PandasOutputNode):
    """Writes a pandas DataFrame into a CSV file.

    Input
    -----
    dataset : pandas.DataFrame
        The pandas DataFrame to write in a CSV file.

    Parameters
    ----------
    path : str
        Of the CSV file.
    delim : str, default ','
        Delimiter symbol of the CSV file.
    include_rows : bool, default True
        Whether to include rows indexes.
    rows_column_label : str, default None
        If rows indexes must be included you can give a name to its column.
    include_columns : bool, default True
        Whether to include column names.
    columns : list[str], default None
        If column names must be included you can give names to them.
        The order is relevant.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.DataFrame.to_csv.html>`_ for Pandas
    to_csv documentation.
    """

    def __init__(
        self,
        node_id: str,
        path: str,
        delim: str = ",",
        include_rows: bool = True,
        rows_column_label: str = None,
        include_columns: bool = True,
        columns: list = None,
    ):
        super(PandasCSVWriter, self).__init__(node_id)
        
        if os.getenv('ISSUED_BY') is not None:
            path = '/tmp/data/' + os.getenv('ISSUED_BY') + '/' + '/'.join(path.split('/')[-2:])

        self.parameters = Parameters(
            path=KeyValueParameter("path_or_buf", str, path),
            delim=KeyValueParameter("sep", str, delim),
            include_rows=KeyValueParameter("index", bool, include_rows),
            rows_column_label=KeyValueParameter("index_label", str, rows_column_label),
            include_columns=KeyValueParameter("header", bool, include_columns),
            columns=KeyValueParameter("columns", list, columns),
        )

        self.parameters.group_all("write_csv")

    def execute(self):
        param_dict = self.parameters.get_dict_from_group("write_csv")

        if not isinstance(self.dataset, pandas.DataFrame):
            self.dataset = pandas.DataFrame(self.dataset)

        self.dataset.to_csv(**param_dict)
