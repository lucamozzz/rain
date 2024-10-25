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
from datetime import datetime
from abc import abstractmethod 
from rain.core.base import InputNode, OutputNode, Tags, LibTag, TypeTag
from rain.core.parameter import KeyValueParameter, Parameters
from pymongo import MongoClient
import pandas
import pm4py


MONGODB_URL = os.environ.get("MONGODB_URL")
RAINFALL_DB = 'rainfall'
FILES_COLLECTION = 'files'
FOLDERS_COLLECTION = 'folders'

class Pm4pyInputNode(InputNode):
    """Parent class for all the nodes that load a pandas DataFrame from some kind of source.
    """
    _output_vars = {"dataset": pandas.DataFrame}

    @abstractmethod
    def execute(self):
        pass  # pragma: no cover

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PM4PY, TypeTag.INPUT)


class Pm4pyOutputNode(OutputNode):
    """Parent class for all the nodes that return a pandas DataFrame toward some kind of destination.
    """
    _input_vars = {"dataset": pandas.DataFrame}

    @abstractmethod
    def execute(self):
        pass  # pragma: no cover

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PM4PY, TypeTag.OUTPUT)



class Pm4pyXESLoader(Pm4pyInputNode):
    """Loads a pandas DataFrame from a XES file.

    Output
    ------
    dataset : pandas.DataFrame
        The loaded XES file as a pandas DataFrame.

    Parameters
    ----------
    path : str
        Of the XES file.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.read_csv.html>`_ for Pandas read_csv
    documentation.
    """

    def __init__(self, node_id: str, path: str):
        super(Pm4pyXESLoader, self).__init__(node_id)
        
        if len(sys.argv) > 1:
            path = '/tmp/data/' + sys.argv[1] + '/folders/' + path

        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path, True),
        )

    def execute(self):
        df: pandas.DataFrame = pm4py.read_xes(self.parameters.path.value)
        self.dataset = df


class Pm4pyXESWriter(Pm4pyOutputNode):
    """Writes a pandas DataFrame into a XES file.

    Input
    ------
    dataset : pandas.DataFrame
        The pandas DataFrame to be written as XES file.

    Parameters
    ----------
    path : str
        Of the XES file.
    case_id_key : str
        Column key that identifies the case identifier.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.read_csv.html>`_ for Pandas read_csv
    documentation.
    """

    def __init__(self, node_id: str, path: str, case_id_key: str = "case:concept:name"):
        super(Pm4pyXESWriter, self).__init__(node_id)
        
        if len(sys.argv) > 1:
            path = '/tmp/data/' + sys.argv[1] + '/folders/' + '/'.join(path.split('/')[-2:])

        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path, True),
            case_id_key=KeyValueParameter("case_id_key", str, case_id_key),
        )

    def execute(self):
        pm4py.write_xes(self.dataset, self.parameters.path.value, self.parameters.case_id_key.value)
        
class Pm4pyBPMNWriter(OutputNode):
    """Writes a BPMN model into a local folder.

    Input
    -----
    model : BPMN
        The BPMN model to write in a BPMN file.

    Parameters
    ----------
    path : str
        Of the BPMN file.
    """

    def __init__(
        self,
        node_id: str,
        path: str,
    ):
        super(Pm4pyBPMNWriter, self).__init__(node_id)
        
        if len(sys.argv) > 1:
            path = '/tmp/data/' + sys.argv[1] + '/folders/' + path

        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path, True),
        )
        
    _input_vars = {"model": pm4py.objects.bpmn.obj.BPMN}

    def execute(self):
        pm4py.write_bpmn(self.model, self.parameters.path.value)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PM4PY, TypeTag.OUTPUT)
    