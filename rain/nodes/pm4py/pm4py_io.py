"""
 Copyright (C) 2023 Università degli Studi di Camerino and Sigma S.p.A.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

from abc import abstractmethod
import uuid
import os
import pm4py
import pandas
from datetime import datetime
from pymongo import MongoClient
from rain.core.base import InputNode, OutputNode, Tags, LibTag, TypeTag
from rain.core.parameter import KeyValueParameter, Parameters

MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://mongo:27017/")
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
        The loaded xes file as a pandas DataFrame.

    Parameters
    ----------
    file : str
        ID of the XES file.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.read_csv.html>`_ for Pandas read_csv
    documentation.
    """

    def __init__(self, node_id: str, file: str):
        super(Pm4pyXESLoader, self).__init__(node_id)

        self.parameters = Parameters(
            file=KeyValueParameter("filepath_or_buffer", str, file, True),
        )

    def execute(self):
        client = MongoClient(MONGODB_URL)
        db = client[RAINFALL_DB]
        collection = db[FILES_COLLECTION]
        document = collection.find_one({'_id': self.parameters.file.value})
        xes_content = document['content']

        file_path = "./" + self.parameters.file.value + ".xes"
        with open(file_path, 'w') as xes_file:
            xes_file.write(xes_content)

        df: pandas.DataFrame = pm4py.read_xes(file_path)
        self.dataset = df

        if os.path.exists(file_path):
            os.remove(file_path)


class Pm4pyXESWriter(Pm4pyOutputNode):
    """Writes a pandas DataFrame into a XES file.

    Input
    -----
    dataset : pandas.DataFrame
        The pandas DataFrame to write in a XES file.

    Parameters
    ----------
    folder : str
        Folder where the XES file will be stored.
    name : str
        Of the XES file.
    case_id_key : str
        Column key that identifies the case identifier.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.DataFrame.to_csv.html>`_ for Pandas
    to_csv documentation.
    """

    def __init__(
        self,
        node_id: str,
        folder: str,
        name: str = "result.xes",
        case_id_key: str = "case:concept:name"
    ):
        super(Pm4pyXESWriter, self).__init__(node_id)
        self.parameters = Parameters(
            folder=KeyValueParameter("folder", str, folder, True),
            name=KeyValueParameter("name", str, name),
            case_id_key=KeyValueParameter("case_id_key", str, case_id_key),
        )

    def execute(self):
        client = MongoClient(MONGODB_URL)
        db = client[RAINFALL_DB]
        collection = db[FILES_COLLECTION]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_id = 'file-' + str(uuid.uuid4())
        file_path = "./" + file_id + ".xes"
        pm4py.write_xes(self.dataset, file_path, self.parameters.case_id_key.value)
        with open(file_path, 'r') as xes_file:
            file_contents = xes_file.read()

        if os.path.exists(file_path):
            os.remove(file_path)

        file = {
            "_id": file_id,
            "created_at": current_time,
            "name": self.parameters.name.value,
            "content": file_contents,
            "folder": self.parameters.folder.value
        }
        collection.insert_one(file)
        collection = db[FOLDERS_COLLECTION]
        collection.update_one(
            {"_id": self.parameters.folder.value},
            {"$push": {"files": file_id}}
        )