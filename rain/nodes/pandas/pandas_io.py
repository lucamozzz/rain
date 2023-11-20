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


from os import getenv
# from google.cloud.storage import Client
# from google.cloud.exceptions import NotFound
# from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError
from abc import abstractmethod
from typing import Union
import uuid
import os
import pandas
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from io import StringIO
from rain.core.base import InputNode, OutputNode, Tags, LibTag, TypeTag
from rain.core.parameter import KeyValueParameter, Parameters

MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27017/")
RAINFALL_DB = 'rainfall'
FILES_COLLECTION = 'files'
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
    file : str
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

    # _parameters = { "filepath_or_buffer": PandasParameter("filepath_or_buffer", str, is_mandatory=True),
    # sep=<no_default>, delimiter=None, header='infer', names=<no_default>, index_col=None, usecols=None,
    # squeeze=False, prefix=<no_default>, mangle_dupe_cols=True, dtype=None, engine=None, converters=None,
    # true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None,
    # na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False,
    # infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True,
    # iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None,
    # quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None, encoding=None,
    # encoding_errors='strict', dialect=None, error_bad_lines=None, warn_bad_lines=None, on_bad_lines=None,
    # delim_whitespace=False, low_memory=True, memory_map=False, float_precision=None, storage_options=None }

    def __init__(self, node_id: str, file: str, delim: str = ",", index_col: Union[int, str] = None):
        super(PandasCSVLoader, self).__init__(node_id)

        self.parameters = Parameters(
            file=KeyValueParameter("filepath_or_buffer", str, file),
            delim=KeyValueParameter("delimiter", str, delim),
            index_col=KeyValueParameter("index_col", str, index_col)
        )

        self.parameters.group_all("read_csv")

    def execute(self):
        client = MongoClient(MONGODB_URL)
        db = client[RAINFALL_DB]
        collection = db[FILES_COLLECTION]
        document = collection.find_one({'_id': self.parameters.file.value})
        csv_content = document['content']
        df = pd.read_csv(StringIO(csv_content))
        self.dataset = df


class PandasCSVWriter(PandasOutputNode):
    """Writes a pandas DataFrame into a CSV file.

    Input
    -----
    dataset : pandas.DataFrame
        The pandas DataFrame to write in a CSV file.

    Parameters
    ----------
    name : str
        Of the CSV file.
    folder : str
        For storing the CSV file.
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

    # _parameters = { "filepath_or_buffer": PandasParameter("filepath_or_buffer", str, is_mandatory=True),
    # sep=<no_default>, delimiter=None, header='infer', names=<no_default>, index_col=None, usecols=None,
    # squeeze=False, prefix=<no_default>, mangle_dupe_cols=True, dtype=None, engine=None, converters=None,
    # true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None,
    # na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False,
    # infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True,
    # iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None,
    # quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None, encoding=None,
    # encoding_errors='strict', dialect=None, error_bad_lines=None, warn_bad_lines=None, on_bad_lines=None,
    # delim_whitespace=False, low_memory=True, memory_map=False, float_precision=None, storage_options=None }

    def __init__(
        self,
        node_id: str,
        folder: str,
        name: str,
        delim: str = ",",
        include_rows: bool = True,
        rows_column_label: str = None,
        include_columns: bool = True,
        columns: list = None,
    ):
        super(PandasCSVWriter, self).__init__(node_id)
        self.parameters = Parameters(
            name=KeyValueParameter("name", str, name),
            folder=KeyValueParameter("folder", str, folder),
            delim=KeyValueParameter("sep", str, delim),
            include_rows=KeyValueParameter("index", bool, include_rows),
            rows_column_label=KeyValueParameter("index_label", str, rows_column_label),
            include_columns=KeyValueParameter("header", bool, include_columns),
            columns=KeyValueParameter("columns", list, columns),
        )

        self.parameters.group_all("write_csv")

    def execute(self):
        client = MongoClient(MONGODB_URL)
        db = client[RAINFALL_DB]
        collection = db[FILES_COLLECTION]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_id = 'file-' + str(uuid.uuid4())
        csv_buffer = StringIO()
        self.dataset.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        file = {
            "_id": file_id,
            "created_at": current_time,
            "name": self.parameters.name.value,
            "content": csv_string,
            "folder": self.parameters.folder.value
        }
        collection.insert_one(file)
        collection = db[FOLDERS_COLLECTION]
        collection.update_one(
            {"_id": self.parameters.folder.value},
            {"$push": {"files": file_id}}
        )


class GCStorageCSVLoader(InputNode):
    """Loads a pandas DataFrame from a CSV stored in a Google Cloud Storage bucket.

    Output
    ------
    dataset : pandas.DataFrame
        The loaded CSV file as a pandas DataFrame.

    Parameters
    ----------
    bucket_name : str
        The ID of the GCS bucket.
    object_path : str
        The path of the GCS object.
    delim : str, default ','
        Delimiter symbol of the CSV file.
    index_col : str, default=None
        Column to use as the row labels of the DataFrame, given as string name.
    """

    _output_vars = {"dataset": pandas.DataFrame}

    def __init__(self, node_id: str, bucket_name: str, object_path: str, delim: str = ",", index_col: Union[int, str] = None):
        super(GCStorageCSVLoader, self).__init__(node_id)

        self.parameters = Parameters(
            path=KeyValueParameter("filepath_or_buffer", str, 'gcs://' + bucket_name + '/' + object_path),
            delim=KeyValueParameter("delimiter", str, delim),
            index_col=KeyValueParameter("index_col", str, index_col)
        )

        self.parameters.group_all("read_csv")

    def execute(self):
        param_dict = self.parameters.get_dict_from_group("read_csv")
        credentials = getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials:
            raise DefaultCredentialsError('Missing credentials')
        self.dataset = pandas.read_csv(**param_dict, storage_options={"token": credentials})

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.GCS, TypeTag.INPUT)