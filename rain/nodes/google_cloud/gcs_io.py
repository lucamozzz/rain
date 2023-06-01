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

import os
from google.cloud import storage
from google.cloud.exceptions import NotFound

from typing import Union

import pandas

from rain.core.base import InputNode, OutputNode, Tags, LibTag, TypeTag
from rain.core.parameter import KeyValueParameter, Parameters


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
    destination_blob_name : str
        The ID of the GCS object.
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
        credentials = {"token": os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}
        self.dataset = pandas.read_csv(**param_dict, storage_options=credentials)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.GCS, TypeTag.INPUT)


class GCStorageCSVWriter(OutputNode):
    """Loads a pandas DataFrame into a Google Cloud Storage bucket.

    Input
    ------
    dataset : pandas.DataFrame
        The Pandas dataframe to be uploaded.

    Parameters
    ----------
    bucket_name : str
        The ID of the GCS bucket.
    destination_blob_name : str
        The ID of the GCS object.
    """

    _input_vars = {"dataset": pandas.DataFrame}
    
    def __init__(
        self,
        node_id: str,
        bucket_name: str,
        destination_blob_name: str,
        delim: str = ",",
        include_rows: bool = True,
        rows_column_label: str = None,
        include_columns: bool = True,
        columns: list = None,
    ):
        super(GCStorageCSVWriter, self).__init__(node_id)
        self.parameters = Parameters(
            bucket_name=KeyValueParameter("bucket_name", str, bucket_name, True),
            destination_blob_name =KeyValueParameter("destination_blob_name", str, destination_blob_name , True),
            delim=KeyValueParameter("sep", str, delim),
            include_rows=KeyValueParameter("index", bool, include_rows),
            rows_column_label=KeyValueParameter("index_label", str, rows_column_label),
            include_columns=KeyValueParameter("header", bool, include_columns),
            columns=KeyValueParameter("columns", list, columns),
        )

        self.parameters.group_all("write_csv")

        self.parameters.add_parameter("bucket_name", KeyValueParameter("bucket_name", str, bucket_name, True))
        self.parameters.add_parameter("destination_blob_name", KeyValueParameter("destination_blob_name", str, destination_blob_name, True))

    def execute(self):
        param_dict = self.parameters.get_dict_from_group("write_csv")

        if not isinstance(self.dataset, pandas.DataFrame):
            self.dataset = pandas.DataFrame(self.dataset)

        client = storage.Client()

        try:
            bucket = client.get_bucket(self.parameters.bucket_name.value)
        except NotFound:
            client.create_bucket(self.parameters.bucket_name.value)
            bucket = client.get_bucket(self.parameters.bucket_name.value)

        bucket.blob(self.parameters.destination_blob_name.value).upload_from_string(self.dataset.to_csv(**param_dict), 'text/csv')

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.GCS, TypeTag.OUTPUT)
