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

from google.cloud.bigquery import Client
from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError
import pandas

from rain.core.base import InputNode, OutputNode, Tags, LibTag, TypeTag
from rain.core.parameter import KeyValueParameter, Parameters
from os import getenv

class BigQueryCSVLoader(InputNode):
    """Runs a SELECT on a BigQuery table and saves the result as a Pandas dataframe.

    Output
    ------
    dataset : pandas.DataFrame
        The result obtained from running the query.

    Parameters
    ----------
    node_id : str
        Id of the node.
    query : str
        The query to be run.
    """

    _output_vars = {"dataset": pandas.DataFrame}

    def __init__(self, node_id: str, query: str):
        super(BigQueryCSVLoader, self).__init__(node_id)

        self.parameters = Parameters(
            query=KeyValueParameter("Query", str, query, True),
        )

    def execute(self):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = Client(credentials=credentials)
        except:
            raise DefaultCredentialsError('Missing credentials')

        query: str = self.parameters.query.value
        if query.strip().lower().split()[0] == "select":
            self.dataset: pandas.DataFrame = client.query(query).result().to_dataframe()
        else:
            raise ValueError("Query must start with select")

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.BIGQUERY, TypeTag.INPUT)


class BigQueryCSVWriter(OutputNode):
    """Upload the contents of a pandas DataFrame to a BigQuery table.

    Input
    ------
    dataset : pandas.DataFrame
        The dataframe to upload as a table.

    Parameters
    ----------
    node_id : str
        Id of the node.
    table_id : str
        The table Id.
    """

    _input_vars = {"dataset": pandas.DataFrame}

    def __init__(self, node_id: str, table_id: str):
        super(BigQueryCSVWriter, self).__init__(node_id)

        self.parameters = Parameters(
            table_id=KeyValueParameter("Table Id", str, table_id, True),
        )

    def execute(self):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = Client(credentials=credentials)
        except:
            raise DefaultCredentialsError('Missing credentials')

        # removes unsupported chars from column names
        def replace_chars(columns):
            new_columns = []
            to_replace = [' ', '(', ')', '[', ']', '{', '}', '!', '@', '~', '\\', '$', '`', '^', '.', ',', '*']
            for column in columns:
                for char in to_replace:
                    column: str = column.replace(char, '_')
                new_columns.append(column)
            return new_columns

        if not isinstance(self.dataset, pandas.DataFrame):
            self.dataset = pandas.DataFrame(self.dataset)
        self.dataset.columns = replace_chars(self.dataset.columns)

        job = client.load_table_from_dataframe(self.dataset, self.parameters.table_id.value)
        job.result()

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.BIGQUERY, TypeTag.OUTPUT)
