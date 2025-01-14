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

from sklearn.datasets import load_iris
from pandas import DataFrame
from rain.nodes.google_cloud_bigquery.bigquery_io import BigQueryCSVLoader, BigQueryCSVWriter
from tests.test_commons import check_param_not_found


class TestBigQueryCSVLoader:
    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        check_param_not_found(BigQueryCSVLoader, path="", x=8)

    def test_dataset_load(self):
        """Tests the execution of the BigQueryCSVLoader."""
        bq_loader = BigQueryCSVLoader(
            node_id='node1', query='select * from rainfall-e8e57.test.table1'
        )
        assert bq_loader.get_output_value('dataset') == None
        bq_loader.execute()
        assert type(bq_loader.get_output_value('dataset')) == DataFrame


class TestBigQueryCSVWriter:
    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        check_param_not_found(BigQueryCSVWriter, path="", x=8)

    def test_dataset_write(self):
        """Tests the execution of the BigQueryCSVWriter."""
        iris = load_iris(as_frame=True).data

        bq_writer = BigQueryCSVWriter(
            "s1", table_id='rainfall-e8e57.test.table1'
        )
        bq_writer.set_input_value("dataset", iris)
        bq_writer.execute()
