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

from sklearn.datasets import load_iris
from pandas import DataFrame
from rain.nodes.google_cloud.gcs_io import GCStorageCSVLoader, GCStorageCSVWriter
from tests.test_commons import check_param_not_found

class TestGCStorageCSVLoader:
    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        check_param_not_found(GCStorageCSVLoader, path="", x=8)

    def test_dataset_load(self):
        """Tests the execution of the PandasCSVLoader."""
        gcstorage_loader = GCStorageCSVLoader(
            "s1", bucket_name="rainfall-e8e57.appspot.com", object_path="test.csv"
        )
        assert gcstorage_loader.get_output_value('dataset') == None
        gcstorage_loader.execute()
        assert type(gcstorage_loader.get_output_value('dataset')) == DataFrame


class TestGCStorageCSVWriter:
    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        check_param_not_found(GCStorageCSVWriter, path="", x=8)

    def test_dataset_write(self):
        """Tests the execution of the GCStorageCSVWriter."""
        iris = load_iris(as_frame=True).data

        gcstorage_writer = GCStorageCSVWriter(
            "s1", bucket_name="rainfall-e8e57.appspot.com", destination_blob_name="test.csv"
        )
        gcstorage_writer.set_input_value("dataset", iris)
        gcstorage_writer.execute()
