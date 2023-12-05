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

import pytest

from rain import PandasCSVLoader, PandasPivot
from rain.core.exception import EdgeConnectionError


class TestEdgesConnections:
    def test_matmul_node_str(self):
        n = PandasCSVLoader("load", "./iris.csv")

        edgespec = n @ "dataset"

        assert (
            edgespec.node == n
            and type(edgespec.nodes_attributes) == list
            and "dataset" in edgespec.nodes_attributes
        )

    def test_matmul_node_str_list(self):
        n = PandasCSVLoader("load", "./iris.csv")

        edgespec = n @ ["dataset", "dataset"]

        assert (
            edgespec.node == n
            and type(edgespec.nodes_attributes) == list
            and edgespec.nodes_attributes == ["dataset", "dataset"]
        )

    def test_matmul_non_output_node(self):
        from rain.nodes.spark import SparkSaveModel

        n = SparkSaveModel("load", "./iris.csv")

        with pytest.raises(EdgeConnectionError):
            n @ "dataset1"

    def test_sub_node_to_non_str_nor_list(self):
        n = PandasCSVLoader("load", "./iris.csv")

        with pytest.raises(EdgeConnectionError):
            n @ 5

        with pytest.raises(EdgeConnectionError):
            n @ ("dataset1", "dataset2")

    def test_sub_node_to_non_existing_var(self):
        n = PandasCSVLoader("load", "./iris.csv")

        with pytest.raises(EdgeConnectionError):
            n @ "non_existing_var"

    # def test_gt_edge_to_destination(self):
    #     n = PandasCSVLoader("load", "./iris.csv")
    #     t = PandasPivot("piv", "r", "c", "v")
    #
    #     edge = n @ "dataset" > t
    #
    #     assert (
    #         edge.source == [n]
    #         and edge.destination == [t]
    #         and "dataset" in edge.source_output
    #         and "dataset" in edge.destination_input
    #     )

    # def test_gt_edge_list_to_destination(self):
    #     n = PandasCSVLoader("load", "./iris.csv")
    #     t = PandasPivot("piv", "r", "c", "v")
    #
    #     edge = n @ ["dataset1", "dataset2"] > t
    #
    #     assert (
    #         edge.source == [n]
    #         and edge.destination == [t]
    #         and ["dataset1", "dataset2"] == edge.source_output
    #         and ["dataset1", "dataset2"] == edge.destination_input
    #     )

    def test_gt_edge_at_to_destination(self):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")

        edge = n @ "dataset" > t @ "dataset"

        assert (
            edge.source.node == n
            and edge.destination.node == t
            and "dataset" in edge.source.nodes_attributes
            and "dataset" in edge.destination.nodes_attributes
        )
