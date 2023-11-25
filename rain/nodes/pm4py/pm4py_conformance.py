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

import pandas
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from rain.core.base import Tags, LibTag, TypeTag, ComputationalNode
from typing import Tuple


class Pm4pyTokenBasedReplay(ComputationalNode):
    """Apply token-based replay for conformance checking analysis.

    Input
    -----
    event_log : pandas.DataFrame
        The source event logs.
    model : Tuple[PetriNet, Marking, Marking]
        The discovered reference model.

    Output
    ------
    diagnostics : pandas.DataFrame
        The resulting diagnostics.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.read_csv.html>`_ for Pandas read_csv
    documentation.
    """

    def __init__(
            self, 
            node_id: str, 
        ):
        super(Pm4pyTokenBasedReplay, self).__init__(node_id)

    _input_vars = {"event_log": pandas.DataFrame, "model": Tuple[PetriNet, Marking, Marking]}
    _output_vars = {"diagnostics": pandas.DataFrame}

    def execute(self):
        net, initial_marking, final_marking = self.model
        self.diagnostics: pandas.DataFrame = pm4py.conformance_diagnostics_token_based_replay(
            self.event_log,
            net,
            initial_marking,
            final_marking,
            return_diagnostics_dataframe=True
        )

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PM4PY, TypeTag.CONFORMANCE_CHECKER)
