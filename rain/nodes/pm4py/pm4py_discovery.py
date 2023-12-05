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

import pm4py
import pandas
from pm4py.objects.bpmn.obj import BPMN
from rain.core.base import ComputationalNode, Tags, LibTag, TypeTag
from rain.core.parameter import KeyValueParameter, Parameters


class Pm4pyInductiveMiner(ComputationalNode):
    """Discovers a model from a the input log using the PM4PY Inductive Miner algortihm.

    Input
    -----
    event_log : pandas.DataFrame
        The pandas DataFrame containing event log.

    Output
    ------
    model : BPMN
        The model discovered by the inductive miner algorithm.

    Parameters
    ----------
    activity_key : str
        Name of the activity field.
    timestamp_key : str
        Name of the timestamp field.
    case_id_key : str
        Name of the case identifier field.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.read_csv.html>`_ for Pandas read_csv
    documentation.
    """

    def __init__(
            self, 
            node_id: str, 
            activity_key: str = "concept:name",
            timestamp_key: str = "time:timestamp",
            case_id_key: str = "case:concept:name"
        ):
        super(Pm4pyInductiveMiner, self).__init__(node_id)
        self.parameters = Parameters(
            activity_key=KeyValueParameter("activity_key", str, activity_key),
            timestamp_key=KeyValueParameter("timestamp_key", str, timestamp_key),
            case_id_key=KeyValueParameter("case_id_key", str, case_id_key),
        )

    _input_vars = {"event_log": pandas.DataFrame}
    _output_vars = {"model": BPMN}

    def execute(self):
        self.model = pm4py.discover_bpmn_inductive(self.event_log)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PM4PY, TypeTag.DISCOVERER)
