import pandas as pd

from rain.nodes.unicam.utils.cdc.utils import process_row
from rain.core.base import ComputationalNode, Tags, LibTag, TypeTag


class CDCPreprocessing(ComputationalNode):
    """Loads a pandas DataFrame from a CSV file and applies some preprocessing steps.

    Input
    -----
    input_dataset : pandas.DataFrame
        The pandas DataFrame to write in a CSV file.
    legend : pandas.DataFrame
        Pandas DataFrame used to retrieve additional information on the input dataset.

    Output
    ------
    output_dataset : pandas.DataFrame
        The loaded csv file as a pandas DataFrame.

    Parameters
    ----------
    node_id : str
        Id of the node.
    src_path : str
        Of the CSV file.
    dest_path : str
        Of the resulting CSV file.
    legend_path : str
        Of the legend file.
    """

    _input_vars = {"input_dataset": pd.DataFrame, "legend": pd.DataFrame}
    _output_vars = {"output_dataset": pd.DataFrame}

    def __init__(self, node_id: str):
        super(CDCPreprocessing, self).__init__(node_id)

    def execute(self):
        df: pd.DataFrame = self.input_dataset
        legend: pd.DataFrame = self.legend
        
        tts_legend = []
        for index, row in legend.iterrows():
            tts_legend.append([int(row[0]), row[1]])

        ids = []
        activities = []
        start = []
        end = []
        for index, row in df.iterrows():
            generated_id, activity, is_start, is_end = process_row(row, tts_legend)
            ids.append(generated_id)
            activities.append(activity)
            start.append(is_start)
            end.append(is_end)

        df['GENERATED_KEY'] = pd.Series(ids)
        df['ACTIVITY'] = pd.Series(activities)
        df['IS_START_ACT'] = pd.Series(start)
        df['IS_END_ACT'] = pd.Series(end)
        
        self.output_dataset = df

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.INPUT)
    