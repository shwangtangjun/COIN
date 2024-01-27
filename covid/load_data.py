import json
import urllib
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, StaticGraphTemporalSignal
import os


class EnglandCovidDatasetLoader(object):
    """A dataset of mobility and history of reported cases of COVID-19
    in England NUTS3 regions, from 3 March to 12 of May. The dataset is
    segmented in days and the graph is directed and weighted. The graph
    indicates how many people moved from one region to the other each day,
    based on Facebook Data For Good disease prevention maps.
    The node features correspond to the number of COVID-19 cases
    in the region in the past **window** days. The task is to predict the
    number of cases in each node after 1 day. For details see this paper:
    `"Transfer Graph Neural Networks for Pandemic Forecasting." <https://arxiv.org/abs/2009.08388>`_
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        self._dataset = json.loads(open('england_covid.json', "r").read())

    def _get_edges(self):
        self._edges = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edges.append(
                np.array(self._dataset["edge_mapping"]["edge_index"][str(time)]).T
            )

    def _get_edge_weights(self):
        self._edge_weights = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edge_weights.append(
                np.array(self._dataset["edge_mapping"]["edge_weight"][str(time)])
            )

    def _get_missing_targets_and_features(self, missing_rate, missing_random_state):
        # missing rate: only (1 - missing_rate) portion of data will be reserved

        stacked_target = np.array(self._dataset["y"]).astype(float)
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
                np.std(stacked_target, axis=0) + 10 ** -10
        )

        standardized_target_backup = np.copy(standardized_target)

        # avoid extreme cases
        is_valid = False
        while not is_valid:
            missing_indices = missing_random_state.choice(standardized_target.size,
                                                          int(missing_rate * standardized_target.size),
                                                          replace=False)
            standardized_target_tmp = np.copy(standardized_target)
            standardized_target_tmp.ravel()[missing_indices] = np.nan
            non_nan_count = np.count_nonzero(~np.isnan(standardized_target_tmp), axis=0)
            if np.min(non_nan_count) >= 2 and np.min(np.nanstd(standardized_target_tmp, axis=0)) > 1e-9:
                # when each row(region) contains no less than 2 different data after masking
                is_valid = True

        standardized_target.ravel()[missing_indices] = np.nan

        self.features = [
            np.nan_to_num(standardized_target[i: i + self.lags, :], nan=0.0).T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]

        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]

        total_time = self._dataset["time_periods"] - self.lags
        for i in range(2 * int(total_time * 0.2), total_time):
            self.targets[i] = standardized_target_backup[i + self.lags, :].T

    def get_dataset(self, lags: int = 8, missing_rate: float = 1.0,
                    missing_random_state=None) -> DynamicGraphTemporalSignal:
        """Returning the England COVID19 data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The England Covid dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_missing_targets_and_features(missing_rate, missing_random_state)
        dataset = DynamicGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
