import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

"""
First we train on our train set. It has 2017/2018 data
We save that
We get new data to predict
Attach our persistent features
"""


class PersistentFeatures:
    def __init__(self, settings, nfl_df):
        self.nfl = nfl_df
        self.rushers = self.nfl.loc[self.nfl.NflId == self.nfl.NflIdRusher]
        self.rushers.set_index('PlayId', inplace=True)
        self.rusher_features = pd.DataFrame(self.rushers['NflId'].unique(), columns=['NflId'])
        self.columns = []
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.settings = settings

    def update(self, features, feature_columns):
        self.standard_scaler.fit(features.drop(columns=['GameId'] + self.settings.categorical_features()))
        self.min_max_scaler.fit(features[self.settings.categorical_features()])
        self.attach_columns(feature_columns)

    def build_features(self):
        self._rusher_features()

    def attach_columns(self, columns):
        self.columns = columns

    def scale(self, features):
        return np.concatenate((self.scale_standard(features), self.scale_min_max(features)), axis=1)

    def scale_standard(self, features):
        return self.standard_scaler.transform(features[self.settings.get_features()])

    def scale_min_max(self, features):
        return self.min_max_scaler.transform(features[self.settings.categorical_features()])

    def _rusher_features(self):
        self._rusher_mean_yards()

    def _rusher_mean_yards(self):
        mean_dict = self.rushers.groupby(['Season', 'NflId'])['Yards'].mean()
        for year in range(2017, 2020):
            try:
                self.rusher_features["RusherMeanYards_{}".format(year)] = self.rusher_features['NflId'].map(
                    mean_dict[year])
            except IndexError:
                pass
        mean_dict = self.rushers.groupby('NflId')['Yards'].mean().to_dict()
        self.rusher_features['RusherMeanYards'] = self.rusher_features['NflId'].map(mean_dict)


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    np.set_printoptions(linewidth=1000)
    pd.set_option('display.max_columns', 50)
    train_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
    pf = PersistentFeatures(train_df)
    pf.build_features()
    print(train_df.head())
