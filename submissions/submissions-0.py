import math
import os
import sys

import pandas as pd

pd.set_option('max_columns', 100)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


class FeatureExtractor:
    def __init__(self, nfl_df, has_yards=True):
        self.nfl = nfl_df
        self._normalise_starting_df()
        self.rushers = self.nfl.loc[self.nfl.IsRusher]
        self.features = self.rushers[['PlayId']]
        self.rushers.set_index('PlayId', inplace=True)
        self.features.set_index('PlayId', inplace=True)
        self.results = np.zeros((self.features.shape[0], 199))
        self.has_yards = has_yards

    def run(self):
        self._add_rusher_features()
        self._add_pre_play_features()
        self._add_defense_play_features()
        self._add_offense_play_features()

        if self.has_yards:
            self._make_result_set()

    def _make_result_set(self):
        for i, yard in enumerate(self.rushers.Yards):
            self.results[i, yard + 99:] = np.ones(shape=(1, 100 - yard))

    def _add_defense_play_features(self):
        self.features['Def_DL'] = np.array(self.rushers['DefensePersonnel'].str[:1], dtype='int8')
        self.features['Def_LB'] = np.array(self.rushers['DefensePersonnel'].str[6:7], dtype='int8')
        self.features['Def_DB'] = np.array(self.rushers['DefensePersonnel'].str[12:13], dtype='int8')

    def _add_offense_play_features(self):
        self.features['Off_RB'] = np.array(self.rushers['OffensePersonnel'].str.extract('(\d) RB'), dtype='int8')
        self.features['Off_TE'] = np.array(self.rushers['OffensePersonnel'].str.extract('(\d) TE'), dtype='int8')
        self.features['Off_WR'] = np.array(self.rushers['OffensePersonnel'].str.extract('(\d) WR'), dtype='int8')
        off_formations = [pd.get_dummies(self.rushers['OffenseFormation'], prefix='Off_Formation')]
        self.features = self.features.join(off_formations)

    def _add_pre_play_features(self):
        self.features['Week'] = self.rushers['Week']
        self.features['Season_2017'] = self.features['Season_2018'] = self.features['Season_2019'] = 0
        self.features.loc[self.rushers.Season == 2017, 'Season_2017'] = 1
        self.features.loc[self.rushers.Season == 2018, 'Season_2018'] = 1
        self.features.loc[self.rushers.Season == 2019, 'Season_2019'] = 1

        self.features['Quarter'] = self.rushers['Quarter']
        self.features['GameClock_std'] = (900.0 - self.rushers['GameClock'].apply(stringtomins)) / 900.0
        self.features['FullGameClock_std'] = (self.features['GameClock_std'] / 4.0) + (
                (self.features['Quarter'] - 1) * 0.25)

        self.features['OffenseScoreDelta'] = self.rushers['HomeScoreBeforePlay'] - self.rushers[
            'VisitorScoreBeforePlay']
        self.features.loc[self.rushers.PossessionTeam != self.rushers.HomeTeamAbbr, 'OffenseScoreDelta'] \
            = -1 * self.features.loc[self.rushers.PossessionTeam != self.rushers.HomeTeamAbbr, 'OffenseScoreDelta']

        self.features['YardLine_std'] = self.rushers['YardLine_std']
        self.features['Down'] = self.rushers['Down']
        self.features['IsFirstAndTen'] = 1
        self.features.loc[(self.rushers.Distance != 10.0) | (self.rushers.Down != 1), 'IsFirstAndTen'] = 0

    def _add_rusher_features(self):
        rushers_features = self.rushers[['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'NflId', 'PlayerHeight',
                                         'PlayerWeight', 'X_std', 'Y_std', 'Dir_rad', 'Dir_std', 'S_std']]
        rushers_features['PlayerHeight'] = rushers_features['PlayerHeight'].apply(lambda x: 12 * int(x.split('-')[0]) \
                                                                                            + int(x.split('-')[1]))

        self.features = self.features.join(rushers_features)

    def _rusher_mean_yards(self):
        pass

    def _rusher_position_ohe(self):
        pass

    def _normalise_starting_df(self):
        self._fix_team_abbr()
        self._fix_orientation()
        self._fix_speed()
        self._add_possession_columns()
        self._flip_left_plays()

    def _fix_team_abbr(self):
        map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
        for abb in self.nfl['PossessionTeam'].unique():
            map_abbr[abb] = abb
        self.nfl['PossessionTeam'] = self.nfl['PossessionTeam'].map(map_abbr)
        self.nfl['HomeTeamAbbr'] = self.nfl['HomeTeamAbbr'].map(map_abbr)
        self.nfl['VisitorTeamAbbr'] = self.nfl['VisitorTeamAbbr'].map(map_abbr)

    def _fix_orientation(self):
        self.nfl.loc[self.nfl['Season'] == 2017, 'Orientation'] \
            = np.mod(90 + self.nfl.loc[self.nfl['Season'] == 2017, 'Orientation'], 360)

    def _fix_speed(self):
        self.nfl['S_std'] = self.nfl['S']
        self.nfl.loc[self.nfl['Season'] == 2017, 'S'] \
            = (self.nfl['S'][self.nfl['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570

    def _flip_left_plays(self):
        self.nfl['ToLeft'] = self.nfl.PlayDirection == "left"
        self.nfl['YardLine_std'] = 100 - self.nfl.YardLine
        self.nfl.loc[self.nfl.FieldPosition.fillna('') == self.nfl.PossessionTeam, 'YardLine_std'] \
            = self.nfl.loc[self.nfl.FieldPosition.fillna('') == self.nfl.PossessionTeam, 'YardLine']

        self.nfl['X_std'] = self.nfl.X
        self.nfl.loc[self.nfl.ToLeft, 'X_std'] = 120 - self.nfl.loc[self.nfl.ToLeft, 'X']
        self.nfl['Y_std'] = self.nfl.Y - 160 / 6
        self.nfl.loc[self.nfl.ToLeft, 'Y_std'] = 160 / 6 - self.nfl.loc[self.nfl.ToLeft, 'Y']

        self.nfl['Dir_rad'] = np.mod(90 - self.nfl.Dir, 360) * math.pi / 180.0
        self.nfl['Dir_std'] = self.nfl.Dir_rad
        self.nfl.loc[self.nfl.ToLeft, 'Dir_std'] = np.mod(np.pi + self.nfl.loc[self.nfl.ToLeft, 'Dir_rad'], 2 * np.pi)

    def _add_possession_columns(self):
        self.nfl['IsRusher'] = self.nfl.NflId == self.nfl.NflIdRusher
        self.nfl['TeamOnOffense'] = "home"
        self.nfl.loc[self.nfl.PossessionTeam != self.nfl.HomeTeamAbbr, 'TeamOnOffense'] = "away"
        self.nfl['IsOnOffense'] = self.nfl.Team == self.nfl.TeamOnOffense


def stringtomins(x):
    h, m, s = map(int, x.split(':'))
    return (h * 60) + m + (s / 60)

import os
import sys
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

class Model:
    def __init__(self, input_shape, type='nn'):
        self.type = type
        self.input_shape = input_shape

    def next(self):
        if self.type == 'nn':
            return self._create_nn()

    def _create_nn(self):
        model = Sequential()
        model.add(Dense(256, input_shape=[self.input_shape], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(199, activation='sigmoid'))

        model.compile(optimizer='adam', loss=['mse'])
        return model


class ModelType(Enum):
    NN = 1


from sklearn.model_selection import KFold
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def train(model_generator, x, y, batch_size=32, epochs=10, k_fold_splits=5):
    print("Training model for {} folds".format(5))
    models = []

    kf = KFold(n_splits=k_fold_splits)
    kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = model_generator.next()
        model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test))
        models.append(model)
    return models
import pandas as pd
import numpy as np
import os
import sys
from feature_extractor import FeatureExtractor
from model import Model
from train import train
from predict import predict
from ..input.nfl_big_data_bowl_2020.kaggle.competitions import nflrush

class App:
    def __init__(self):
        self.models = []
        pass

    def run(self, make_submission=False):
        fe = FeatureExtractor(pd.read_csv('../input/nfl_big_data_bowl_2020/train.csv', low_memory=False))
        print('Getting features...')
        fe.run()
        m = Model(fe.features.shape[1])
        self.models.append(train(m, fe.features, fe.results))
        if make_submission:
            self.predict_on_env()

    def predict_on_env(self):
        env = nflrush.make_env()
        preds = []
        for (test_df, sample) in env.iter_test():
            fe = FeatureExtractor(test_df)
            fe.run()
            pred = predict(fe.features, self.models)
            env.predict(pd.DataFrame(data=pred, columns=sample.columns))
            preds.append(pred)
        env.write_submission_file()

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    pd.set_option('display.width', 1000)
    np.set_printoptions(linewidth=1000)
    pd.set_option('display.max_columns', 50)
    app = App()
    app.run()
