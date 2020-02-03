import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


class Trainer:
    def __init__(self, settings, pf):
        self.settings = settings
        self.pf = pf

    def train(self, model_generator, x_df, y, games_df, k_fold_splits=5):
        print("Training model for {} folds".format(5))
        models = []

        x = self._create_x(x_df, y)
        games = np.unique(games_df.values)

        kf = KFold(n_splits=k_fold_splits, shuffle=True, random_state=42)
        kf.get_n_splits(games)
        for game_train_index, game_test_index in kf.split(games):
            train_index = games_df.loc[games_df.isin(games[game_train_index])].index.tolist()
            test_index = games_df.loc[games_df.isin(games[game_test_index])].index.tolist()
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = model_generator.next(x_train, y_train, x_test, y_test)
            models.append(model)
        return models

    def _create_x(self, x_df, y):
        if self.settings.model_type == 'lgbm':
            x = x_df.copy()
            x['Yards'] = y
            x[self.settings.get_features()] = self.pf.scale_standard(x_df)
            x[self.settings.categorical_features()] = self.pf.scale_min_max(x_df)
            return x
        else:
            return self.pf.scale(x_df)
