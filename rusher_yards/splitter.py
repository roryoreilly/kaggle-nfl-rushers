import os
import sys
import numpy as np
from rusher_yards.model import ModelNN, ModelLGBM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def make_result_set_nn(yards_df):
    results = np.zeros((yards_df.shape[0], 199))
    for i, yard in enumerate(yards_df):
        results[i, yard + 99:] = np.ones(shape=(1, 100 - yard))
    return results


def make_result_set_lgbm(yards_df):
    results = np.zeros((yards_df.shape[0]))
    for i, yard in enumerate(yards_df):
        results[i] = yard
    return results


class Splitter:
    def __init__(self, settings, trainer, predictor):
        self.models = []
        self.settings = settings
        self.trainer = trainer
        self.predictor = predictor
        self.model_generator = None

    def train(self, fe):
        features = self.split(fe.get_features_with_yards())
        y = make_result_set_nn(features.Yards)
        x = features.drop(columns=['Yards', 'GameId'])
        games = features.reset_index()['GameId']
        self.models = self.trainer.train(self.model_generator, x, y, games)
        self.model_generator.eval()

    def predict(self, fe):
        x = self.split(fe.get_useable_features()).drop(columns=['GameId'])
        if x.empty:
            return False, np.zeros((1, 199))
        return True, self.predictor.predict(x, self.models)

    def split(self, df):
        return df


class SplitterNN(Splitter):
    def __init__(self, settings, trainer, predictor, input_shape):
        super().__init__(settings, trainer, predictor)
        self.model_generator = ModelNN(settings, input_shape, 'full')


class SplitterNNGoingLong(SplitterNN):
    def split(self, df):
        return df.loc[(df.IsGoingLong == 1) & (df.IsQB == 0)]


class SplitterNNGoingShort(SplitterNN):
    def split(self, df):
        return df.loc[(df.IsGoingShort == 1) & (df.IsQB == 0)]


class SplitterNNGoingBackwards(SplitterNN):
    def split(self, df):
        return df.loc[df.IsGoingBackwards == 1]


class SplitterNNGoingWide(SplitterNN):
    def split(self, df):
        return df.loc[df.IsGoingWide == 1]


class SplitterNNQB(SplitterNN):
    def split(self, df):
        return df.loc[df.IsQB == 1]


class SplitterNNFirstAndTenRB(SplitterNN):
    def split(self, df):
        return df.loc[df.IsFirstAndTenRB == 1]


class SplitterNNOther(SplitterNN):
    def split(self, df):
        return df.loc[(df.IsFirstAndTenRB == 0) & (df.IsQB == 0) & (df.IsGoingLong == 0) & (df.IsGoingShort == 0) \
                      & (df.IsGoingBackwards == 0) & (df.IsGoingWide == 0)]


class SplitterLGBM(Splitter):
    def __init__(self, settings, trainer, predictor, input_shape):
        super().__init__(settings, trainer, predictor)
        self.model_generator = ModelLGBM(settings, input_shape)

    def train(self, fe):
        features = self.split(fe.get_features_with_yards())
        y = make_result_set_lgbm(features.Yards)
        x = features.drop(columns=['Yards'])
        games = features.reset_index()['GameId']
        self.models = self.trainer.train(self.model_generator, features, y, games)
        self.model_generator.eval()
