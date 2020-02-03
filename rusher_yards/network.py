
import os
import sys
import timeit

import numpy as np

from rusher_yards.feature_extractor import FeatureExtractor
from rusher_yards.persistent_features import PersistentFeatures
from rusher_yards.predict import Predictor
from rusher_yards.splitter import *
from rusher_yards.trainer import Trainer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


class Network:
    def __init__(self, settings):
        self.settings = settings
        self.pf = None
        self.basicFactory = None
        self.trainer = None
        self.predictor = None
        self.train_fe = None
        self.input_shape = 0
        self.splitters = []

    def setup(self, train_df):
        self.splitters = []
        self.pf = PersistentFeatures(self.settings, train_df)
        self._timer(self.pf.build_features, "building Persistent features")

        self.train_fe = FeatureExtractor(self.settings, train_df, self.pf)
        self._timer(self.train_fe.run, "getting Train features")

        self.pf.update(self.train_fe.get_useable_features(), self.train_fe.features.columns)
        self.trainer = Trainer(self.settings, self.pf)
        self.predictor = Predictor(self.settings, self.pf)
        self.input_shape = self.train_fe.get_useable_features().shape[1] - 1

    def add_splitter(self):
        if self.settings.model_type == 'nn':
            self._add_basic_nn_splitters()
        if self.settings.model_type == 'nn_split':
            self._add_split_nn_splitters()
        if self.settings.model_type == 'lgbm':
            self._add_basic_lgbm_splitters()

    def _add_basic_nn_splitters(self):
        self.splitters.append(SplitterNN(self.settings, self.trainer, self.predictor, self.input_shape))

    def _add_split_nn_splitters(self):
        self.splitters.append(SplitterNNFirstAndTenRB(self.settings, self.trainer, self.predictor, self.input_shape))
        self.splitters.append(SplitterNNGoingLong(self.settings, self.trainer, self.predictor, self.input_shape))
        self.splitters.append(SplitterNNGoingShort(self.settings, self.trainer, self.predictor, self.input_shape))
        self.splitters.append(SplitterNNGoingBackwards(self.settings, self.trainer, self.predictor, self.input_shape))
        self.splitters.append(SplitterNNGoingWide(self.settings, self.trainer, self.predictor, self.input_shape))
        self.splitters.append(SplitterNNQB(self.settings, self.trainer, self.predictor, self.input_shape))
        self.splitters.append(SplitterNNOther(self.settings, self.trainer, self.predictor, self.input_shape))

    def _add_basic_lgbm_splitters(self):
        self.splitters.append(SplitterLGBM(self.settings, self.trainer, self.predictor, self.input_shape))

    def train(self):
        for split in self.splitters:
            print('Training for split: {}'.format(split.__class__.__name__))
            split.train(self.train_fe)

    def predict(self, test_df):
        fe = FeatureExtractor(self.settings, test_df, self.pf, is_train=False)
        fe.run()
        fe.attach_missing_columns(self.pf.columns)
        pred = np.zeros((1, 199))

        predictions_made = 0
        for split in self.splitters:
            made_prediction, _pred = split.predict(fe)
            if made_prediction:
                pred += _pred
                predictions_made += 1
        pred /= predictions_made
        return pred

    def _timer(self, function, description):
        t0 = timeit.default_timer()
        function()
        t1 = timeit.default_timer()
        print('Finished {}. It took {}s'.format(description, t1 - t0))
