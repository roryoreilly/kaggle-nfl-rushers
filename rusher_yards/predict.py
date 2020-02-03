import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


class Predictor:
    def __init__(self, settings, pf):
        self.settings = settings
        self.pf = pf

    def predict(self, test_features, models, yardline=0):
        pred = np.zeros((1, 199))
        x = self.pf.scale(test_features)
        for model in models:
            _pred = model.predict(x)
            _pred = np.clip(np.cumsum(_pred, axis=1), 0, 1)
            pred += _pred

        pred /= len(models)
        # Clip predictions past the yardline, as a player cannot go further than endzone
        # pred[0:99 - yardline] = 0.0
        # pred[100 + yardline:-1] = 1.0
        return pred
