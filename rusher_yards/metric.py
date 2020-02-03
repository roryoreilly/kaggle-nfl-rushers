import os
import sys

import numpy as np
from tensorflow.keras.callbacks import Callback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])


class Metric(Callback):
    def __init__(self, model, callbacks, data, verbose=True):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        x_train, y_train = self.data[0][0], self.data[0][1]
        tr_s = crps(y_train, self.model.predict(x_train))
        tr_s = np.round(tr_s, 6)
        logs['tr_CRPS'] = tr_s

        x_valid, y_valid = self.data[1][0], self.data[1][1]
        val_s = crps(y_valid, self.model.predict(x_valid))
        val_s = np.round(val_s, 6)
        logs['val_CRPS'] = val_s
        if self.verbose:
            print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)
