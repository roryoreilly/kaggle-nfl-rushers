import os
import sys

import numpy as np
import pandas as pd

from rusher_yards.network import Network
from rusher_yards.settings import Settings


class App:
    def __init__(self):
        self.settings = Settings()
        self.n = Network(self.settings)

    def run(self):
        train_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
        self.n.setup(train_df)
        self.n.add_splitter()
        self.n.train()

    def test_features(self):
        test_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
        test_df_3_DITB = test_df.loc[test_df.DefendersInTheBox == 3].iloc[0:22]
        test_df_6_DITB = test_df.loc[test_df.DefendersInTheBox == 6].iloc[0:22]
        test_df_9_DITB = test_df.loc[test_df.DefendersInTheBox == 9].iloc[0:22]

        for i in range(5, 14):
            training_column_id = 0
            self.batch_size = 2 ** i
            print("**************\nWorking on feature: {}".format(self.settings.get_features(training_column_id)))
            self.run()
            self.dummy_predict(test_df_3_DITB)
            self.dummy_predict(test_df_6_DITB)
            self.dummy_predict(test_df_9_DITB)

    def dummy_predict(self):
        test_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False).iloc[0:22]
        pred = self.n.predict(test_df)
        print(pred)


if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    pd.set_option('display.width', 1000)
    np.set_printoptions(linewidth=1000)
    pd.set_option('display.max_columns', 50)
    app = App()
    app.run()
    app.dummy_predict()
