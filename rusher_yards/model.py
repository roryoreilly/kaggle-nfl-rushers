import os
import sys

import lightgbm as lgb
from lofo import LOFOImportance, Dataset, plot_importance
import matplotlib as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout

from rusher_yards.metric import Metric, crps

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def custom_loss(y_true, y_pred):
    y_true_cum = tf.cumsum(y_true, axis=1)
    y_pred_cum = tf.cumsum(y_pred, axis=1)
    mse = tf.reduce_mean(tf.square(y_true_cum - y_pred_cum))
    return mse


class ModelNN:
    def __init__(self, settings, input_shape, name):
        self.settings = settings
        self.input_shape = input_shape
        self.fold = 0
        self.score = 0
        self.name = name

    def next(self, x_train, y_train, x_test, y_test):
        model_path = "./models_{}_{}.h5".format(self.name, self.fold)
        if self.settings.load_model_if_exists and os.path.exists(model_path):
            print('Loaded model: {}'.format(model_path))
            model = tf.keras.models.load_model(model_path)
        else:
            model = self._create_nn(x_train, y_train, x_test, y_test)
            model.save(model_path)

        score_ = crps(y_test, model.predict(x_test))
        print("Score fold {}: {}".format(self.fold, score_))
        self.score += score_
        self.fold += 1
        return model

    def _create_nn(self, x_train, y_train, x_test, y_test):
        model = self._nn_model_v1()
        es = EarlyStopping(monitor='val_CRPS', mode='min', verbose=0, patience=8)
        es.set_model(model)
        metric = Metric(model, [es], [(x_train, y_train), (x_test, y_test)], verbose=False)

        model.fit(x_train, y_train, callbacks=[metric], epochs=self.settings.epochs, batch_size=self.settings.batch_size, verbose=0)
        return model

    def _nn_model_v1(self):
        model = Sequential()
        model.add(Dense(512, input_shape=[self.input_shape], activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(199, activation='softmax'))

        model.compile(optimizer='Adam', loss=[custom_loss])
        return model

    def eval(self):
        print("Total average score: {}".format(self.score / self.fold))


class ModelLGBM:
    def __init__(self, settings, input_shape):
        self.settings = settings
        self.input_shape = input_shape
        self.feature_importance_df = pd.DataFrame()
        self.feature_columns = self.settings.get_features() + self.settings.categorical_features()
        self.fold = 0

    def next(self, x_train, y_train, x_test, y_test):
        model = self._create_lgbm(x_train, y_train, x_test, y_test)
        self.fold += 1
        return model

    def _create_lgbm(self, x_train, y_train, x_test, y_test):
        x = pd.concat([x_train, x_test])
        params = {'num_leaves': 15,
                  'objective': 'mae',
                  'learning_rate': 0.1,
                  "boosting": "gbdt",
                  "num_rounds": 100
        }
        model = lgb.LGBMRegressor(**params)
        dataset = Dataset(df=x, target="Yards", features=self.settings.get_features() + self.settings.categorical_features())
        lofo_imp = LOFOImportance(dataset, model=model, scoring="neg_mean_absolute_error",
                                  fit_params={"categorical_feature": self.settings.categorical_features()})
        self._lgbm_feature_importance(lofo_imp)

        model.fit(x_train,
                  y_train,
                  eval_set=[(x_test, y_test)],
                  early_stopping_rounds=self.settings.epochs,
                  eval_metric='mae',
                  verbose=False)
        return model

    def _lgbm_feature_importance(self, lofo_imp):
        importance_df = lofo_imp.get_importance()
        plot_importance(importance_df, figsize=(12, 18))
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = importance_df['feature']
        fold_importance_df["importance"] = importance_df['importance_mean']
        fold_importance_df["fold"] = self.fold
        self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)

    def eval(self):
        self.eval_feature_importance()

    def eval_feature_importance(self):
        print("Features importance...")
        cols_imp = (self.feature_importance_df[["feature", "importance"]]
                    .groupby("feature")
                    .mean()
                    .sort_values(by="importance", ascending=False).index)
        best_features = self.feature_importance_df.loc[self.feature_importance_df.Feature.isin(cols_imp)]

        plt.figure(figsize=(14, 40))
        sns.barplot(x="importance_mean", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (averaged over folds)')
        plt.tight_layout()
        plt.show()
