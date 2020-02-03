import os
import sys
import timeit
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import matplotlib.pylab as plt
import seaborn as sns
import scipy as sp
import lightgbm as lgb
# !pip install lofo-importance
# from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Dropout, ReLU, Input, Layer, Concatenate, \
    Reshape, Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from IPython.display import display
import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', 200)
pd.set_option('max_rows', 100)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=1000)
np.set_printoptions(suppress=True)


class PersistentFeatures:
    def __init__(self, settings, nfl_df):
        self.nfl = nfl_df
        self._fix_nfl()
        self.rushers = self.nfl.loc[self.nfl.NflId == self.nfl.NflIdRusher]
        self.rushers.set_index('PlayId', inplace=True)
        self.rusher_features = pd.DataFrame(self.rushers['NflId'].unique(), columns=['NflId'])
        self.columns = []
        self.standard_scalers = []
        self.min_max_scalers = []
        self.settings = settings
        self.rusher_mean_yards = dict()
        self.rusher_median_yards = dict()
        self.defense_mean_yards = dict()
        self.defense_median_yards = dict()

    def build_features(self):
        self._rusher_features()

    def attach_columns(self, columns):
        self.columns = columns

    def fit(self, x, fold):
        self.standard_scalers.append(StandardScaler())
        self.standard_scalers[fold].fit(x[self.settings.get_features()])
        self.min_max_scalers.append(MinMaxScaler())
        self.min_max_scalers[fold].fit(x[self.settings.categorical_features()])

    def transform(self, x_df, fold):
        x = x_df.copy()
        x[self.settings.get_features()] = self.standard_scalers[fold].transform(x_df[self.settings.get_features()])
        x[self.settings.categorical_features()] = self.min_max_scalers[fold].transform(
            x_df[self.settings.categorical_features()])
        return x

    def _fix_nfl(self):
        map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
        for abb in self.nfl['PossessionTeam'].unique():
            map_abbr[abb] = abb
        self.nfl['PossessionTeam'] = self.nfl['PossessionTeam'].map(map_abbr)
        self.nfl['HomeTeamAbbr'] = self.nfl['HomeTeamAbbr'].map(map_abbr)
        self.nfl['VisitorTeamAbbr'] = self.nfl['VisitorTeamAbbr'].map(map_abbr)
        self.nfl['TeamOnOffense'] = "home"
        self.nfl.loc[self.nfl.PossessionTeam != self.nfl.HomeTeamAbbr, 'TeamOnOffense'] = "away"
        self.nfl['IsOnOffense'] = self.nfl.Team == self.nfl.TeamOnOffense
        self.nfl['TeamOnDefensiveAbbr'] = self.nfl['HomeTeamAbbr']
        self.nfl.loc[self.nfl.PossessionTeam == self.nfl.HomeTeamAbbr, 'TeamOnDefensiveAbbr'] \
            = self.nfl.loc[~self.nfl.IsOnOffense, 'VisitorTeamAbbr']

    def _rusher_features(self):
        self._rusher_mean_yards()
        self._defensive_mean_yards()

    def _rusher_mean_yards(self):
        for year in range(2017, 2020):
            rusher_counts = self.rushers.loc[self.rushers.Season == year, 'NflId'].value_counts()
            rusher_list = rusher_counts[rusher_counts > 10].index.tolist()
            self.rusher_mean_yards[year] = self.rushers.loc[(self.rushers.Season == year)
                                                            & (self.rushers['NflId'].isin(rusher_list))].groupby(
                'NflId')['Yards'].mean().to_dict()
            self.rusher_mean_yards[year][0] = self.rushers.loc[self.rushers.Season == year, 'Yards'].mean()
            self.rusher_median_yards[year] = self.rushers.loc[(self.rushers.Season == year)
                                                              & (self.rushers['NflId'].isin(rusher_list))].groupby(
                'NflId')['Yards'].median().to_dict()
            self.rusher_median_yards[year][0] = self.rushers.loc[self.rushers.Season == year, 'Yards'].median()
        if self.rusher_mean_yards[2019].get(0, True):
            self.rusher_mean_yards[2019][0] = 4.2
            self.rusher_median_yards[2019][0] = 3

    def _defensive_mean_yards(self):
        for year in range(2017, 2020):
            self.defense_mean_yards[year] = self.nfl.loc[(self.nfl.Season == year) & (~self.nfl.IsOnOffense)] \
                .groupby('TeamOnDefensiveAbbr')['Yards'].mean().to_dict()
            self.defense_median_yards[year] = self.nfl.loc[(self.nfl.Season == year) & (~self.nfl.IsOnOffense)] \
                .groupby('TeamOnDefensiveAbbr')['Yards'].median().to_dict()
            self.defense_mean_yards[year][0] = 4.19
            self.defense_median_yards[year][0] = 3


def stringtomins(x):
    h, m, s = map(int, x.split(':'))
    return (h * 60) + m + (s / 60)


class FeatureExtractor:
    PICKLE_PATH = './features.pkl'

    def __init__(self, settings, nfl_df, persistent_features, is_train=True):
        self.settings = settings
        self.nfl = nfl_df
        self.persistent_features = persistent_features
        self._normalise_starting_df()
        self.rushers = self.nfl.loc[self.nfl.IsRusher]
        self.features = self.rushers[['PlayId', 'GameId', 'NflId']]
        self.rushers.set_index('PlayId', inplace=True)
        self.features.set_index('PlayId', inplace=True)
        self.is_train = is_train

    def run(self):
        if self.is_train and self.settings.load_features and self.load_if_exists():
            print("Loaded features for train set from pickle.")
            return
        self.attach_persistent_features()
        self.rusher_features()
        self.splitting_features()
        self.game_features()
        self.defense_features()
        self.offense_features()
        self.play_features()
        #         self.voronoi_features()
        self.features = self.features.fillna(self.features.mean())
        if self.is_train:
            self.features.to_pickle(self.PICKLE_PATH)

    def load_if_exists(self):
        if os.path.exists(self.PICKLE_PATH):
            self.features = pd.read_pickle(self.PICKLE_PATH)
            return True
        return False

    def get_useable_features(self, columns=None):
        base_columns = ['GameId', 'PossessionTeam']
        if columns == None:
            columns = self.settings.categorical_features() + self.settings.get_features()
        return self.features[base_columns + columns]

    def get_features_with_yards(self, columns=None):
        return self.get_useable_features(columns).join(self.rushers['Yards'])

    def attach_missing_columns(self, persistent_columns):
        missing_cols = set(persistent_columns) - set(self.features.columns)
        for c in missing_cols:
            self.features[c] = 0

    def attach_persistent_features(self):
        self.features['RusherMeanYardsSeason'] = self.rushers.apply(
            self.get_rusher_mean_yards, mean_dict=self.persistent_features.rusher_mean_yards,
            reference='NflId', axis=1)
        self.features['RusherMedianYardsSeason'] = self.rushers.apply(
            self.get_rusher_mean_yards, mean_dict=self.persistent_features.rusher_median_yards,
            reference='NflId', axis=1)
        self.features['DefenseMeanYardsSeason'] = self.rushers.apply(
            self.get_rusher_mean_yards, mean_dict=self.persistent_features.defense_mean_yards,
            reference='TeamOnDefensiveAbbr', axis=1)
        self.features['DefenseMedianYardsSeason'] = self.rushers.apply(
            self.get_rusher_mean_yards, mean_dict=self.persistent_features.defense_median_yards,
            reference='TeamOnDefensiveAbbr', axis=1)

    def get_rusher_mean_yards(self, row, mean_dict=[], reference=''):
        try:
            return mean_dict[row['Season']][row[reference]]
        except:
            return mean_dict[row['Season']][0]

    def splitting_features(self):
        self.features['IsFirstAndTenRB'] = self.rushers['IsFirstAndTen']
        self.features.loc[(self.rushers.Position != 'RB'), 'IsFirstAndTenRB'] = 0
        self.features['IsGoingLong'] = 0
        self.features.loc[(self.rushers.Distance > 14.0), 'IsGoingLong'] = 1
        self.features['IsGoingShort'] = 0
        self.features.loc[(self.rushers.Distance < 5.0), 'IsGoingShort'] = 1
        self.features['IsRunningForTD'] = 0
        self.features.loc[(self.rushers.YardLine_std > 94), 'IsRunningForTD'] = 1
        self.features['IsQB'] = 0
        self.features.loc[(self.rushers.Position == 'QB'), 'IsQB'] = 1
        self.features['IsGoingWide'] = (np.abs(self.features['Dir_std']) > 0.95) & (
                np.abs(self.features['Dir_std']) < 1.57)
        self.features['IsGoingBackwards'] = (np.abs(self.features['Dir_std']) > 1.57)
        self.features['IsLookingBackwards'] = 0
        self.features.loc[self.features.Orientation2 > 180, 'IsLookingBackwards'] = 1

    def play_features(self):
        self.features['DistanceToQB'] = np.sqrt(np.sum(
            self.nfl.loc[(self.nfl.Position == 'QB') | self.nfl.IsRusher, ['PlayId', 'X_std', 'Y_std']].groupby(
                'PlayId').agg(['min', 'max']).diff(axis=1).drop([('X_std', 'min'), ('Y_std', 'min')], axis=1) ** 2,
            axis=1))
        self.distances_to_rusher_features()
        self.features['MinTimeToTackle'] = np.abs(
            self.nfl.loc[~self.nfl.IsOnOffense & (self.nfl.RankDisFromPlayStart == 1), 'DisToRusher']) \
                                           / (np.abs(
            self.nfl.loc[~self.nfl.IsOnOffense & (self.nfl.RankDisFromPlayStart == 1), 'S']) \
                                              + np.abs(self.nfl.loc[self.nfl.IsRusher, 'S']))
        self.features['MinTimeToTackle'] = self.features['MinTimeToTackle'].fillna(100)
        self.features['CentripedidalForceToD'] = ((self.features['PlayerWeight'] / 32.2) * (self.features['S'] ** 2)) \
                                                 / self.features['D_Dis_to_R_Min']
        self.features['CentripedidalSpeedToD'] = (self.features['S'] ** 2) / self.features['D_Dis_to_R_Min']
        self.features['ForceFromPrevious'] = 1000
        self.features.loc[self.features.Dis != 0, 'ForceFromPrevious'] = \
            ((self.features.loc[self.features.Dis != 0, 'S'] - self.features.loc[self.features.Dis != 0, 'A']) ** 2) \
            / self.features.loc[self.features.Dis != 0, 'Dis']
        self.features['AngularVelocityChange'] = 1000
        self.features.loc[(self.features['S'] * self.features['Orientation_std']) != 0, 'AngularVelocityChange'] = ((
                                                                                                                                self.features.loc[
                                                                                                                                    (
                                                                                                                                                self.features[
                                                                                                                                                    'S'] *
                                                                                                                                                self.features[
                                                                                                                                                    'Orientation_std']) != 0, 'S'] -
                                                                                                                                self.features.loc[
                                                                                                                                    (
                                                                                                                                                self.features[
                                                                                                                                                    'S'] *
                                                                                                                                                self.features[
                                                                                                                                                    'Orientation_std']) != 0, 'A']) *
                                                                                                                    self.features.loc[
                                                                                                                        (
                                                                                                                                    self.features[
                                                                                                                                        'S'] *
                                                                                                                                    self.features[
                                                                                                                                        'Orientation_std']) != 0, 'Dir_std']) \
                                                                                                                   / (
                                                                                                                               self.features.loc[
                                                                                                                                   (
                                                                                                                                               self.features[
                                                                                                                                                   'S'] *
                                                                                                                                               self.features[
                                                                                                                                                   'Orientation_std']) != 0, 'S'] *
                                                                                                                               self.features.loc[
                                                                                                                                   (
                                                                                                                                               self.features[
                                                                                                                                                   'S'] *
                                                                                                                                               self.features[
                                                                                                                                                   'Orientation_std']) != 0, 'Orientation_std'])

    def distances_to_rusher_features(self):
        self.features['D_Dis_to_R_Min'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher'].min()
        self.features['D_Dis_to_R_Max'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher'].max()
        self.features['D_Dis_to_R_Mean'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher'].mean()
        self.features['D_Dis_to_R_Std'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher'].std()
        self.features['Dis_to_R_Min'] = self.nfl.loc[self.nfl.IsRusher == 0].groupby(['PlayId'])[
            'DisToRusher'].min()
        self.features['Dis_to_R_Max'] = self.nfl.loc[self.nfl.IsRusher == 0].groupby(['PlayId'])[
            'DisToRusher'].max()
        self.features['Dis_to_R_Mean'] = self.nfl.loc[self.nfl.IsRusher == 0].groupby(['PlayId'])[
            'DisToRusher'].mean()
        self.features['Dis_to_R_Std'] = self.nfl.loc[self.nfl.IsRusher == 0].groupby(['PlayId'])[
            'DisToRusher'].std()
        self.features['D_Dis_to_R_Mean_in_1'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher_in_1'].mean()
        self.features['D_Dis_to_R_Min_in_1'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher_in_1'].min()
        self.features['D_Dis_to_R_Mean_in_2'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher_in_2'].mean()
        self.features['D_Dis_to_R_Min_in_2'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher_in_2'].min()
        self.features['D_Dis_to_R_Mean_in_3'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher_in_3'].mean()
        self.features['D_Dis_to_R_Min_in_3'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher_in_3'].min()

    def defense_features(self):
        self.features['Def_DL'] = np.array(self.rushers['DefensePersonnel'].str[:1], dtype='int8')
        self.features['Def_LB'] = np.array(self.rushers['DefensePersonnel'].str[6:7], dtype='int8')
        self.features['Def_DB'] = np.array(self.rushers['DefensePersonnel'].str[12:13], dtype='int8')

        self.features['D_A_mean'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])['A'].mean()
        self.features['D_S_mean'] = self.nfl.loc[~self.nfl.IsOnOffense].groupby(['PlayId'])['S'].mean()
        self._defenders_in_the_box()
        self.defenders_vectors()

    def defenders_vectors(self):
        for i in range(1, 11):
            self.features['D_{}_A_h'.format(i)] = self.nfl.loc[~self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'A_horizontal'].values - self.nfl.loc[
                                                      self.nfl.IsRusher, 'A_horizontal'].values
            self.features['D_{}_A_v'.format(i)] = self.nfl.loc[~self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'A_vertical'].values - self.nfl.loc[
                                                      self.nfl.IsRusher, 'A_vertical'].values
            self.features['D_{}_S_h'.format(i)] = self.nfl.loc[~self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'S_std_horizontal'].values - self.nfl.loc[
                                                      self.nfl.IsRusher, 'S_std_horizontal'].values
            self.features['D_{}_S_v'.format(i)] = self.nfl.loc[~self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'S_std_vertical'].values - self.nfl.loc[
                                                      self.nfl.IsRusher, 'S_std_vertical'].values
            self.features['D_{}_X'.format(i)] = self.nfl.loc[~self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'X_std'].values - self.nfl.loc[
                                                    self.nfl.IsRusher, 'X_std'].values
            self.features['D_{}_Y'.format(i)] = self.nfl.loc[~self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'Y_std'].values - self.nfl.loc[
                                                    self.nfl.IsRusher, 'Y_std'].values
            self.features['O_{}_A_h'.format(i)] = self.nfl.loc[self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'A_horizontal'].values - self.nfl.loc[
                                                      self.nfl.IsRusher, 'A_horizontal'].values
            self.features['O_{}_A_v'.format(i)] = self.nfl.loc[self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'A_vertical'].values - self.nfl.loc[
                                                      self.nfl.IsRusher, 'A_vertical'].values
            self.features['O_{}_S_h'.format(i)] = self.nfl.loc[self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'S_std_horizontal'].values - self.nfl.loc[
                                                      self.nfl.IsRusher, 'S_std_horizontal'].values
            self.features['O_{}_S_v'.format(i)] = self.nfl.loc[self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'S_std_vertical'].values - self.nfl.loc[
                                                      self.nfl.IsRusher, 'S_std_vertical'].values
            self.features['O_{}_X'.format(i)] = self.nfl.loc[self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'X_std'].values - self.nfl.loc[
                                                    self.nfl.IsRusher, 'X_std'].values
            self.features['O_{}_Y'.format(i)] = self.nfl.loc[self.nfl.IsOnOffense & (
                        self.nfl.RankDisFromPlayStart == i), 'Y_std'].values - self.nfl.loc[
                                                    self.nfl.IsRusher, 'Y_std'].values
        self.features.loc[self.features.HasQB, 'QB_A_h'] = self.nfl.loc[self.nfl.Position == 'QB', ['PlayId',
                                                                                                    'A_horizontal']].groupby(
            'PlayId').first().unstack().values - self.nfl.loc[self.nfl.IsRusher & self.nfl.HasQB, 'A_horizontal'].values
        self.features.loc[self.features.HasQB, 'QB_A_v'] = self.nfl.loc[self.nfl.Position == 'QB', ['PlayId',
                                                                                                    'A_vertical']].groupby(
            'PlayId').first().unstack().values - self.nfl.loc[self.nfl.IsRusher & self.nfl.HasQB, 'A_vertical'].values
        self.features.loc[self.features.HasQB, 'QB_S_h'] = self.nfl.loc[self.nfl.Position == 'QB', ['PlayId',
                                                                                                    'S_std_horizontal']].groupby(
            'PlayId').first().unstack().values - self.nfl.loc[
                                                               self.nfl.IsRusher & self.nfl.HasQB, 'S_std_horizontal'].values
        self.features.loc[self.features.HasQB, 'QB_S_v'] = self.nfl.loc[self.nfl.Position == 'QB', ['PlayId',
                                                                                                    'S_std_vertical']].groupby(
            'PlayId').first().unstack().values - self.nfl.loc[
                                                               self.nfl.IsRusher & self.nfl.HasQB, 'S_std_vertical'].values
        self.features.loc[self.features.HasQB, 'QB_X'] = self.nfl.loc[
                                                             self.nfl.Position == 'QB', ['PlayId', 'X_std']].groupby(
            'PlayId').first().unstack().values - self.nfl.loc[self.nfl.IsRusher & self.nfl.HasQB, 'X_std'].values
        self.features.loc[self.features.HasQB, 'QB_Y'] = self.nfl.loc[
                                                             self.nfl.Position == 'QB', ['PlayId', 'Y_std']].groupby(
            'PlayId').first().unstack().values - self.nfl.loc[self.nfl.IsRusher & self.nfl.HasQB, 'Y_std'].values

    def _defenders_in_the_box(self):
        self.features['DefendersInTheBox'] = self.rushers['DefendersInTheBox']
        self.features['DITB_Centroid_X'] = \
            self.nfl.loc[self.nfl.IsDefenderInBox].groupby('PlayId')[['X_std']].mean()
        self.features['DITB_Centroid_Y'] = \
            self.nfl.loc[self.nfl.IsDefenderInBox].groupby('PlayId')[['Y_std']].mean()
        self.features['DITB_Spread_X'] = \
            self.nfl.loc[self.nfl.IsDefenderInBox].groupby('PlayId')['X_std'].agg(['min', 'max']).diff(axis=1)[
                'max']
        self.features['DITB_Spread_Y'] = \
            self.nfl.loc[self.nfl.IsDefenderInBox].groupby('PlayId')['Y_std'].agg(['min', 'max']).diff(axis=1)[
                'max']

        self.features['DITB_Centroid_X_Dis_to_rusher'] = \
            np.absolute(self.features['DITB_Centroid_X'] - self.rushers['X_std'])
        self.features['DITB_Centroid_Y_Dis_to_rusher'] = \
            np.absolute(self.features['DITB_Centroid_Y'] - self.rushers['Y_std'])

        self.features['DITB_A_mean'] = self.nfl.loc[self.nfl.IsDefenderInBox].groupby(['PlayId'])['A'].mean()
        self.features['DITB_S_mean'] = self.nfl.loc[self.nfl.IsDefenderInBox].groupby(['PlayId'])['S'].mean()

    def offense_features(self):
        self.features['Off_RB'] = np.array(self.rushers['OffensePersonnel'].str.extract('(\d) RB'), dtype='int8')
        self.features['Off_TE'] = np.array(self.rushers['OffensePersonnel'].str.extract('(\d) TE'), dtype='int8')
        self.features['Off_WR'] = np.array(self.rushers['OffensePersonnel'].str.extract('(\d) WR'), dtype='int8')
        #         self.features['OffenseFormation'] = self.rushers['OffenseFormation']
        off_formations = [pd.get_dummies(self.rushers['OffenseFormation'], prefix='Off_Formation')]
        self.features = self.features.join(off_formations)

    def game_features(self):
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
        self.features['ScoreUrgency'] = self.rushers['Quarter'] * self.features['OffenseScoreDelta']
        self.features['DefendersInTheBox_vs_Distance'] = self.rushers['DefendersInTheBox'] / self.rushers['Distance']
        self.features.loc[self.rushers.PossessionTeam != self.rushers.HomeTeamAbbr, 'OffenseScoreDelta'] \
            = -1 * self.features.loc[self.rushers.PossessionTeam != self.rushers.HomeTeamAbbr, 'OffenseScoreDelta']

        self.features['YardLine_std'] = self.rushers['YardLine_std']
        self.features['Down'] = self.rushers['Down']

    def rusher_features(self):
        rushers_features = self.rushers[['S', 'A', 'Dis', 'Orientation', 'Orientation2', 'Dir', 'Dir2',
                                         'PlayerHeight', 'PlayerWeight', 'X_std', 'Y_std', 'Dir_rad', 'Dir_std',
                                         'S_std', 'Distance', 'YardLine', 'A_horizontal', 'A_vertical',
                                         'S_horizontal', 'S_vertical', 'S_std_horizontal', 'S_std_vertical',
                                         'PossessionTeam', 'Orientation_std', 'HasQB']].copy(deep=True)
        rushers_features['PlayerHeight'] = rushers_features['PlayerHeight'] \
            .apply(lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))

        self.features = self.features.join(rushers_features)
        self._rusher_position_ohe()
        self.features['Back_from_Scrimmage'] = self.rushers['YardLine_std'] - self.rushers['X_std']
        self.features["Dir_sin"] = np.sin(self.rushers['Dir_std'])
        self.features["Dir_cos"] = np.cos(self.rushers['Dir_std'])
        self.features['Dir_vs_Orientation'] = np.abs(self.rushers['Dir_std'] - self.rushers['Orientation_std'])
        self.features['Force'] = self.rushers['A'] * self.rushers['PlayerWeight'] / 32.2
        self.features['Momentum'] = self.rushers['S'] * self.rushers['PlayerWeight'] / 32.2
        self.features['Momentum_in_1'] = (self.rushers['S'] + self.rushers['A']) * self.rushers['PlayerWeight'] / 32.2
        self.features['Momentum_in_2'] = (self.rushers['S'] + (self.rushers['A'] * 2)) * self.rushers[
            'PlayerWeight'] / 32.2

    def _rusher_position_ohe(self):
        #         self.features['Position'] = self.rushers['Position']
        rusher_position = [pd.get_dummies(self.rushers['Position'], prefix='Rusher_Position')]
        self.features = self.features.join(rusher_position)

    def _normalise_starting_df(self):
        self._fix_team_abbr()
        self._fix_orientation()
        self._fix_speed()
        self._add_possession_columns()
        self._flip_left_plays()
        self._distance_to_centers()
        self._vectors()
        self.all_positions_after_t_seconds()
        self._distances_to_rusher()
        for i in range(1, 5):
            self._distances_to_rusher("_in_" + str(i))
        self.nfl['IsFirstAndTen'] = 1
        self.nfl.loc[(self.nfl.Distance != 10.0) | (self.nfl.Down != 1), 'IsFirstAndTen'] = 0
        has_qb = self.nfl.groupby('PlayId')['Position'].apply(lambda p: 'QB' in p.values).to_dict()
        self.nfl['HasQB'] = self.nfl['PlayId'].map(has_qb)

    def _distances_to_rusher(self, t=''):
        nfl_copy = self.nfl.copy()
        rusher = nfl_copy.loc[nfl_copy.IsRusher]
        rusher.set_index('PlayId', inplace=True)
        nfl_copy['Rusher_X_std'] = nfl_copy['PlayId'].map(rusher['X_std' + t].to_dict())
        nfl_copy['Rusher_Y_std'] = nfl_copy['PlayId'].map(rusher['Y_std' + t].to_dict())
        distances_to_rusher = (nfl_copy.X_std - nfl_copy.Rusher_X_std) ** 2 + (
                nfl_copy.Y_std - nfl_copy.Rusher_Y_std) ** 2
        self.nfl['DisToRusher' + t] = np.sqrt(distances_to_rusher)

    def _vectors(self):
        self.nfl['A_horizontal'] = self.nfl['A'] * np.cos(self.nfl['Dir_rad'])
        self.nfl['A_vertical'] = self.nfl['A'] * np.sin(self.nfl['Dir_rad'])
        self.nfl['S_horizontal'] = self.nfl['S'] * np.cos(self.nfl['Dir_rad'])
        self.nfl['S_vertical'] = self.nfl['S'] * np.sin(self.nfl['Dir_rad'])
        self.nfl['S_std_horizontal'] = self.nfl['S_std'] * np.cos(self.nfl['Dir_rad'])
        self.nfl['S_std_vertical'] = self.nfl['S_std'] * np.sin(self.nfl['Dir_rad'])

    def _fix_team_abbr(self):
        map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
        for abb in self.nfl['PossessionTeam'].unique():
            map_abbr[abb] = abb
        self.nfl['PossessionTeam'] = self.nfl['PossessionTeam'].map(map_abbr)
        self.nfl['HomeTeamAbbr'] = self.nfl['HomeTeamAbbr'].map(map_abbr)
        self.nfl['VisitorTeamAbbr'] = self.nfl['VisitorTeamAbbr'].map(map_abbr)

    def _fix_orientation(self):
        self.nfl['Orientation2'] = self.nfl['Orientation']
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
        self.nfl.loc[self.nfl['Y_std'] >= 160 / 6, 'Y_std'] = 26.6

        self.nfl.loc[self.nfl.ToLeft, 'Orientation2'] = 360 - self.nfl.loc[self.nfl.ToLeft, 'Orientation2']
        self.nfl.loc[self.nfl.ToLeft, 'Orientation'] = np.mod(180 + self.nfl.loc[self.nfl.ToLeft, 'Orientation'], 360)
        self.nfl['Orientation_std'] = np.mod(90 - self.nfl.Orientation, 360) * math.pi / 180
        self.nfl['Dir2'] = self.nfl['Dir']
        self.nfl.loc[self.nfl.ToLeft, 'Dir2'] = 360 - self.nfl.loc[self.nfl.ToLeft, 'Dir2']
        self.nfl['Dir_rad'] = np.mod(270 - self.nfl.Dir, 360) * math.pi / 180.0
        self.nfl['Dir_std'] = self.nfl.Dir_rad
        self.nfl.loc[self.nfl.ToLeft, 'Dir_std'] = np.mod(np.pi + self.nfl.loc[self.nfl.ToLeft, 'Dir_rad'],
                                                          2 * np.pi)
        self.nfl['Dir_std'] = self.nfl['Dir_std'] - math.pi

    def _add_possession_columns(self):
        self.nfl['IsRusher'] = self.nfl.NflId == self.nfl.NflIdRusher
        self.nfl['TeamOnOffense'] = "home"
        self.nfl.loc[self.nfl.PossessionTeam != self.nfl.HomeTeamAbbr, 'TeamOnOffense'] = "away"
        self.nfl['IsOnOffense'] = self.nfl.Team == self.nfl.TeamOnOffense
        self.nfl['TeamOnDefensiveAbbr'] = self.nfl['HomeTeamAbbr']
        self.nfl.loc[self.nfl.PossessionTeam == self.nfl.HomeTeamAbbr, 'TeamOnDefensiveAbbr'] \
            = self.nfl.loc[self.nfl.PossessionTeam == self.nfl.HomeTeamAbbr, 'VisitorTeamAbbr']

    def _distance_to_centers(self):
        self.nfl['DisFromPlayStart'] = np.sqrt(
            (self.nfl.X_std - self.nfl.YardLine_std - 10) ** 2 + (self.nfl.Y_std ** 2))
        ranks = self.nfl.groupby(['PlayId', 'IsOnOffense'])['DisFromPlayStart'] \
            .rank(ascending=True, method='first')
        ranks.name = 'RankDisFromPlayStart'
        self.nfl = pd.concat([self.nfl, ranks], axis=1)
        self.nfl['IsDefenderInBox'] = False
        self.nfl.loc[(~self.nfl.IsOnOffense) &
                     (self.nfl.DefendersInTheBox >= self.nfl.RankDisFromPlayStart), ['IsDefenderInBox']] = True

    def all_positions_after_t_seconds(self):
        self._positions_after_t_seconds(1)
        self._positions_after_t_seconds(2)
        self._positions_after_t_seconds(3)
        self._positions_after_t_seconds(4)
        self._positions_after_t_seconds(5)

    def _positions_after_t_seconds(self, t):
        if t <= 2:
            self.nfl['X_std_in_{}'.format(t)] = self.nfl['X_std'] \
                                                + (self.nfl['S_horizontal'] * t) + (
                                                            0.5 * self.nfl['A_horizontal'] * (t ** 2))
            self.nfl['Y_std_in_{}'.format(t)] = self.nfl['Y_std'] \
                                                + (self.nfl['S_vertical'] * t) + (
                                                            0.5 * self.nfl['A_vertical'] * (t ** 2))
        else:
            self.nfl['X_std_in_{}'.format(t)] = self.nfl['X_std'] \
                                                + ((self.nfl['S_horizontal'] + self.nfl['A_horizontal'] * 2) * 2) + (
                                                            0.5 * self.nfl['A_horizontal'] * (2 ** 2)) \
                                                + ((self.nfl['S_horizontal'] + (self.nfl['A_horizontal'] * 2)) * (
                        t - 2))
            self.nfl['Y_std_in_{}'.format(t)] = self.nfl['Y_std'] \
                                                + ((self.nfl['S_vertical'] + self.nfl['A_vertical'] * 2) * 2) + (
                                                            0.5 * self.nfl['A_vertical'] * (2 ** 2)) \
                                                + ((self.nfl['S_vertical'] + (self.nfl['A_vertical'] * 2)) * (t - 2))

    def voronoi_features(self):
        vertices = self.nfl.groupby('PlayId')[['X_std', 'Y_std']].apply(
            lambda play: self._get_voronoi_vertices(np.array(play))).unstack()
        self.nfl['Voronoi_vertices'] = vertices.values

        areas = vertices.apply(lambda v: sp.spatial.ConvexHull(v).volume)
        self.nfl['Voronoi_area'] = areas.values

        self.features['Voronoi_rusher_area'] = self.nfl.loc[self.nfl.IsRusher, 'Voronoi_area'].values
        self.features['Voronoi_DITB_area'] = \
            self.nfl.loc[self.nfl.IsDefenderInBox].groupby('PlayId')[['Voronoi_area']].sum()
        self.features['Voronoi_Offense_area'] = \
            self.nfl.loc[self.nfl.IsOnOffense].groupby('PlayId')[['Voronoi_area']].sum()
        self.features['Voronoi_Defense_area'] = \
            self.nfl.loc[~self.nfl.IsOnOffense].groupby('PlayId')[['Voronoi_area']].sum()
        self._voronoi_max_offense_box_y()

    def _voronoi_max_offense_box_y(self):
        nfl_copy = self.nfl.copy()
        nfl_copy['MaxYDITB'] = nfl_copy['PlayId'] \
            .map(nfl_copy.loc[nfl_copy.IsDefenderInBox].groupby('PlayId')['Y_std'].max().to_dict())
        nfl_copy['MinYDITB'] = nfl_copy['PlayId'] \
            .map(nfl_copy.loc[nfl_copy.IsDefenderInBox].groupby('PlayId')['Y_std'].min().to_dict())
        nfl_copy['YInDITBRange'] = (nfl_copy['MaxYDITB'] + 1 > nfl_copy['Y_std']) & (
                nfl_copy['Y_std'] > nfl_copy['MinYDITB'] - 1)
        self.features['LargestYForOffenseInBox'] = self.features.index \
            .map((nfl_copy.loc[nfl_copy.IsOnOffense & nfl_copy.YInDITBRange].groupby('PlayId')['X_std'].max() -
                  self.features['YardLine_std']).to_dict())

    def _get_voronoi_vertices(self, points_center):
        bounding_box = np.array([0, 120, -160 / 6, 160 / 6])  # [x_min, x_max, y_min, y_max]

        # Mirror points
        points_left = np.copy(points_center)
        points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
        points_right = np.copy(points_center)
        points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
        points_up = np.copy(points_center)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
        points = np.append(points_center,
                           np.append(np.append(points_left, points_right, axis=0),
                                     np.append(points_down, points_up, axis=0),
                                     axis=0),
                           axis=0)
        vor = sp.spatial.Voronoi(points)

        point_regions = vor.point_region[0:22]
        all_regions = np.array(vor.regions)
        all_vertices = np.array(vor.vertices)
        vertices = [all_vertices[r] for r in all_regions[point_regions]]

        return pd.Series(vertices)


def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])


def custom_loss(y_true, y_pred):
    y_true_cum = tf.math.cumsum(y_true, axis=1)
    y_pred_cum = tf.math.cumsum(y_pred, axis=1)
    mse = tf.math.reduce_mean(tf.math.square(y_true_cum - y_pred_cum))

    #     epsilon = tf.constant(0.00001, shape=[199])
    #     stop_empty = tf.math.reduce_sum(tf.math.divide(y_true, tf.math.add(y_pred, epsilon)))
    #     stop_empty = tf.math.divide(tf.math.minimum(stop_empty, 1000), 10)
    return mse


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


class ModelNN:
    def __init__(self, settings, input_shape, name):
        self.settings = settings
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

        y_pred = model.predict(x_test)
        score_ = crps(y_test, y_pred)
        print("Score fold {}: {}".format(self.fold, score_))
        self.score += score_
        self.fold += 1
        return model

    def _create_nn(self, x_train, y_train, x_test, y_test):
        model_path = "./models_{}_{}.hdf5".format(self.name, self.fold)
        model = self._nn_model_v1()
        es = EarlyStopping(monitor='val_CRPS', mode='min', verbose=0, patience=5)
        es.set_model(model)
        metric = Metric(model, [es], [(x_train, y_train), (x_test, y_test)], verbose=False)
        mc = ModelCheckpoint(model_path, monitor='val_CRPS', mode='min', save_best_only=True,
                             verbose=0, save_weights_only=True)
        #         for i in range(5, 8):
        #             model.fit(x_train, y_train, batch_size=2**i, verbose=0)
        model.fit(x_train, y_train, callbacks=[metric, mc], epochs=self.settings.epochs,
                  batch_size=self.settings.batch_size, verbose=0)
        model.load_weights(model_path)
        return model

    def _nn_model_v1(self):
        model = Sequential()
        model.add(Dense(1024, input_shape=[self.settings.get_input_shape()], activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(199, activation='softmax'))

        model.compile(optimizer='Adam', loss=['categorical_crossentropy'])
        return model

    def _nn_model_v2(self):
        inputs = []
        embeddings = []
        for i in self.settings.categorical_features():
            input_ = Input(shape=(1,))
            embedding = Embedding(2, 10, input_length=1)(input_)
            embedding = Reshape(target_shape=(10,))(embedding)
            inputs.append(input_)
            embeddings.append(embedding)
        input_numeric = Input(shape=(len(self.settings.get_features()),))
        embedding_numeric = Dense(512, activation='relu')(input_numeric)
        inputs.append(input_numeric)
        embeddings.append(embedding_numeric)
        x = Concatenate()(embeddings)
        x = Concatenate()(embeddings)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(199, activation='softmax')(x)
        model = Model(inputs, output)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[])
        return model

    def eval(self):
        print("Total average score: {}".format(self.score / self.fold))


class ModelFeaturesImportance:
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
        dataset = Dataset(df=x, target="Yards",
                          features=self.settings.get_features() + self.settings.categorical_features())
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
        cols_imp = (self.feature_importance_df[["Feature", "importance"]]
                    .groupby("Feature")
                    .mean()
                    .sort_values(by="importance", ascending=False).index)
        best_features = self.feature_importance_df.loc[self.feature_importance_df.Feature.isin(cols_imp)]

        plt.figure(figsize=(14, 40))
        sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (averaged over folds)')
        plt.tight_layout()
        plt.show()


class Trainer:
    def __init__(self, settings, pf):
        self.settings = settings
        self.pf = pf

    def train(self, model_generator, x_df, y):
        models = []

        if self.settings.using_last_games_as_validation:
            model = self.train_on_last(model_generator, x_df, y)
            models.append(model)
        else:
            models = self.train_with_ksplit(model_generator, x_df, y)
        return models

    def train_on_last(self, model_generator, x_df, y):
        print("Training model with validation of last 5 team games")
        game_ids = self.split_into_last_games(x_df)

        x_train = x_df[~x_df['GameId'].isin(game_ids)].drop(columns=['GameId', 'PossessionTeam'])
        x_test = x_df[x_df['GameId'].isin(game_ids)].drop(columns=['GameId', 'PossessionTeam'])

        self.pf.fit(x_train, 0)
        x_train = self.pf.transform(x_train, 0)
        x_test = self.pf.transform(x_test, 0)

        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)

        # Use train/test index to split target variable
        train_inds, test_inds = x_train.index, x_test.index
        y_train, y_test = y[train_inds], y[test_inds]

        if self.settings.model_type == 'lgbm':
            x_train['Yards'] = y_train
            x_test['Yards'] = y_test
        model = model_generator.next(x_train, y_train, x_test, y_test)
        return model

    @staticmethod
    def split_into_last_games(df):
        games = df[['GameId', 'PossessionTeam']].drop_duplicates()

        # Sort so the latest games are first and label the games with cumulative counter
        games = games.sort_values(['PossessionTeam', 'GameId'], ascending=[True, False])
        games['row_number'] = games.groupby(['PossessionTeam']).cumcount() + 1

        # Use last 5 games for each team as validation. There will be overlap since two teams will have the same
        # GameId
        game_set = set([1, 2, 3, 4, 5])

        # Set of unique game ids
        game_ids = set(games[games['row_number'].isin(game_set)]['GameId'].unique().tolist())

        return game_ids

    def train_with_ksplit(self, model_generator, x_df, y):
        print("Training model for {} folds".format(self.settings.k_fold_splits))
        games_df = x_df.reset_index()['GameId']
        x = x_df.drop(columns=['GameId', 'PossessionTeam'])
        games = np.unique(games_df.values)
        models = []

        for k_folds_set in range(self.settings.k_folds_sets):
            kf = KFold(n_splits=self.settings.k_fold_splits, shuffle=True, random_state=42 + k_folds_set)
            kf.get_n_splits(games)

            fold = 0
            for game_train_index, game_test_index in kf.split(games):
                train_index = games_df.loc[games_df.isin(games[game_train_index])].index.tolist()
                test_index = games_df.loc[games_df.isin(games[game_test_index])].index.tolist()
                y_train, y_test = y[train_index], y[test_index]
                x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                self.pf.fit(x_train, (k_folds_set * self.settings.k_fold_splits) + fold)
                x_train = self.pf.transform(x_train, (k_folds_set * self.settings.k_fold_splits) + fold)
                x_test = self.pf.transform(x_test, (k_folds_set * self.settings.k_fold_splits) + fold)
                if self.settings.model_type == 'lgbm':
                    x_train['Yards'] = y_train
                    x_test['Yards'] = y_test
                model = model_generator.next(x_train, y_train, x_test, y_test)
                models.append(model)
                fold += 1
        return models


class Predictor:
    def __init__(self, settings, pf):
        self.settings = settings
        self.pf = pf

    def predict(self, test_features, models, yardlines=0):
        pred = np.zeros((len(test_features), 199))
        for fold, model in enumerate(models):
            x = self.pf.transform(test_features, fold)
            _pred = model.predict(x)
            _pred = np.clip(np.cumsum(_pred, axis=1), 0, 1)
            pred += _pred

        pred /= len(models)
        # Clip predictions past the yardline, as a player cannot go further than endzone
        for index, yardline in enumerate(yardlines):
            pred[index][:int(99 - yardline)] = 0.0
            pred[index][int(199 - yardline):] = 1.0
        return pred


def make_result_set_nn(yards_df):
    results = np.zeros((yards_df.shape[0], 199))
    for i, yard in enumerate(yards_df):
        #         results[i, yard + 99:] = np.ones(shape=(1, 100 - yard))
        results[i, yard + 99] = 1
    return results


def make_result_set_dummy(yards_df):
    results = np.zeros((yards_df.shape[0], 199))
    for i, yard in enumerate(yards_df):
        #         results[i, yard + 99:] = np.ones(shape=(1, 100 - yard))
        results[i, yard + 99] = 1
    return np.clip(np.cumsum(results, axis=1), 0, 1)


def make_result_set_lgbm(yards_df):
    results = np.zeros((yards_df.shape[0]))
    for i, yard in enumerate(yards_df):
        results[i] = yard
    return results


class Splitter:
    def __init__(self, settings, trainer, predictor, weight=1.):
        self.models = []
        self.settings = settings
        self.trainer = trainer
        self.predictor = predictor
        self.model_generator = None
        self.weight = weight

    def train(self, fe):
        features = self.split(
            fe.get_features_with_yards(columns=self.settings.categorical_features() + self.settings.get_features()))
        y = make_result_set_nn(features.Yards)
        x = features.drop(columns=['Yards'])
        self.models = self.trainer.train(self.model_generator, x, y)
        self.model_generator.eval()

    def predict(self, fe):
        x = self.split(
            fe.get_useable_features(columns=self.settings.categorical_features() + self.settings.get_features())).drop(
            columns=['GameId', 'PossessionTeam'])
        x = x.reset_index(drop=True)
        yardline = fe.features['YardLine_std']
        if x.empty:
            return False, np.zeros((1, 199))
        return True, self.predictor.predict(x, self.models, yardline) * self.weight

    def split(self, df):
        return df


class SplitterNN(Splitter):
    def __init__(self, settings, trainer, predictor, input_shape, weight=1.):
        super().__init__(settings, trainer, predictor, weight)
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
        self.models = self.trainer.train(self.model_generator, x, y)
        self.model_generator.eval()


class SplitterFeatureImportance(SplitterLGBM):
    def __init__(self, settings, trainer, predictor, input_shape):
        super().__init__(settings, trainer, predictor)
        self.model_generator = ModelFeatureImportance(settings, input_shape)


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
        self.splitters_weight = []

    def setup(self, train_df):
        self.splitters = []
        self.pf = PersistentFeatures(self.settings, train_df)
        self._timer(self.pf.build_features, "building Persistent features")

        self.train_fe = FeatureExtractor(self.settings, train_df, self.pf)
        self._timer(self.train_fe.run, "getting Train features")

        self.pf.attach_columns(self.train_fe.features.columns)
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
        if self.settings.model_type == 'feat_impor':
            self._add_basic_feat_impor_splitters()

    def _add_basic_nn_splitters(self):
        self.splitters.append(SplitterNN(self.settings, self.trainer, self.predictor, self.input_shape))

    def _add_split_nn_splitters(self):
        self.splitters.append(SplitterNN(self.settings, self.trainer, self.predictor, self.input_shape, weight=0.65))

        settings2 = Settings()
        settings2.columns = ['A_horizontal', 'A_vertical', 'S_horizontal', 'S_horizontal', 'X_std', 'Y_std',
                             'D_1_A_h', 'D_1_S_h', 'D_1_A_v', 'D_1_S_h', 'D_1_X', 'D_1_Y',
                             'D_2_A_h', 'D_2_S_h', 'D_2_A_v', 'D_2_S_h', 'D_2_X', 'D_2_Y',
                             'D_3_A_h', 'D_3_S_h', 'D_3_A_v', 'D_3_S_h', 'D_3_X', 'D_3_Y',
                             'D_4_A_h', 'D_4_S_h', 'D_4_A_v', 'D_4_S_h', 'D_4_X', 'D_4_Y',
                             'D_5_A_h', 'D_5_S_h', 'D_5_A_v', 'D_5_S_h', 'D_5_X', 'D_5_Y',
                             'QB_A_h', 'QB_S_h', 'QB_A_v', 'QB_S_h', 'QB_X', 'QB_Y']
        pf = PersistentFeatures(settings2, self.train_fe.nfl)
        trainer = Trainer(settings2, pf)
        predictor = Predictor(settings2, pf)
        self.splitters.append(SplitterNN(settings2, trainer, predictor, len(settings2.columns), weight=0.35))

    def _add_basic_lgbm_splitters(self):
        self.splitters.append(SplitterLGBM(self.settings, self.trainer, self.predictor, self.input_shape))

    def _add_basic_feat_impor_splitters(self):
        self.splitters.append(SplitterFeaturesImportance(self.settings, self.trainer, self.predictor, self.input_shape))

    def train(self):
        for split in self.splitters:
            print('Training for split: {}'.format(split.__class__.__name__))
            split.train(self.train_fe)

    def predict(self, test_df):
        fe = FeatureExtractor(self.settings, test_df, self.pf, is_train=False)
        fe.run()
        fe.attach_missing_columns(self.pf.columns)

        x_train = self.train_fe.get_features_with_yards().drop(columns=['GameId', 'PossessionTeam'])
        x_test = fe.get_useable_features().drop(columns=['GameId', 'PossessionTeam'])

        self.pf.fit(x_train, 0)
        x_train = self.pf.transform(x_train, 0).values
        x_test = self.pf.transform(x_test, 0).values

        play_indices, _ = self.knn(x_train, x_test, 10, self.euclidean_distance, self.mean)

        play_recommendations = []
        original_plays = self.train_fe.get_features_with_yards()
        original_plays.reset_index()
        for _, index in play_indices:
            play_recommendations.append(original_plays.iloc[index])

        print(pred)
        return pred

    def knn(self, data, query, k, distance_fn, choice_fn):
        neighbor_distances_and_indices = []

        for index, example in enumerate(data):
            distance = distance_fn(example[:-1], query)
            neighbor_distances_and_indices.append((distance, index))
        sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
        k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
        k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]

        return k_nearest_distances_and_indices, choice_fn(k_nearest_labels)

    def euclidean_distance(self, point1, point2):
        return math.sqrt(((point1 - point2) ** 2).sum(axis=1).sum(axis=0))

    def mean(self, labels):
        return sum(labels) / len(labels)

    def _timer(self, function, description):
        t0 = timeit.default_timer()
        function()
        t1 = timeit.default_timer()
        print('Finished {}. It took {}s'.format(description, t1 - t0))


class Settings:
    def __init__(self, epochs=100):
        self.epochs = epochs
        self.batch_size = 1024
        self.model_type = 'nn'
        self.load_model_if_exists = False
        self.load_features = True
        self.k_fold_splits = 8
        self.k_folds_sets = 1
        self.using_last_games_as_validation = True
        self.columns = ['Back_from_Scrimmage', 'Y_std', 'A', 'A_horizontal', 'A_vertical',
                        'Dis_to_R_Min', 'Dis_to_R_Max', 'Dis_to_R_Mean', 'Dis_to_R_Std',
                        'D_Dis_to_R_Min', 'D_Dis_to_R_Mean', 'D_Dis_to_R_Std', 'D_Dis_to_R_Max',
                        'S_std', 'S_std_vertical', 'S_std_horizontal', 'Dis', 'YardLine_std',
                        'Orientation', 'DistanceToQB', 'MinTimeToTackle', 'Orientation_std', 'Force',
                        'DefendersInTheBox', 'RusherMeanYardsSeason', 'RusherMedianYardsSeason',
                        'DefenseMeanYardsSeason', 'DefenseMedianYardsSeason',
                        'D_1_A_h', 'D_1_S_h', 'D_1_A_v', 'D_1_S_v',
                        'D_2_A_h', 'D_2_S_h', 'D_2_A_v', 'D_2_S_v',
                        'D_3_A_h', 'D_3_S_h', 'D_3_A_v', 'D_3_S_v',
                        'D_4_A_h', 'D_4_S_h', 'D_4_A_v', 'D_4_S_v',
                        'D_5_A_h', 'D_5_S_h', 'D_5_A_v', 'D_5_S_v',
                        'QB_A_h', 'QB_S_h', 'QB_A_v', 'QB_S_v', 'QB_X', 'QB_Y'
                        ]

        self.categorical_columns = ['IsFirstAndTenRB', 'IsQB', 'IsGoingLong', 'IsGoingShort', 'IsGoingWide',
                                    'IsGoingBackwards', 'Season_2017', 'Season_2018', 'Season_2019',
                                    'IsRunningForTD', 'Off_Formation_ACE',
                                    'Off_Formation_EMPTY',
                                    'Off_Formation_I_FORM',
                                    'Off_Formation_JUMBO',
                                    'Off_Formation_PISTOL',
                                    'Off_Formation_SHOTGUN',
                                    'Off_Formation_SINGLEBACK',
                                    'Off_Formation_WILDCAT']

    def get_features(self):
        #                    ,'CentripedidalForceToD', 'CentripedidalForceToD','ForceFromPrevious','AngularVelocityChange'
        #                    ,'Orientation2', 'Dir2'
        #                    'DITB_A_mean','DITB_S_mean', 'D_A_mean', 'D_S_mean'
        #                    'ScoreUrgency'
        #                    'Dir','PlayerHeight','PlayerWeight', 'S',
        #                        'Distance', 'Week','Quarter','X_std'
        #                        'FullGameClock_std','OffenseScoreDelta','Down','Def_DL','Def_LB','Def_DB',
        #                        'DefendersInTheBox','DITB_Centroid_X','DITB_Centroid_Y','DITB_Spread_X','DITB_Spread_Y',
        #                        'DITB_Centroid_X_Dis_to_rusher','DITB_Centroid_Y_Dis_to_rusher','Off_RB','Off_TE','Off_WR',
        #                        'DistanceToQB','MinTimeToTackle',
        #                        'Voronoi_rusher_area','Voronoi_DITB_area','Voronoi_Offense_area','Voronoi_Defense_area',
        #                        'LargestYForOffenseInBox',
        #                    'D_Dis_to_R_Mean_in_1','D_Dis_to_R_Min_in_1','D_Dis_to_R_Mean_in_2','D_Dis_to_R_Min_in_2'
        return self.columns

    def categorical_features(self):
        #         return ['IsGoingBackwards']
        return self.categorical_columns

    def get_input_shape(self):
        return len(self.get_features() + self.categorical_features())


class App:
    def __init__(self):
        self.settings = Settings()
        self.n = Network(self.settings)

    def run(self):
        train_df = pd.read_csv('../resources/train.csv', low_memory=False)
        self.n.setup(train_df)
        self.n.add_splitter()
        self.n.train()

    def test_features(self):
        train_df = pd.read_csv('../resources/train.csv', low_memory=False)
        self.n.setup(train_df)
        test_df = pd.read_csv('../resources/train.csv', low_memory=False)
        test_df_3_DITB = test_df.loc[test_df.DefendersInTheBox == 3].iloc[0:22]
        test_df_6_DITB = test_df.loc[test_df.DefendersInTheBox == 6].iloc[0:22]
        test_df_9_DITB = test_df.loc[test_df.DefendersInTheBox == 9].iloc[0:22]

        self.n.predict(test_df_3_DITB)

    def dummy_predict(self, test_df):
        pred = self.n.predict(test_df)
        pred = np.round(pred, decimals=4)
        columns = np.core.defchararray.add('Yards', np.arange(-99, 100).astype(str))
        pred_df = pd.DataFrame(data=pred, columns=columns)
        display(pred_df)


    def predict_dummy_eval(self):
        print("*********\nStarting Prediction\n\n")
        train_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
        last_games = train_df[train_df['GameId'].isin(Trainer.split_into_last_games(train_df))]
        y_true = make_result_set_dummy(last_games.loc[last_games.NflId == last_games.NflIdRusher, 'Yards'])
        y_pred = self.n.predict(last_games)
        crps = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])
        print("Total score: {}".format(crps))

app = App()
app.test_features()