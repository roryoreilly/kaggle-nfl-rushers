import math
import os
import sys

import numpy as np
import pandas as pd
import scipy as sp

from rusher_yards.persistent_features import PersistentFeatures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

"""
Possibile Features:
    - Basic
    - Rusher
        ✓ X_std
        ✓ S_std
            - Horizontal
            - Vertical
        ✓ A
        ✓ Dis
        ✓ Orientation
        ✓ NflId
        ✓ mean_yards_season
        ✓ mean_yards
        ✓ Position
    - Defense
        ✓ DefendersInTheBox
            ✓ Centroid
            o Circumcenter
            o Radius
            o Largest Gap
                o Horizontal
                o Vertical
            ✓ Spread Y
        ✓ DL
        ✓ LB
    - Offense
        ✓ OffenseFormation
        ✓ RB
        ✓ TE
        ✓ WR
        o num_team_members_in_hole (defined rusher x -> defense x, defense min y defense max y
    - Game
        ✓ Week
        ✓ Season hot ohe
        ✓ Quarter
        ✓ GameClock_std
        ✓ FullGameClock
        ✓ OffensiveScoreDelta
        ✓ Yardline_std
        ✓ Down
        ✓ IsFirstAndTen
        o Weather
        o Humidity
        o Stadium
        o Wind
        o is_going_for_goal
        o distance_to_los
    - Play
        ✓ min_time_to_tackle
        o is_going_wide
        o players_in_cone_of_movement
        o num_defense_matched
        o num_defense_in_the_hole_matched
        ✓ rusher distance from QB
        o num_defense_x_distance_away
    - Voroni
        o size of rusher
        o size of defenseinthehole
        o size of areas behind defense
        o 
"""


def stringtomins(x):
    h, m, s = map(int, x.split(':'))
    return (h * 60) + m + (s / 60)

# Mean yards of rushers. Will remove when 2019 has been added tio train
rusher_yards = {
    2019: {"Dalvin Cook": [203, 4.9], "Christian McCaffrey": [185, 5.3], "Nick Chubb": [174, 5.3],
           "Derrick Henry": [187, 4.4], "Leonard Fournette": [174, 4.8], "Josh Jacobs": [168, 4.8],
           "Ezekiel Elliott": [178, 4.4], "Chris Carson": [175, 4.4], "Marlon Mack": [178, 4.2],
           "Carlos Hyde": [149, 4.7], "Lamar Jackson": [106, 6.6], "Mark Ingram": [123, 5.0], "Aaron Jones": [135, 4.4],
           "Phillip Lindsay": [118, 4.9], "Jordan Howard": [119, 4.4], "Matt Breida": [99, 5.3],
           "Adrian Peterson": [115, 4.3], "Sony Michel": [144, 3.3], "David Montgomery": [129, 3.6],
           "Le'Veon Bell": [143, 3.1], "Frank Gore": [111, 4.0], "Joe Mixon": [131, 3.3], "Todd Gurley": [104, 4.1],
           "Ronald Jones": [103, 4.0], "Saquon Barkley": [101, 4.0], "Alvin Kamara": [90, 4.4],
           "Alexander Mattison": [79, 4.9], "James Conner": [97, 3.9], "Royce Freeman": [93, 4.0],
           "Devonta Freeman": [107, 3.5], "LeSean McCoy": [72, 5.2], "Latavius Murray": [85, 4.4],
           "Tevin Coleman": [83, 4.3], "Kyler Murray": [59, 5.9], "Austin Ekeler": [90, 3.8],
           "Miles Sanders": [76, 4.4], "Peyton Barber": [94, 3.4], "Devin Singletary": [48, 6.4],
           "Kerryon Johnson": [92, 3.3], "Raheem Mostert": [55, 5.6], "David Johnson": [82, 3.7],
           "Damien Williams": [79, 3.8], "Melvin Gordon": [86, 3.5], "Chase Edmonds": [58, 5.1],
           "Jamaal Williams": [65, 4.5], "Duke Johnson": [54, 5.3], "Deshaun Watson": [52, 5.4],
           "Gus Edwards": [63, 4.4], "Josh Allen": [67, 4.1], "Gardner Minshew": [42, 5.6], "Tony Pollard": [49, 4.6],
           "Daniel Jones": [32, 6.5], "Russell Wilson": [44, 4.6], "Mark Walton": [53, 3.8], "J.D. McKissic": [30, 5.9],
           "Dak Prescott": [27, 6.5], "Rashaad Penny": [34, 4.9], "Jordan Wilkins": [27, 6.1],
           "Malcolm Brown": [42, 3.9], "Ty Johnson": [44, 3.5], "Justin Jackson": [20, 7.5], "Kenyan Drake": [25, 5.8],
           "Carson Wentz": [37, 3.9], "Jameis Winston": [31, 4.5], "Rex Burkhead": [31, 4.3],
           "DeAndre Washington": [38, 3.5], "Marcus Mariota": [24, 5.4], "Darrell Henderson": [33, 3.7],
           "Benny Snell": [28, 4.2], "Kalen Ballage": [55, 2.1], "Jacoby Brissett": [37, 3.0],
           "Wayne Gallman": [28, 3.9], "Aaron Rodgers": [27, 4.0], "Ito Smith": [22, 4.8], "James White": [32, 3.3],
           "Brian Hill": [28, 3.6], "Unknown": [0, 4.2]},
    2018: {"Ezekiel Elliott": [304, 4.7], "Saquon Barkley": [261, 5.0], "David Johnson": [258, 3.6],
           "Todd Gurley": [256, 4.9], "Adrian Peterson": [251, 4.2], "Jordan Howard": [250, 3.7],
           "Chris Carson": [247, 4.7], "Joe Mixon": [237, 4.9], "Peyton Barber": [234, 3.7],
           "Christian McCaffrey": [219, 5.0], "James Conner": [215, 4.5], "Derrick Henry": [215, 4.9],
           "Lamar Miller": [210, 4.6], "Sony Michel": [209, 4.5], "Marlon Mack": [195, 4.7], "Alvin Kamara": [194, 4.6],
           "Nick Chubb": [192, 5.2], "Phillip Lindsay": [192, 5.4], "Kareem Hunt": [181, 4.6],
           "Melvin Gordon": [175, 5.1], "Doug Martin": [172, 4.2], "Tevin Coleman": [167, 4.8],
           "LeSean McCoy": [161, 3.2], "Frank Gore": [156, 4.6], "Dion Lewis": [155, 3.3],
           "LeGarrette Blount": [154, 2.7], "Matt Breida": [153, 5.3], "Alfred Blue": [150, 3.3],
           "Lamar Jackson": [147, 4.7], "Isaiah Crowell": [143, 4.8], "atavius Murray": [140, 4.1],
           "Mark Ingram": [138, 4.7], "Gus Edwards": [137, 5.2], "Dalvin Cook": [133, 4.6],
           "Leonard Fournette": [133, 3.3], "Aaron Jones": [133, 5.5], "Royce Freeman": [130, 4.0],
           "Jamaal Williams": [121, 3.8], "Josh Adams": [120, 4.3], "Kenyan Drake": [120, 4.5],
           "Kerryon Johnson": [118, 5.4], "Chris Ivory": [115, 3.3], "Alex Collins": [114, 3.6],
           "Mike Davis": [112, 4.6], "Alfred Morris": [111, 3.9], "Austin Ekeler": [106, 5.2],
           "T.J. Yeldon": [104, 4.0], "Cam Newton": [101, 4.8], "Tarik Cohen": [99, 4.5], "Deshaun Watson": [99, 5.6],
           "James White": [94, 4.5], "Elijah McGuire": [92, 3.0], "Marshawn Lynch": [90, 4.2], "Ito Smith": [90, 3.5],
           "Josh Allen": [89, 7.1], "Wendell Smallwood": [87, 4.2], "Nyheim Hines": [85, 3.7],
           "Rashaad Penny": [85, 4.9], "Bilal Powell": [80, 4.3], "Dak Prescott": [75, 4.1], "Corey Clement": [68, 3.8],
           "Mitchell Trubisky": [68, 6.2], "Russell Wilson": [67, 5.6], "Jeff Wilson": [66, 4.0],
           "Marcus Mariota": [64, 5.6], "Kenneth Dixon": [60, 5.6], "Chase Edmonds": [60, 3.5],
           "Patrick Mahomes": [60, 4.5], "Jordan Wilkins": [60, 5.6], "Blake Bortles": [58, 6.3],
           "Carlos Hyde": [58, 3.3], "Rex Burkhead": [57, 3.3], "Giovani Bernard": [56, 3.8],
           "Jaylen Samuels": [56, 4.6], "Jalen Richard": [55, 4.7], "Zach Zenner": [55, 4.8],
           "Marcus Murphy": [52, 4.8], "Wayne Gallman": [51, 3.5], "Spencer Ware": [51, 4.8],
           "Justin Jackson": [50, 4.1], "Damien Williams": [50, 5.1], "Jameis Winston": [49, 5.7],
           "Andrew Luck": [46, 3.2], "Jay Ajayi": [45, 4.1], "Kirk Cousins": [44, 2.8], "Sam Darnold": [44, 3.1],
           "Rod Smith": [44, 2.9], "C.J. Anderson": [43, 7.0], "Malcolm Brown": [43, 4.9], "Jared Goff": [43, 2.5],
           "Aaron Rodgers": [43, 6.3], "Chris Thompson": [43, 4.1], "Cordarrelle Patterson": [42, 5.4],
           "Javorius Allen": [41, 2.7], "Alex Smith": [41, 4.1], "Duke Johnson": [40, 5.0], "Theo Riddick": [40, 4.3],
           "Baker Mayfield": [39, 3.4], "Trenton Cannon": [38, 3.0], "Taysom Hill": [37, 5.3],
           "Kalen Ballage": [36, 5.3], "Ryan Fitzpatrick": [36, 4.2], "Devontae Booker": [34, 5.4],
           "Raheem Mostert": [34, 7.7], "Carson Wentz": [34, 2.7], "Jacquizz Rodgers": [33, 3.2],
           "Matt Ryan": [33, 3.8], "Ryan Tannehill": [32, 4.5], "Drew Brees": [31, 0.7],
           "Ben Roethlisberger": [31, 3.2], "DeAndre Washington": [30, 3.8], "Unknown": [0, 4.355]},
    2017: {"Le'Veon Bell": [321, 4.0], "LeSean McCoy": [287, 4.0], "Melvin Gordon": [284, 3.9],
           "Todd Gurley": [279, 4.7], "Jordan Howard": [276, 4.1], "Kareem Hunt": [272, 4.9],
           "Leonard Fournette": [268, 3.9], "Frank Gore": [261, 3.7], "C.J. Anderson": [245, 4.1],
           "Ezekiel Elliott": [242, 4.1], "Carlos Hyde": [240, 3.9], "Lamar Miller": [238, 3.7],
           "Mark Ingram": [230, 4.9], "Latavius Murray": [216, 3.9], "Alex Collins": [212, 4.6],
           "Marshawn Lynch": [207, 4.3], "Isaiah Crowell": [206, 4.1], "Jonathan Stewart": [198, 3.4],
           "Devonta Freeman": [196, 4.4], "DeMarco Murray": [184, 3.6], "Dion Lewis": [180, 5.0],
           "Joe Mixon": [178, 3.5], "Bilal Powell": [178, 4.3], "Derrick Henry": [176, 4.2],
           "Samaje Perine": [175, 3.4], "LeGarrette Blount": [173, 4.4], "Orleans Darkwa": [171, 4.4],
           "Ameer Abdullah": [165, 3.3], "Tevin Coleman": [156, 4.0], "Javorius Allen": [153, 3.9],
           "Jamaal Williams": [153, 3.6], "Jerick McKinnon": [150, 3.8], "Cam Newton": [139, 5.4],
           "Doug Martin": [138, 2.9], "Kenyan Drake": [133, 4.8], "Adrian Peterson": [129, 3.5],
           "Alvin Kamara": [120, 6.1], "Kerwynn Williams": [120, 3.6], "Christian McCaffrey": [117, 3.7],
           "Alfred Morris": [115, 4.8], "Chris Ivory": [112, 3.4], "Wayne Gallman": [111, 4.3],
           "Peyton Barber": [108, 3.9], "Giovani Bernard": [105, 4.4], "Matt Breida": [105, 4.4],
           "Mike Gillislee": [104, 3.7], "Matt Forte": [103, 3.7], "Russell Wilson": [95, 6.2],
           "Marlon Mack": [93, 3.8], "Elijah McGuire": [88, 3.6], "Tarik Cohen": [87, 4.3], "Theo Riddick": [84, 3.4],
           "Tyrod Taylor": [84, 5.1], "Duke Johnson": [82, 4.2], "Aaron Jones": [81, 5.5], "Devontae Booker": [79, 3.8],
           "D'Onta Foreman": [78, 4.2], "DeShone Kizer": [77, 5.4], "Corey Clement": [74, 4.3],
           "Dalvin Cook": [74, 4.8], "Alfred Blue": [71, 3.7], "Ty Montgomery": [71, 3.8], "Jay Ajayi": [70, 5.8],
           "Jamaal Charles": [69, 4.3], "Eddie Lacy": [69, 2.6], "Mike Davis": [68, 3.5], "Mike Tolbert": [66, 3.7],
           "Rex Burkhead": [64, 4.1], "Jacquizz Rodgers": [64, 3.8], "Chris Thompson": [64, 4.6],
           "Carson Wentz": [64, 4.7], "Jacoby Brissett": [63, 4.1], "Malcolm Brown": [63, 3.9], "Rob Kelley": [62, 3.1],
           "Marcus Mariota": [60, 5.2], "Alex Smith": [60, 5.9], "Tavon Austin": [59, 4.6], "Thomas Rawls": [58, 2.7],
           "Blake Bortles": [57, 5.6], "Dak Prescott": [57, 6.3], "DeAndre Washington": [57, 2.7],
           "Jalen Richard": [56, 4.9], "Rod Smith": [55, 4.2], "Chris Carson": [49, 4.2], "Kirk Cousins": [49, 3.7],
           "T.J. Yeldon": [49, 5.2], "Austin Ekeler": [47, 5.5], "Wendell Smallwood": [47, 3.7],
           "J.D. McKissic": [46, 4.1], "Damien Williams": [46, 3.9], "Chris Johnson": [45, 2.5],
           "Shane Vereen": [45, 3.6], "James White": [43, 4.0], "Tion Green": [42, 3.9], "Paul Perkins": [41, 2.2],
           "Mitchell Trubisky": [41, 6.0], "Case Keenum": [40, 4.0], "Terrance West": [39, 3.5],
           "Andy Dalton": [38, 2.6], "Jeremy Hill": [37, 3.1], "Josh McCown": [37, 3.4], "Brett Hundley": [36, 7.5],
           "Deshaun Watson": [36, 7.5], "Branden Oliver": [35, 2.4], "Drew Brees": [33, 0.4],
           "Jameis Winston": [33, 4.1], "James Conner": [32, 4.5], "Matt Ryan": [32, 4.5], "Elijhaa Penny": [31, 4.0],
           "Trevor Siemian": [31, 4.1], "Corey Grant": [30, 8.3], "Terron Ward": [30, 4.3], "Unknown": [0, 3.868]}
}

class FeatureExtractor:
    PICKLE_PATH = '../resources/features.pkl'

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
        if self.is_train and self.load_if_exists():
            print("Loaded features for train set from pickle.")
            return
        self.attach_persistent_features()
        self.rusher_features()
        self.splitting_features()
        self.game_features()
        self.defense_features()
        self.offense_features()
        self.play_features()
        self.voronoi_features()
        self.features = self.features.fillna(self.features.mean())
        if self.is_train:
            self.features.to_pickle(self.PICKLE_PATH)

    def load_if_exists(self):
        if os.path.exists(self.PICKLE_PATH):
            self.features = pd.read_pickle(self.PICKLE_PATH)
            return True
        return False

    def get_useable_features(self):
        base_columns = ['GameId']
        columns = base_columns + self.settings.categorical_features() + self.settings.get_features()
        return self.features[columns]

    def get_features_with_yards(self):
        return self.get_useable_features().join(self.rushers['Yards'])

    def attach_missing_columns(self, persistent_columns):
        missing_cols = set(persistent_columns) - set(self.features.columns)
        for c in missing_cols:
            self.features[c] = 0

    def attach_persistent_features(self):
        self.features.reset_index().merge(self.persistent_features.rusher_features, on='NflId').set_index('PlayId')

    def splitting_features(self):
        self.features['IsFirstAndTenRB'] = self.rushers['IsFirstAndTen']
        self.features.loc[(self.rushers.Position != 'RB'), 'IsFirstAndTenRB'] = 0
        self.features['IsGoingLong'] = 0
        self.features.loc[(self.rushers.Distance > 14.0), 'IsGoingLong'] = 1
        self.features['IsGoingShort'] = 0
        self.features.loc[(self.rushers.Distance < 5.0), 'IsGoingShort'] = 1
        self.features['IsQB'] = 0
        self.features.loc[(self.rushers.Position == 'QB'), 'IsQB'] = 1
        self.features['IsGoingWide'] = (np.abs(self.features['Dir_std']) > 0.95) & (
                np.abs(self.features['Dir_std']) < 1.57)
        self.features['IsGoingBackwards'] = (np.abs(self.features['Dir_std']) > 1.57)

    def play_features(self):
        self.features['DistanceToQB'] = np.sqrt(np.sum(
            self.nfl.loc[(self.nfl.Position == 'QB') | self.nfl.IsRusher, ['PlayId', 'X_std', 'Y_std']].groupby(
                'PlayId').agg(['min', 'max']).diff(axis=1).drop([('X_std', 'min'), ('Y_std', 'min')], axis=1) ** 2,
            axis=1))

        self.features['MinTimeToTackle'] = np.array(self.nfl.loc[self.nfl.IsRusher, 'S']) / \
                                           np.array(self.nfl.loc[~self.nfl.IsOnOffense & (
                                                   self.nfl.RankDisFromPlayStart == 1), 'S'])
        self.features['MinTimeToTackle'] = self.features['MinTimeToTackle'].fillna(100)
        self.distances_to_rusher_features()

    def distances_to_rusher_features(self):
        nfl_copy = self.nfl.copy()
        nfl_copy['Rusher_X_std'] = nfl_copy['PlayId'].map(self.rushers['X_std'].to_dict())
        nfl_copy['Rusher_Y_std'] = nfl_copy['PlayId'].map(self.rushers['Y_std'].to_dict())
        distances_to_rusher = (nfl_copy.X_std - nfl_copy.Rusher_X_std) ** 2 + (
                nfl_copy.Y_std - nfl_copy.Rusher_Y_std) ** 2
        nfl_copy['DisToRusher'] = np.sqrt(distances_to_rusher)
        self.features['D_Dis_to_R_Min'] = nfl_copy.loc[~nfl_copy.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher'].min()
        self.features['D_Dis_to_R_Max'] = nfl_copy.loc[~nfl_copy.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher'].max()
        self.features['D_Dis_to_R_Mean'] = nfl_copy.loc[~nfl_copy.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher'].mean()
        self.features['D_Dis_to_R_Std'] = nfl_copy.loc[~nfl_copy.IsOnOffense].groupby(['PlayId'])[
            'DisToRusher'].std()
        self.features['Dis_to_R_Min'] = nfl_copy.loc[nfl_copy.IsRusher == 0].groupby(['PlayId'])[
            'DisToRusher'].min()
        self.features['Dis_to_R_Max'] = nfl_copy.loc[nfl_copy.IsRusher == 0].groupby(['PlayId'])[
            'DisToRusher'].max()
        self.features['Dis_to_R_Mean'] = nfl_copy.loc[nfl_copy.IsRusher == 0].groupby(['PlayId'])[
            'DisToRusher'].mean()
        self.features['Dis_to_R_Std'] = nfl_copy.loc[nfl_copy.IsRusher == 0].groupby(['PlayId'])[
            'DisToRusher'].std()

    def defense_features(self):
        self.features['Def_DL'] = np.array(self.rushers['DefensePersonnel'].str[:1], dtype='int8')
        self.features['Def_LB'] = np.array(self.rushers['DefensePersonnel'].str[6:7], dtype='int8')
        self.features['Def_DB'] = np.array(self.rushers['DefensePersonnel'].str[12:13], dtype='int8')
        self._defenders_in_the_box()

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
        self.features.loc[self.rushers.PossessionTeam != self.rushers.HomeTeamAbbr, 'OffenseScoreDelta'] \
            = -1 * self.features.loc[self.rushers.PossessionTeam != self.rushers.HomeTeamAbbr, 'OffenseScoreDelta']

        self.features['YardLine_std'] = self.rushers['YardLine_std']
        self.features['Down'] = self.rushers['Down']

    def rusher_features(self):
        rushers_features = self.rushers[['S', 'A', 'Dis', 'Orientation', 'Dir', 'PlayerHeight',
                                         'PlayerWeight', 'X_std', 'Y_std', 'Dir_rad', 'Dir_std',
                                         'S_std', 'Distance', 'YardLine']].copy(deep=True)
        rushers_features['PlayerHeight'] = rushers_features['PlayerHeight'] \
            .apply(lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))

        self.features = self.features.join(rushers_features)
        self._rusher_position_ohe()
        self.features['A_horizontal'] = self.rushers['A'] * np.cos(self.rushers['Dir_rad'])
        self.features['A_Vertical'] = self.rushers['A'] * np.sin(self.rushers['Dir_rad'])
        self.features['S_horizontal'] = self.rushers['S'] * np.cos(self.rushers['Dir_rad'])
        self.features['S_vertical'] = self.rushers['S'] * np.sin(self.rushers['Dir_rad'])
        self.features['S_std_horizontal'] = self.rushers['S_std'] * np.cos(self.rushers['Dir_rad'])
        self.features['S_std_vertical'] = self.rushers['S_std'] * np.sin(self.rushers['Dir_rad'])
        self.features['RusherMeanYardsSeason'] = self.rushers.apply(self.get_rusher_mean_yards, index=1, axis=1)
        self.features['Back_from_Scrimmage'] = self.rushers['YardLine_std'] - self.rushers['X_std']

    def get_rusher_mean_yards(self, row, index):
        try:
            return rusher_yards[row['Season']][row['DisplayName']][index]
        except:
            return rusher_yards[row['Season']]['Unknown'][index]

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
        self.nfl['IsFirstAndTen'] = 1
        self.nfl.loc[(self.nfl.Distance != 10.0) | (self.nfl.Down != 1), 'IsFirstAndTen'] = 0

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
        self.nfl['Orientation'] = (self.nfl['Orientation'] * math.pi / 180) - math.pi

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


def make_result_set(rushers, model_type='nn'):
    if model_type == 'lgbm':
        return make_result_set_lgbm(rushers)
    return make_result_set_nn(rushers)

def make_result_set_lgbm(rushers):
    results = np.zeros((rushers.shape[0]))
    for i, yard in enumerate(rushers.Yards):
        results[i] = yard
    return results

def make_result_set_nn(rushers):
    results = np.zeros((rushers.shape[0], 199))
    for i, yard in enumerate(rushers.Yards):
        results[i, yard + 99:] = np.ones(shape=(1, 100 - yard))
    return results


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    np.set_printoptions(linewidth=1000)
    pd.set_option('display.max_columns', 50)
    train_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
    pf = PersistentFeatures(train_df)
    fe = FeatureExtractor(train_df, pf)
    fe.run()
    print(train_df.head())
    print("\n********************\n")
    print(fe.features.head())
