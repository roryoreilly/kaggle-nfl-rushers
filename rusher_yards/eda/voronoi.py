import pandas as pd
pd.set_option('max_columns', 100)
import os
import sys
import numpy as np
from scipy.spatial import distance
import scipy as sp
import matplotlib.pyplot as plt

from rusher_yards.feature_extractor import FeatureExtractor
from rusher_yards.persistent_features import PersistentFeatures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

train_df = pd.read_csv('../../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
pf = PersistentFeatures(train_df)
pf.build_features()
fe = FeatureExtractor(train_df, pf)
fe.run()
pf.attach_columns(fe.features.columns)
nfl_copy = fe.nfl.copy()
features_copy = fe.features.copy()

# towers = np.array([(0,0),(0,1),(1,0),(1,1)])
# bounding_box = np.array([-10, 10, -10, 10]) # [x_min, x_max, y_min, y_max]

def voronoi_vertices(points_center, bounding_box = np.array([0, 120, -160 / 6, 160 / 6])):
    eps = sys.float_info.epsilon * 1000000

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
                       np.append(np.append(points_left, points_right,axis=0),
                                 np.append(points_down, points_up, axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)

    point_regions = vor.point_region[0:22]
    all_regions = np.array(vor.regions)
    all_vertices = np.array(vor.vertices)
    vertices = [ all_vertices[r] for r in all_regions[point_regions] ]

    return pd.Series(vertices)

def display_voronoi_play(play):
    points = np.array(play[['X_std', 'Y_std']])

    fig = plt.figure()
    ax = fig.gca()
    # Plot initial points
    ax.plot(points[:, 0], points[:, 1], 'b.')
    # Plot ridges
    for vertices in play['Vertices']:
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-')

    for vertices in play.loc[play.IsOnOffense, 'Vertices']:
        plt.fill(*zip(*vertices), "lightblue")

    for vertices in play.loc[~play.IsOnOffense, 'Vertices']:
        plt.fill(*zip(*vertices), "grey")

    plt.fill(*zip(*play.loc[play.IsRusher, 'Vertices'].values[0]), "pink")

    ax.set_xlim([-10, 130])
    ax.set_ylim([-30, 30])
    plt.savefig("bounded_voronoi.png")

# How to display a single play for voronoi
play = nfl_copy.loc[nfl_copy.PlayId == 20181230022972]
points = np.array(nfl_copy.loc[nfl_copy.PlayId == 20181230022972, ['X_std', 'Y_std']])
rusher = play.loc[play.IsRusher]
bounding_box = np.array([rusher['X_std'], rusher['X_std']+15, -10, 10])
vertices_series = voronoi_vertices(points, bounding_box)  # [x_min, x_max, y_min, y_max])
play['Vertices'] = vertices_series.values
display_voronoi_play(play)

# Get the set of voronoi vertices and volumes
# plays = nfl_copy[0:]
# print('Testing how long this will take')
# t0 = timeit.default_timer()
# vertices = plays.groupby('PlayId')[['X_std', 'Y_std']].apply(
#     lambda play: voronoi_vertices(np.array(play))).unstack()
# volumes = vertices.apply(lambda v: ConvexHull(v).volume)
# plays['Voronoi_vertices'] = vertices.values
# plays['Voronoi_volume'] = volumes.values
# t1 = timeit.default_timer()
# print('It took {} seconds'.format(t1-t0))

