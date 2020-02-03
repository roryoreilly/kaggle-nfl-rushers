class Settings:
    def __init__(self):
        self.epochs = 1
        self.batch_size = 256
        self.model_type = 'lgbm'
        self.load_model_if_exists = False

    def get_features(self):
        columns = ['S','A','Dis','Orientation','Dir','PlayerHeight','PlayerWeight','X_std','Y_std',
                   'Dir_rad','Dir_std','S_std','Distance','A_horizontal','A_Vertical',
                   'S_std_horizontal','S_std_vertical','RusherMeanYardsSeason','Back_from_Scrimmage',
                   'Week','Quarter','GameClock_std','FullGameClock_std','OffenseScoreDelta','YardLine_std','Down',
                   'Def_DL','Def_LB','Def_DB','DefendersInTheBox','DITB_Centroid_X','DITB_Centroid_Y','DITB_Spread_X',
                   'DITB_Spread_Y','DITB_Centroid_X_Dis_to_rusher','DITB_Centroid_Y_Dis_to_rusher','Off_RB','Off_TE',
                   'Off_WR','DistanceToQB','MinTimeToTackle','D_Dis_to_R_Min','D_Dis_to_R_Max','D_Dis_to_R_Mean',
                   'D_Dis_to_R_Std','Dis_to_R_Min','Dis_to_R_Max','Dis_to_R_Mean','Dis_to_R_Std','Voronoi_rusher_area',
                   'Voronoi_DITB_area','Voronoi_Offense_area','Voronoi_Defense_area','LargestYForOffenseInBox'
                   ]
        return columns

    def categorical_features(self):
        return ['IsFirstAndTenRB', 'IsGoingLong', 'IsGoingShort', 'IsQB', 'IsGoingWide', 'IsGoingBackwards']