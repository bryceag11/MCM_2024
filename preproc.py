from itertools import groupby
import numpy as np
import pandas as pd

class MomentumFeatures():
    def __init__(self, df):
        self.df = df
        self.trans_df = pd.DataFrame({})
        self.current_index = 0
        self.df_last_ind = len(df) - 1
        # values needed as rows are processed iteratively
        self.current_index = 0
        self.current_point_streak = {'p1': 0, 'p2': 0}
        self.total_point_streaks = {'p1': 0, 'p2': 0}
        self.current_game_streak = {'p1': 0, 'p2': 0}
        self.total_game_streaks = {'p1': 0, 'p2': 0}
        self.current_set_streak = {'p1': 0, 'p2': 0}
        self.total_set_streaks = {'p1': 0, 'p2': 0}

    def point_ratio(self, index):

        # Return total_points 
        point_no = self.df.iloc[index]['point_no']
        p1_running, p2_running = self.running_ratio(index)
        if index > 0: 
            p1_points_won = self.df.iloc[index-1]['p1_points_won'] + p1_running
            p2_points_won = self.df.iloc[index-1]['p2_points_won'] + p2_running
        else:
            p1_points_won = self.df.iloc[index]['p1_points_won'] = p1_running
            p2_points_won = self.df.iloc[index]['p2_points_won'] = p2_running


        p1_won_ratio = p1_points_won / point_no
        p2_won_ratio = p2_points_won / point_no

            
        self.trans_df.loc[index, 'p1_won_ratio'] = p1_won_ratio
        self.trans_df.loc[index, 'p2_won_ratio'] = p2_won_ratio
        
    def running_ratio(self, index):
        point_victor = self.df.iloc[index]['point_victor']
        server = self.df.iloc[index]['server']
        p1_running = 0
        p2_running = 0
        if point_victor == 2:
            if server == 2:
                p2_running = 2
            else:
                p2_running = 1
        else:
            if server == 1:
                p1_running = 2
            else:
                p1_running = 1

        return p1_running, p2_running

    def calculate_win_streaks_points(self, index):
        point_victor = self.df.iloc[index]['point_victor']
 
        if point_victor in [1, 2]:
            player_key = 'p1' if point_victor == 1 else 'p2'
            opponent_key = 'p2' if point_victor == 1 else 'p1'
            
            self.current_point_streak[player_key] += 1

            # if the other team member scores update the streak
            if self.current_point_streak[opponent_key] >= 2:
                self.total_point_streaks[opponent_key] += 1
            self.current_point_streak[opponent_key] = 0
            
            # a new set starts update the streak variable
            if index == self.df_last_ind or self.df.iloc[index]['game_no'] != self.df.iloc[index+1]['game_no']:
                if self.current_point_streak[player_key] >= 2:
                    self.total_point_streaks[player_key] += 1
                
                self.current_point_streak[opponent_key] = 0
                self.current_point_streak[player_key] = 0
                    
        self.trans_df.loc[index, 'p1_point_streaks'] = self.total_point_streaks['p1']
        self.trans_df.loc[index, 'p2_point_streaks'] = self.total_point_streaks['p2']
        
    
    def calculate_win_streaks_games(self, index):
        game_victor = self.df.iloc[index]['game_victor']

        if game_victor in [1, 2]:
            player_key = 'p1' if game_victor == 1 else 'p2'
            opponent_key = 'p2' if game_victor == 1 else 'p1'

            self.current_game_streak[player_key] += 1

            if self.current_game_streak[player_key] >= 2:
                # Update total game streaks only when the streak is starting
                if self.current_game_streak[player_key] == 2:
                    self.total_game_streaks[player_key] += 1

            # if the other team member scores update the streak
            if self.current_game_streak[opponent_key] >= 2:
                self.total_game_streaks[opponent_key] += 1
            self.current_game_streak[opponent_key] = 0

            # A new set starts, update the streak variable
            if index == self.df_last_ind or self.df.iloc[index]['set_no'] != self.df.iloc[index + 1]['set_no']:
                if self.current_game_streak[player_key] >= 2:
                    self.total_game_streaks[player_key] += 1

                self.current_game_streak[player_key] = 0
                self.current_game_streak[opponent_key] = 0

        self.trans_df.loc[index, 'p1_game_streaks'] = self.total_game_streaks['p1']
        self.trans_df.loc[index, 'p2_game_streaks'] = self.total_game_streaks['p2']

    
        
    def calculate_win_streaks_sets(self, index):
        set_victor = self.df.iloc[index]['set_victor']
        
        # end of set
        if set_victor in [1, 2]:
            player_key = 'p1' if set_victor == 1 else 'p2'
            opponent_key = 'p2' if set_victor == 1 else 'p1'
            
            if self.current_set_streak[player_key] > 0:
                self.current_set_streak[player_key] += 1
            else:
                self.current_set_streak[player_key] = 1
        
            self.current_set_streak[opponent_key] = 0

            if index == self.df_last_ind or self.df.iloc[index]['set_no'] != self.df.iloc[index + 1]['set_no']:
                if self.current_set_streak[player_key] >= 2:
                    self.total_set_streaks[player_key] += 1
                if self.current_set_streak[opponent_key] >= 2:
                    self.total_set_streaks[opponent_key] += 1
        
        self.trans_df.loc[index, 'p1_set_streaks'] = self.total_set_streaks['p1']
        self.trans_df.loc[index, 'p2_set_streaks'] = self.total_set_streaks['p2']

    def pressure_index(self, row):
        self.df['p1_break_pt_efficiency'] = np.where(self.df['p1_break_pt'].cumsum() != 0, self.df['p1_break_pt_won'].cumsum() / self.df['p1_break_pt'].cumsum(), 0)
        self.df['p2_break_pt_efficiency'] = np.where(self.df['p2_break_pt'].cumsum() != 0, self.df['p2_break_pt_won'].cumsum() / self.df['p2_break_pt'].cumsum(), 0)

        self.df['p1_break_pt_saved_rat'] = np.where(self.df['p2_break_pt_efficiency'].cumsum() != 0, 1 - self.df['p2_break_pt_efficiency'], 0)
        self.df['p2_break_pt_saved_rat'] = np.where(self.df['p1_break_pt_efficiency'].cumsum() != 0, 1 - self.df['p1_break_pt_efficiency'], 0)

        self.df['speed_mph_average'] = self.df['speed_mph'].cumsum() / (self.df.index + 1)
        self.df['rally_count_average'] = self.df['rally_count'].cumsum() / (self.df.index + 1)

        self.df['p1_high_pressure'] = np.where(((self.df['p1_break_pt'] == 1)) & (self.df['speed_mph'] > self.df['speed_mph_average'])
                                        & (self.df['rally_count'] > self.df['rally_count_average']), 1, 0)

        self.df['p2_high_pressure'] = np.where(((self.df['p2_break_pt'] == 1)) & (self.df['speed_mph'] > self.df['speed_mph_average'])
                                        & (self.df['rally_count'] > self.df['rally_count_average']), 1, 0)

        self.df['p1_high_pressure_win'] = np.where(self.df['point_victor'] == 1, self.df['p1_high_pressure'], 0)
        self.df['p2_high_pressure_win'] = np.where(self.df['point_victor'] == 2, self.df['p2_high_pressure'], 0)

        self.df['p1_high_pressure_win_ratio'] = np.where(self.df['p1_high_pressure'].cumsum() != 0,
                                                    self.df['p1_high_pressure_win'].cumsum() / self.df['p1_high_pressure'].cumsum(), 0)
        self.df['p2_high_pressure_win_ratio'] = np.where(self.df['p2_high_pressure'].cumsum() != 0,
                                                    self.df['p2_high_pressure_win'].cumsum() / self.df['p2_high_pressure'].cumsum(), 0)

        self.df['p1_pressure_index'] = self.df['p1_break_pt_efficiency'] + self.df['p1_break_pt_saved_rat'] + self.df['p1_high_pressure_win_ratio']
        self.df['p2_pressure_index'] = self.df['p2_break_pt_efficiency'] + self.df['p2_break_pt_saved_rat'] + self.df['p2_high_pressure_win_ratio']

        self.trans_df['p1_pressure_index'] = self.df['p1_pressure_index'].values[:row + 1]
        self.trans_df['p2_pressure_index'] = self.df['p2_pressure_index'].values[:row + 1]
        
        self.trans_df['p1_break_pt_efficiency'] = self.df['p1_break_pt_efficiency'].values[:row + 1]
        self.trans_df['p2_break_pt_efficiency'] = self.df['p2_break_pt_efficiency'].values[:row + 1]


    def fatigue_factor(self, row):

        if row > 0:
            elapsed_time = self.df['elapsed_time'].iloc[row] - self.df['elapsed_time'].iloc[row-1] # Assuming 'elapsed_time' is already normalized between 0 and 1
        else:
            elapsed_time = 0

        diff_times_rally = elapsed_time * self.df.iloc[row]['rally_count']
        total_elapsed_time = self.df['elapsed_time'].iloc[-1]

        self.trans_df.at[self.current_index, 'tw_arc'] = diff_times_rally / total_elapsed_time

        p1_fatigue_index = (self.df.iloc[row]['p1_distance_run'] + self.df.iloc[row]['rally_count']) / total_elapsed_time
        p2_fatigue_index = (self.df.iloc[row]['p2_distance_run'] + self.df.iloc[row]['rally_count']) / total_elapsed_time

        self.trans_df.at[self.current_index, 'p1_fatigue_index'] = p1_fatigue_index
        self.trans_df.at[self.current_index, 'p2_fatigue_index'] = p2_fatigue_index

    def serve_efficiency(self, row):
        main_df = self.df.copy()

        main_df['p1_ace_count'] = main_df['p1_ace'].cumsum()
        main_df['p2_ace_count'] = main_df['p2_ace'].cumsum()

        main_df['p1_serve_count'] = (self.df['server'] == 1).cumsum()
        main_df['p2_serve_count'] = (self.df['server'] == 2).cumsum()

        main_df['p1_ace_ratio'] = np.where(main_df['p1_serve_count'] != 0, main_df['p1_ace_count'] / main_df['p1_serve_count'], 0)
        main_df['p2_ace_ratio'] = np.where(main_df['p2_serve_count'] != 0, main_df['p2_ace_count'] / main_df['p2_serve_count'], 0)

        main_df['p1_double_fault_count'] = main_df['p1_double_fault'].cumsum()
        main_df['p2_double_fault_count'] = main_df['p2_double_fault'].cumsum()

        main_df['p1_double_fault_ratio'] = np.where(main_df['p1_serve_count'] != 0, main_df['p1_double_fault_count'] / main_df['p1_serve_count'], 0)
        main_df['p2_double_fault_ratio'] = np.where(main_df['p2_serve_count'] != 0, main_df['p2_double_fault_count'] / main_df['p2_serve_count'], 0)

        main_df['p1_fault_count'] = ((main_df['serve_no'] == 2) & (main_df['server'] == 1)).cumsum()
        main_df['p2_fault_count'] = ((main_df['serve_no'] == 2) & (main_df['server'] == 2)).cumsum()

        main_df['p1_fault_ratio'] = np.where(main_df['p1_serve_count'] != 0, main_df['p1_fault_count'] / main_df['p1_serve_count'], 0)
        main_df['p2_fault_ratio'] = np.where(main_df['p2_serve_count'] != 0, main_df['p2_fault_count'] / main_df['p2_serve_count'], 0)

        main_df['p1_good_serve_ratio'] = np.where(main_df['p1_fault_ratio'] != 0, 1 - main_df['p1_fault_ratio'], 0)
        main_df['p2_good_serve_ratio'] = np.where(main_df['p2_fault_ratio'] != 0, 1 - main_df['p2_fault_ratio'], 0)

        self.trans_df['p1_serve_efficiency'] = main_df['p1_ace_ratio'].values[:row + 1] + (main_df['p1_good_serve_ratio'].values[:row + 1]) - main_df['p1_double_fault_ratio'].values[:row + 1]
        self.trans_df['p2_serve_efficiency'] = main_df['p2_ace_ratio'].values[:row + 1] + (main_df['p2_good_serve_ratio'].values[:row + 1]) - main_df['p2_double_fault_ratio'].values[:row + 1]
        
    def match_context(self, row):
        self.trans_df.loc[row, 'elapsed_time'] = self.df['elapsed_time'].iloc[row]

    def process_row(self, index):
        current_row_df = pd.DataFrame()

        if 'p1_serve_efficiency' not in self.trans_df.columns: # 0 & 1
            self.trans_df['p1_serve_efficiency'] = np.nan
        if 'p2_serve_efficiency' not in self.trans_df.columns:
            self.trans_df['p2_serve_efficiency'] = np.nan

        if 'p1_fatigue_index' not in self.trans_df.columns: # 2 & 3
            self.trans_df['p1_fatigue_index'] = np.nan
        if 'p2_fatigue_index' not in self.trans_df.columns:
            self.trans_df['p2_fatigue_index'] = np.nan

        if 'p1_pressure_index' not in self.trans_df.columns: # 4 & 5
            self.trans_df['p1_pressure_index'] = np.nan
        if 'p2_pressure_index' not in self.trans_df.columns:
            self.trans_df['p2_pressure_index'] = np.nan

        if 'p1_won_ratio' not in self.trans_df.columns: # 6 & 7
            self.trans_df['p1_won_ratio'] = np.nan
        if 'p2_won_ratio' not in self.trans_df.columns:
            self.trans_df['p2_won_ratio'] = np.nan

        if 'p1_point_streaks' not in self.trans_df.columns: #8 & 9
            self.trans_df['p1_point_streaks'] = np.nan
        if 'p2_point_streaks' not in self.trans_df.columns:
            self.trans_df['p2_point_streaks'] = np.nan

        if 'p1_game_streaks' not in self.trans_df.columns: # 10 & 11
            self.trans_df['p1_game_streaks'] = np.nan
        if 'p2_game_streaks' not in self.trans_df.columns:
            self.trans_df['p2_game_streaks'] = np.nan      

        if 'p1_set_streaks' not in self.trans_df.columns: # 12 & 13
            self.trans_df['p1_set_streaks'] = np.nan
        if 'p2_set_streaks' not in self.trans_df.columns:
            self.trans_df['p2_set_streaks'] = np.nan


        if 'p1_break_pt_efficiency' not in self.trans_df.columns: # 14 & 15
            self.trans_df['p1_break_pt_efficiency'] = np.nan
        if 'p2_break_pt_efficiency' not in self.trans_df.columns:
            self.trans_df['p2_break_pt_efficiency'] = np.nan

    
        self.point_ratio(index)
        self.calculate_win_streaks_sets(index)
        self.calculate_win_streaks_games(index)
        self.calculate_win_streaks_points(index)
        
        self.pressure_index(index)
        self.fatigue_factor(index)
        self.serve_efficiency(index)
        self.match_context(index)

        # Append the new row DataFrame to the trans_df
        self.trans_df = pd.concat([self.trans_df, current_row_df], axis=0, ignore_index=True)

        # Increment the current_index for the next row
        self.current_index += 1

        # Set index for the trans_df
        self.trans_df.index = [f"row_{i}" for i in range(self.current_index)]

        return self.trans_df
