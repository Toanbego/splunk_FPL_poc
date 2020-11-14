# # !/usr/bin/python
# import exec_anaconda
# exec_anaconda.exec_anaconda()
import pandas as pd
import requests as req
import datetime
import numpy as np
import seaborn as sns
import re
import ast
import keras
import os
# import keras_preprocessing

sns.set_style('whitegrid')
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 100)


def api_call(url=r"https://fantasy.premierleague.com/api/bootstrap-static/"):
    """API call"""
    # url = "https://fantasy.premierleague.com/api/element-summary/193/"
    return req.get(url).json()


def top_player_points(df):
    """Returns a sorted data frame with descending on the column total_points"""
    return df.sort_values("total_points", ascending=False)


def top_player_ROI(df):
    """Returns a sorted data frame with descending on the column value_season"""
    return df.sort_values("value_season", ascending=False)


def player_by_status(df):
    """Assumes news means the player is unable to play. This should statistically be true,
    but there might be some exceptions"""
    df = df.dropna()
    return df["second_name"].loc[(df["news"].str.contains(" "))].values


def position(position_int):
    """Map position int to string"""
    if position_int == 1:
        return "Goalkeeper"
    elif position_int == 2:
        return "Defender"
    elif position_int == 3:
        return "Midfielder"
    elif position_int == 4:
        return "Forward"


def map_team_and_position(elements, element_type, teams):
    """In dataframe, maps team_code to actual team name. Same with position"""
    elements['position'] = elements.element_type.map(element_type.set_index('id').singular_name)
    elements['team'] = elements.team.map(teams.set_index('id').name)
    return elements


class TeamSelectorAI:
    """
    Uses a dumb algorithm to pick most valued players. Picks 3 star players (top most valued player by points)
    then fills in the rest with top ROI (Region Of Interest. This is players that has the highest value per million rate)
    Budget is set to 1000 since player cost is set a factor of 10 higher than on the
    fantasy webpage. meaning that if a player costs 5.5£ on the home page, it will cost 55 with this dataframe.
    """
    def __init__(self, data, use_last_season, budget=1000, star_player_limit=3,
                 goalkeepers=2, defenders=5,
                 midfielders=5, forwarders=3):
        self.data_path = r"C:\Users\torstein.gombos\Desktop\FPL\historic_data\Fantasy-Premier-League\data\\"
        self.data = data
        self.use_last_season = use_last_season
        self.budget = budget
        self.star_player_limit = star_player_limit
        self.money_team = pd.DataFrame()
        self.positions = {"Goalkeeper": goalkeepers, "Defender": defenders,
                          "Midfielder": midfielders, "Forward": forwarders}
        self.df, self.df_team, self.df_element_types = self.create_dataframe(use_last_season)
        self.df = map_team_and_position(self.df, self.df_element_types, self.df_team)
        self.injured = player_by_status(self.df)
        self.seasons = os.listdir(self.data_path)

        # ML related attributes
        self.batch_size = 16
        self.network = self.model_fcn



    def train(self):
        keras.Model.fit()

    def model_fcn(self):
        """
        Creates a Fully Connected neural network
        Input shape is 1x144
        Returns the expected reward for a set of actions.
        :return:
        """

        model = keras.models.Sequential()

        model.add(keras.layers.Dense(4096, activation='relu',
                                     batch_size=self.batch_size,
                                     ))

        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(4096, activation='relu'
                                     ))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(2096, activation='relu'
                                     ))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(512, activation='relu'
                                     ))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(12, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,

                      optimizer=keras.optimizers.adadelta(),
                      metrics=['accuracy'])

        return model

    def create_dataframe(self, use_last_season):
        """
        Create dataframe from data, either with last season or this season
        """
        if use_last_season:
            # Create dataFrames based on data categories
            df_element_types = pd.DataFrame(self.data["element_types"])
            df_team = pd.DataFrame(self.data['teams'])
            # TODO: Test this solution below
            df = pd.read_csv("../elements/2020-09-01.txt")
        else:
            # Create dataFrames based on data categories
            df_element_types = pd.DataFrame(self.data["element_types"])
            df_team = pd.DataFrame(self.data['teams'])
            df = pd.DataFrame(self.data["elements"])
        pd.options.display.max_columns = None
        df['value_season'] = df.value_season.astype(float)  # Convert all values to float, in case some are strings

        return df, df_team, df_element_types

    @staticmethod
    def get_team_fixture(historic_data_path, teams_df, fixtures_df, players_season, gw=1):
        """
        Returns a dataframe containing the fixtures, players and scores for a given game week.
        """
        fixtures_df = fixtures_df.loc[fixtures_df["event"] == gw]
        gw = pd.read_csv(historic_data_path + r"\gws\gw"+str(gw)+".csv")
        gw['first_name'], gw['second_name'] = gw.element.map(players_season.set_index('id').first_name), gw.element.map(players_season.set_index('id').second_name)
        gw = gw.loc[gw["minutes"] > 0]
        gw['team'] = gw.element.map(players_season.set_index('id').team)
        gw['opponent_team'] = gw.opponent_team.map(teams_df.set_index('id').name)
        stats = ast.literal_eval(re.search('({.+})', fixtures_df.stats.values[0]).group(0))[-1]
        teams = gw.team.unique()
        match_setups = {}
        for i, team in enumerate(teams):
            team1 = gw.loc[gw["team"] == team]
            opponent_team = team1["opponent_team"].unique()[0]
            team2 = gw.loc[gw["team"] == opponent_team]
            teams = np.delete(teams, np.argwhere(teams == opponent_team))
            match_setups[i] = [team1, team2]

        return match_setups, stats

    def get_season_data(self, season="2019-20", gw=1):
        """
        Creates input for neural network
        """
        # Get Data frames for the given season
        historic_data_path = self.data_path + season
        teams_df = pd.read_csv(historic_data_path + r"\teams.csv")

        fixtures_df = pd.read_csv(historic_data_path + r"\fixtures.csv")
        fixtures_df['team_h'] = fixtures_df.team_h.map(teams_df.set_index('id').name)
        fixtures_df['team_a'] = fixtures_df.team_a.map(teams_df.set_index('id').name)

        players_season = pd.read_csv(historic_data_path + r"\players_raw.csv")
        players_season['team'] = players_season.team.map(teams_df.set_index('id').name)

        # Extract game setup dict
        gw_setups, stats = self.get_team_fixture(historic_data_path, teams_df, fixtures_df, players_season, gw)
        return gw_setups, stats



    def create_dataset(self):
        """
        Creates dataset for training. 1 batch is one game week.
        Current setting: Total dataset size is number of game weeks for a given season.
        """
        batches = len(os.listdir(self.data_path + self.seasons[-2] +"/gws"))




    def simple_AI(self):
        """
        Simple AI to optimize team selection
        """
        # Select Top players by points
        for _, player in top_player_points(self.df).iterrows():

            # Add player if condition is met
            if len(self.money_team) < self.star_player_limit and player["second_name"] not in self.injured \
                    and self.budget >= player["now_cost"] \
                    and self.positions[position(player.element_type)] > 0:
                player["Star"] = "Yes"
                self.money_team = self.money_team.append(player)
                self.budget -= player["now_cost"]
                self.positions[position(player.element_type)] = self.positions[position(player.element_type)] - 1
            else:
                break

        # Add top ROI players afterwards
        for _, player_roi in top_player_ROI(self.df).iterrows():
            if player_roi["second_name"] not in self.money_team["second_name"].values and \
                    self.budget >= player_roi["now_cost"] and player_roi["second_name"] not in self.injured \
                    and self.positions[position(player_roi.element_type)] > 0:
                player_roi["Star"] = "No"
                self.money_team = self.money_team.append(player_roi)
                self.budget -= player_roi["now_cost"]
                self.positions[position(player_roi.element_type)] = self.positions[position(player_roi.element_type)] - 1

        self.money_team = map_team_and_position(self.money_team, self.df_element_types, self.df_team)  # Maps which team and position belongs to each player
        return self.money_team

    def print_team(self):
        """Prints currently selected team"""
        # Print data
        print(self.money_team.sort_values("position")[
                  ["second_name", 'position', "Star", "team", "now_cost"  # "total_points", 'value_season', 'now_cost'
                   ]])

        print("Total team cost: " + str(round(sum(self.money_team['now_cost'].values) / 10, 2)) + "£")
        print("Total team points: " + str(round(sum(self.money_team['total_points'].values), 2)))

    def save_data_to_csv(self):
        """
        Save data
        """
        if self.use_last_season:
            description = "previous_season"
        else:
            description = datetime.date.today()
        try:
            self.money_team[["first_name", "second_name",
                             'position', "Star", "team",
                             "now_cost", "total_points",
                             'value_season']].to_csv(f"C:/Splunk/etc/apps/FPL/bin"
                                                     f"/team_suggestion/suggested_team {description}.txt")
        except FileNotFoundError:
            print("file not found - splunk")
            pass

        try:
            self.money_team[["first_name", "second_name",
                             'position', "Star", "team",
                             "now_cost", "total_points",
                             'value_season']].to_csv(f"../team_suggestion/suggested_team {description}.txt")
        except FileNotFoundError:
            print("file not found - local")
            pass


if __name__ == '__main__':

    data = api_call()
    use_last_season = False
    team_selector_ai = TeamSelectorAI(data, use_last_season)
    # team_selector_ai.knapsack_01()
    # team_selector_ai.simple_AI()
    # team_selector_ai.print_team()
    team_selector_ai.get_season_data()
