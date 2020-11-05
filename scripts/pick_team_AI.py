# # !/usr/bin/python
# import exec_anaconda
# exec_anaconda.exec_anaconda()
import pandas as pd
import requests as req
import datetime



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
        return "Forwarder"


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
        self.data = data
        self.use_last_season = use_last_season
        self.budget = budget
        self.star_player_limit = star_player_limit
        self.money_team = pd.DataFrame()
        self.positions = {"Goalkeeper": goalkeepers, "Defender": defenders,
                          "Midfielder": midfielders, "Forwarder": forwarders}
        self.df, self.df_team, self.df_element_types = self.create_dataframe(use_last_season)
        self.injured = player_by_status(self.df)

    def create_dataframe(self, use_last_season):
        """
        Create dataframe from data, either with last season or this seasion
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


def main():
    data = api_call()
    use_last_season = False
    team_selector_ai = TeamSelectorAI(data, use_last_season)
    team_selector_ai.simple_AI()
    team_selector_ai.print_team()


main()
