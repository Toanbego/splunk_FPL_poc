# # !/usr/bin/python
# import exec_anaconda
# exec_anaconda.exec_anaconda()
import pandas as pd
import requests as req
import datetime



def api_call(url=r"https://fantasy.premierleague.com/api/bootstrap-static/"):
    """API call"""
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


def get_money_team_objects(data, use_last_season, budget=1000, star_player_limit=3,
                           goalkeepers=2, defenders=5,
                           midfielders=5, forwarders=3):
    """
    Uses a dumb algorithm to pick most valued players. Picks 3 star players (top most valued player by points)
    then fills in the rest with top ROI (Region Of Interest. This is players that has the highest value per million rate)
    Budget is set to 1000 since player cost is set a factor of 10 higher than on the
    fantasy webpage. meaning that if a player costs 5.5£ on the home page, it will cost 55 with this dataframe.
    """

    if use_last_season:
        data2 = pd.read_csv("historic_data/2020-09-01.csv")
        # Create dataFrames based on data categories
        df_element_types = pd.DataFrame(data["element_types"])
        df_team = pd.DataFrame(data['teams'])
        df = pd.DataFrame(data2)
    else:
        # Create dataFrames based on data categories
        df_element_types = pd.DataFrame(data["element_types"])
        df_team = pd.DataFrame(data['teams'])
        df = pd.DataFrame(data["elements"])
    df['value_season'] = df.value_season.astype(float)  # Convert all values to float, in case some are strings

    # Set up variables
    money_team = pd.DataFrame()
    star_player_limit = star_player_limit
    budget = budget
    injured = player_by_status(df)
    positions = {"Goalkeeper": goalkeepers, "Defender": defenders, "Midfielder": midfielders, "Forwarder": forwarders}

    # Select X top points players

    for _, player in top_player_points(df).iterrows():

        # Add player if condition is met

        if len(money_team) < star_player_limit and player["second_name"] not in injured \
                and budget >= player["now_cost"] \
                and positions[position(player.element_type)] > 0:
            player["Star"] = "Yes"
            money_team = money_team.append(player)

            budget -= player["now_cost"]
            positions[position(player.element_type)] = positions[position(player.element_type)] - 1
        else:
            break

    # Add top ROI players afterwards
    for _, player_roi in top_player_ROI(df).iterrows():
        if player_roi["second_name"] not in money_team["second_name"].values and budget >= player_roi["now_cost"] and player_roi["second_name"] not in injured and positions[position(player_roi.element_type)] > 0:
            player_roi["Star"] = "No"
            money_team = money_team.append(player_roi)
            budget -= player_roi["now_cost"]
            positions[position(player_roi.element_type)] = positions[position(player_roi.element_type)] - 1

    money_team = map_team_and_position(money_team, df_element_types, df_team)  # Maps which team and position belongs to each player
    pd.options.display.max_columns = None
    # Print data
    print(money_team.sort_values("position")[["second_name", 'position', "Star", "team", "now_cost" #"total_points", 'value_season', 'now_cost'
                                              ]])

    print("Total team cost: " + str(round(sum(money_team['now_cost'].values)/10, 2)) + "£")
    save_data_to_csv(money_team)


def save_data_to_csv(df):
    
    try:
        df[["first_name", "second_name",
            'position', "Star", "team",
            "now_cost", "total_points",
            'value_season']].to_csv(f"C:/Program Files/Splunk/etc/apps/Fantasy_PL/bin"
                                    f"/data/team_suggestion/suggested_team {datetime.date.today()}.txt")
    except FileNotFoundError:
        print("file not found - splunk")
        pass

    try:
        df[["first_name", "second_name",
            'position', "Star", "team",
            "now_cost", "total_points",
            'value_season']].to_csv(f"../team_suggestion/suggested_team {datetime.date.today()}.txt")
    except FileNotFoundError:
        print("file not found - local")
        pass

def main():
    data = api_call()
    use_last_season = False
    get_money_team_objects(data, use_last_season)



main()
