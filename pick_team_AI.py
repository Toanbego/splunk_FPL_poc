# !/usr/bin/python
import exec_anaconda
exec_anaconda.exec_anaconda()
import pandas as pd
import requests as req



def api_call(url=r"https://fantasy.premierleague.com/api/bootstrap-static/"):
    return req.get(url).json()


def top_player_points(df):
    """Returns the top players by points"""
    return df.sort_values("total_points", ascending=False)


def top_player_ROI(df):
    """Returns the top 3 players ROI"""
    return df.sort_values("value_season", ascending=False)


def player_by_status(df):
    """Assumes news means the player is unable to play. This should statistically be true,
    but there might be some exceptions"""
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


def get_money_team_objects(data, budget=1000, star_player_limit=3,
                           goalkeepers=2, defenders=5,
                           midfielders=5, forwarders=3):
    df_element_types = pd.DataFrame(data["element_types"])
    df = pd.DataFrame(data["elements"])
    df['value_season'] = df.value_season.astype(float)
    money_team = pd.DataFrame()
    star_player_limit = star_player_limit
    budget = budget
    injured = player_by_status(df)
    positions = {"Goalkeeper": goalkeepers, "Defender": defenders, "Midfielder": midfielders, "Forwarder": forwarders}

    # Select 3 top points players
    for _, player in top_player_points(df).iterrows():
        # Add player if condition is met

        if len(money_team) < star_player_limit and player["second_name"] not in injured \
                and budget >= player["now_cost"] \
                and positions[position(player.element_type)] > 0:

            money_team = money_team.append(player)
            budget -= player["now_cost"]
            positions[position(player.element_type)] = positions[position(player.element_type)] - 1
        else:
            break

    # Add top ROI players afterwards
    for _, player_roi in top_player_ROI(df).iterrows():
        if player_roi["second_name"] not in money_team["second_name"].values and budget >= player_roi["now_cost"] and player_roi["second_name"] not in injured and positions[position(player_roi.element_type)] > 0:
            money_team = money_team.append(player_roi)
            budget -= player_roi["now_cost"]
            positions[position(player_roi.element_type)] = positions[position(player_roi.element_type)] - 1

    money_team['position'] = money_team.element_type.map(df_element_types.set_index('id').singular_name)
    print(money_team.sort_values("position")[["first_name", "second_name", 'position', 'now_cost']])
    print("Total team cost: " + str(sum(money_team['now_cost'].values)/10) + "£")
    return money_team




def create_montioring_file(data):
    # try:
    #     pd.read_csv("historic_data/suggested_team.csv")
    # except Exception:
    print(data)
    # data.to_csv("historic_data/suggested_team.csv")


def main():
    # import requests as req
    # import pandas as pd
    data = api_call()

    money_team = get_money_team_objects(data)
    create_montioring_file(money_team)


main()
