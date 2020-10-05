#!/usr/bin/python
# import exec_anaconda
# exec_anaconda.exec_anaconda()

"""
Scripted input that returns current premier league stats for each player.
Function perforsm API call and prints the result in a json format to syslog. Suggested splunk sourcetype is therefore
_json.

Working on support storing data to a csv file using pandas. However, to use pandas Splunk needs the app
"python for scientific computing" which installs a python env with access to pandas, numpy and etc. This is the
reason for the lines:
    #!/usr/bin/python
    # import exec_anaconda
    # exec_anaconda.exec_anaconda()
"""

import json
import os

import requests as req
import sys
import datetime


def api_call(url=r"https://fantasy.premierleague.com/api/bootstrap-static/"):
    """Performs API call"""
    return req.get(url).json()


def checkpoint_mark():
    """Used to checkpoint that data was written this day. Mostly used for debugging. Currently not in use"""
    import datetime
    with open(r"../checkpoint/API_call_timestamp.txt", "w") as f:
        f.writelines(f"Timestamp: {datetime.date.today()} - {datetime.datetime.now().strftime('%H:%M:%S')}")


def store_value_in_dataframe(json_data):
    """
    Uses pandas to store data to a csv file under directory /historic_data/<today_date>.csv. Currently not in use until
    support
    for pandas is configured properly.
    """

    import pandas as pd
    df = pd.read_csv(r"/bin/backup/historic_data/elements_data.csv")
    df_new = pd.DataFrame(json_data["elements"])

    df = df.append(df_new)
    df.to_csv("historic_data/elements_data.csv")


def store_value_in_new_csv(json_data):
    """
    Uses pandas to store data to a csv file under directory /historic_data/<today_date>.csv.
    Currently not in use until support
    for pandas is configured properly.
    """
    import datetime

    import pandas as pd
    df = pd.DataFrame(json_data["elements"])
    df.to_csv(f"historic_data/{datetime.date.today()}.csv")
    df.to_csv(f"historic_data/elements_data11.csv")


def store_value_in_txt(json_data):
    """
    Uses pandas to store data to a csv file under directory /historic_data/<today_date>.csv.
    Currently not in use until support
    for pandas is configured properly.
    """
    import datetime

    import pandas as pd
    df = pd.DataFrame(json_data["elements"])
    df.to_csv(f"historic_data/{datetime.date.today()}.csv")
    df.to_csv(f"historic_data/elements_data11.csv")


def suggest_team(dt):
    """Uses the bot in pick_team_AI.py to suggest best pick based on current season so far.
    Currently not in use until support for pandas is configured properly for Splunk. However, this can be run locally to
    get a suggestion
    Try/Except is set up to bypass splunks monitoring input function, which is based on changes in a file. Have not been
    tested that it works. Splunk might see that it is the same name and not consider the new file different from the
    deleted one.
    """
    money_team = pick_team_AI.get_money_team_objects(dt)

    # 1. Try to create csv file
    # 2. if file already exists, delete file and create a new one

    try:
        money_team.to_csv(r"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\suggested_team.csv",
                          index=False)
    except Exception:
        os.remove(r"/historic_data/suggested_team.csv")
        money_team.to_csv(r"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\suggested_team.csv",
                          index=False)


def save_data_to_json(json_data):
    unpacked_data = json.dumps(json_data["elements"])

    f = open(f"C:/Program Files/Splunk/etc/apps/Fantasy_PL/bin/data/elements/{datetime.date.today()}.txt", "w")
    f.write(f"{unpacked_data}")
    f.close()


if __name__ == '__main__':

    # Get data
    data = api_call()
    print(data["elements"])
    # Write data to file
    save_data_to_json(data)

