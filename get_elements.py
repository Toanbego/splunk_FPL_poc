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



def api_call(url=r"https://fantasy.premierleague.com/api/bootstrap-static/"):
    """Performs API call"""
    return req.get(url).json()


def checkpoint_mark():
    """Used to checkpoint that data was written this day. Mostly used for debugging. Currently not in use"""
    import datetime
    with open(r"checkpoint\API_call_timestamp.txt", "w") as f:
        f.writelines(f"Timestamp: {datetime.date.today()} - {datetime.datetime.now().strftime('%H:%M:%S')}")


def store_value_in_dataframe(json_data):
    """
    Uses pandas to store data to a csv file under directory /historic_data/<todays_date>.csv. Currently not in use until support
    for pandas is configured properly.
    """
    import datetime

    import exec_anaconda
    exec_anaconda.exec_anaconda()
    import pandas as pd
    pd.DataFrame(data=json_data["elements"]).to_csv(fr"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\{datetime.date.today()}.csv", index=False)


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
        money_team.to_csv(r"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\suggested_team.csv", index=False)
    except Exception:
        os.remove(r"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\suggested_team.csv")
        money_team.to_csv(r"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\suggested_team.csv", index=False)



if __name__ == '__main__':

    # Get data
    data = api_call()

    # Send data to LOG for Splunk
    unpacked_data = json.dumps(data["elements"])
    print(r'{}'.format(unpacked_data))





