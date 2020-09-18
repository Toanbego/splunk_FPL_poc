#!/usr/bin/python
import exec_anaconda
exec_anaconda.exec_anaconda()
import pandas as pd
import pick_team_AI
import json
import os

import requests as req




def api_call(url=r"https://fantasy.premierleague.com/api/bootstrap-static/"):
    return req.get(url).json()


def checkpoint_mark():
    import datetime
    with open(r"checkpoint\API_call_timestamp.txt", "w") as f:
        f.writelines(f"Timestamp: {datetime.date.today()} - {datetime.datetime.now().strftime('%H:%M:%S')}")


# def store_value_in_dataframe(json_data):
#     # import datetime
#     # import time
#     import exec_anaconda
#     exec_anaconda.exec_anaconda()
#     import pandas as pd
#     pd.DataFrame(data=json_data["elements"]).to_csv(fr"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\{datetime.date.today()}.csv", index=False)


def suggest_team(dt):

    money_team = pick_team_AI.get_money_team_objects(dt)
    # try:
    #     money_team.to_csv(r"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\suggested_team.csv", index=False)
    # except Exception:
    #     os.remove(r"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\suggested_team.csv")
    #     money_team.to_csv(r"C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\historic_data\suggested_team.csv", index=False)


if __name__ == '__main__':

    # Get data
    data = api_call()

    # Send data to LOG for Splunk
    unpacked_data = json.dumps(data["elements"])
    print(r'{}'.format(unpacked_data))

    # Store value in history. Used later for trendlines
    suggest_team(data)



