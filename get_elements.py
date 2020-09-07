#!/usr/bin/python
# import exec_anaconda
# exec_anaconda.exec_anaconda()
# Put the rest of your imports below, e.g.:
# import numpy as np
import requests as req
import json
import datetime
# import pandas as pd


def api_call(url="https://fantasy.premierleague.com/api/bootstrap-static/"):
    return req.get(url).json()


def checkpoint_mark():

    with open("checkpoint/API_call_timestamp.txt", "w") as f:
        f.writelines(f"Timestamp: {datetime.date.today()} - {datetime.datetime.now().strftime('%H:%M:%S')}")


# def store_value_in_dataframe(json_data):
#     pd.DataFrame(data=json_data["elements"]).to_csv(f"historic_Data/{datetime.date.today()}.csv")


if __name__ == '__main__':
    # Add checkpoint for Splunk call
    # checkpoint_mark()

    # Get data
    data = api_call()

    # Send data to LOG for Splunk
    print(json.dumps(data["elements"]))

    # # Store value in history. Used later for trendlines
    # store_value_in_dataframe(data)




