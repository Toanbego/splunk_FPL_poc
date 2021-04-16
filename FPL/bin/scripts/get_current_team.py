"""
OBS. Does not integrate with splunk currently. Perhaps would work better if run from a linux server.


Script that attempts to fetch a given FPL team based on team ID.
The idea is that a dashboard in Splunk should take team ID as input and return the users current team in a Splunk
Dashboard. Further development could include "best transfer this week" or "Likely captain" options.
"""


import json
import os

import pandas as pd
import requests as req
from fpl import FPL
sessionId = ".eJyrVopPLC3JiC8tTi2Kz0xRslIyNbQwMzExM1PSQZZKSkzOTs0DyRfkpBXk6IFk9AJ8QoFyxcHB_o5ALqqGjMTiDKBqMzNTE8tU81Qjc8NUU2OjNMtEizSTNKM0gzSD1NSUZINUixRjc7OUJKVaAIFpLOk:1kppnq:A3KJR_5xxRvqPIeIQvFvOYtxqiI"
fpl = FPL(sessionId)
# print(fpl.get_team(12))
hey = fpl.get_user(3631718, True)
for something in hey:
    print(hey)
exit()

def api_call(url=r"https://fantasy.premierleague.com/api/bootstrap-static/"):
    """Performs API call"""
    return req.get(url).json()



if __name__ == '__main__':
    # Get data
    # Get data
    API_BASE_URL = "https://fantasy.premierleague.com/api/"
    url = "{}my-team/3631718".format(API_BASE_URL)
    print(url)
    # exit()
    data = api_call(url)
    # df = data["elements"]
    print(data)
