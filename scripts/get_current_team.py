"""
OBS. Does not integrate with splunk currently. Perhaps would work better if run from a linux server.


Script that attempts to fetch a given FPL team based on team ID.
The idea is that a dashboard in Splunk should take team ID as input and return the users current team in a Splunk
Dashboard. Further development could include "best transfer this week" or "Likely captain" options.
"""

import aiohttp
import asyncio
import requests as req
from fpl import FPL, fpl


async def my_team(user_id):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login()
        user = await fpl.get_user(user_id)
        team = await user.get_team()
    await print(team)

def api_call(url=r"https://fantasy.premierleague.com/api/bootstrap-static/"):
    return req.get(url).json()


async def main():
    """
    Main function used for testing
    """
    data = api_call()
    await my_team(3631718)

main()