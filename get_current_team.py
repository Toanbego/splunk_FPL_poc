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
    print(team)

def api_call(url=r"https://fantasy.premierleague.com/api/bootstrap-static/"):
    return req.get(url).json()


if __name__ == '__main__':
    data = api_call()
    my_team(3631718)
