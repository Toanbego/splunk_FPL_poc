# Table of Contents
1. [Description](#description)
2. [FPL](#fpl)
3. [How to use this repo](#use-this-repo)
4. [How to use without Splunk](#without-splunk)

## Description<a name="description"></a>
This is a project to extract Fantasy Premier Leauge data, using their official API, and foward it to a Spunk dashboard.
The idea is to generate insight in how to best optimize your Fantasy team to get the most points for your buck. 
The project also looks into some optimization algorithms, like ML, Evolutionary Programming, Knapsack and etc.

### FPL<a name="fpl"></a>
The Fantasy Premier League, or the FPL, is an online game which is played at http://www.premierleague.com

You are given a budget of 100 million(fake money, obviously), to buy a squad of 15 players consisting of 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards with the added rule of being able to select a maximum of 3 players from any particular Barclays Premier League team. The cost of a player is pre-determined by the game developers, based on the player's popularity and performance in the last football season.

The objective of the game is to create the best team possible and earn maximum points based on their performance in real 
matches.

### Preview:
![ROI](https://github.com/Toanbego/splunk_FPL_poc/blob/master/images/FPL_ROI.PNG)
![Points](https://github.com/Toanbego/splunk_FPL_poc/blob/master/images/FPL_points.PNG)
![AI](https://github.com/Toanbego/splunk_FPL_poc/blob/master/images/FPL_AI.PNG)
![Extra](https://github.com/Toanbego/splunk_FPL_poc/blob/master/images/extra_stats.PNG)


## To Use This Repo<a name="use-this-repo"></a>
Copy paste the the FPL folder and its content to your $SPLUNK_HOME/apps/ folder. This will set up the app in splunk automatically. Restart Splunk for the changes to take effect.
Indexed data is not included in the repo. To start indexing data, see the section below.

### Getting data to your local folder.
Set up two scripted inputs in Splunk Web (or inputs.conf) which runs <i>get_elements.py</i> and <i>get_current_team.py</i>. They will read the latest content of the <i>elements</i> and <i>team_suggestion</i> folders. To add data to these folders, run the two scripts in the script folder.

<i>Obs* Splunk's python env might not have the proper setup to run pick_AI_team.py. My workaround is to schedule the scripts to run with my local python env, using the .bat file in the apps bin folder. </i>

The <i>run_locally_at_interval.BAT</i> runs the scripts <i>get_elements.py</i> and <i>get_current_team.py</i>.
This will create a jsnon and a csv file in the <i>elements</i> and <i>team_suggestion</i> folders.
* The first script stores statistical data about each player as of latest update
* The second script uses a simple algorithm to maximise player value given a budget of 100 million bucks.
https://towardsdatascience.com/beating-the-fantasy-premier-league-game-with-python-and-data-science-cf62961281be


## How to use without Splunk<a name="without-splunk"></a>
In the scripts folder you will find scripts for API calls and various scripts for testing different optimization
algorithms.
Each script should be able to run separately, though many are work in progress and might not deliver any
usable results. 
 



