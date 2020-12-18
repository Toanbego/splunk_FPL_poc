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

![Drag Racing](https://github.com/Toanbego/splunk_FPL_poc/blob/master/images/FPL_AI.PNG)


## To Use This Repo<a name="use-this-repo"></a>
This repo is built like the bin folder for a Splunk app. To use this in Splunk, create an App in your
Splunk instance, then clone this repo to the $SPLUNK_HOME/etc/app/bin folder. 
* The conf folder contains inputs and look up tables. The have to be
extracted to the necessary locations manually.
* The dashboards folder contains the xml files for the dashboard. Either extract the file
to the correct ui folder, or simply click create a new dashboard in Splunk, and copy
paste the content.

### Getting data to your local folder.
The <i>run_locally_at_interval.BAT</i> runs the scripts <i>get_elements.py</i> and <i>get_current_team.py</i>.
This will create csv files in the <i>elements</i> and <i>team_suggestion</i> folders.
* The first script stores statistical data about each player as of latest update
* The second script uses a simple algorithm to maximise player value given a budget of 100 million bucks.
https://towardsdatascience.com/beating-the-fantasy-premier-league-game-with-python-and-data-science-cf62961281be

### Getting data into Splunk<a name="description"></a> 
<i>read_data_to_splunk.py</i> and <i>read_team_suggestion.py</i> will read and print data from the latest txt file
stored in elements and team_suggestion. To add the data in Splunk, create a scripted input in splunk which runs each of these
scripts. Sourcetype is csv format. 

## How to use without Splunk<a name="without-splunk"></a>
In the script fold you will find scripts for API calls and various scripts for testing different optimization
algorithms. Each script should be able to run separately, though many are work in progress and might not deliver any
usable results. 
 



