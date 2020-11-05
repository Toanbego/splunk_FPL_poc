import pick_team_AI
import numpy as np
import pandas as pd


class SelectTeamEA(pick_team_AI.TeamSelectorAI):

    def __init__(self, data, use_last_season):
        super().__init__(data, use_last_season)
        self.population_size = 500



    def create_population(self):
        """
        Creates a random population of solutions
        """
        population = []

        # Loop through
        for i in range(self.population_size):
            team = pd.DataFrame()
            position_slots = self.positions
            budget = self.budget
            df = self.df
            while len(team) < 15:

                # Randomly select a player
                available_positions = self.check_position_slots(position_slots)
                position = np.random.choice(available_positions, 1)[0]
                player = self.randomly_select_players(df, n=1, seed=None, position=position)
                df.drop([df.loc[df["second_name"] == player["second_name"]].index.values[0]])

                # Check if injured or money left in budget
                if player["second_name"] not in self.injured and budget >= player["now_cost"] and position_slots[position] > 0:
                    team = team.append(player)
                    budget -= player["now_cost"]
                    position_slots[position] = position_slots[position] - 1
            population.append(team)
        return population

    @staticmethod
    def check_position_slots(position_slots):
        positions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
        if position_slots["Goalkeeper"] < 1:
            positions.remove("Goalkeeper")
        if position_slots["Defender"] < 1:
            positions.remove("Defender")
        if position_slots["Midfielder"] < 1:
            positions.remove("Midfielder")
        if position_slots["Forward"] < 1:
            positions.remove("Forward")
        return positions

    @staticmethod
    def randomly_select_players(df, n, seed, position=None):
        if position is not None:
            player = df.loc[df["position"] == position].sample(n=n)
        else:
            player = df.sample(n=n)
        return player.iloc[0]




if __name__ == '__main__':
    api_data = pick_team_AI.api_call()
    select_team_EA = SelectTeamEA(api_data, False)
    select_team_EA.create_population()

