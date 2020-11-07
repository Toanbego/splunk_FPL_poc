import pick_team_AI
import numpy as np
import pandas as pd
import copy
import random
from tqdm import tqdm


def get_total_team_score(individual, metric="total_points"):
    return individual[metric].sum()


class SelectTeamEA(pick_team_AI.TeamSelectorAI):

    def __init__(self, data, use_last_season):
        super().__init__(data, use_last_season)
        self.population_size = 5
        self.population = self.create_population()
        self.evaluation = self.evaluate_fitness(fitness_metric="total_points")

    def evaluate_fitness(self, fitness_metric="total_points"):
        """
        Evaluates the fitness of a populations.
        :param fitness_metric: What metric to use for fitness evaluations: Fantasy score, Player ROI or a hybrid.
        """

        return [(get_total_team_score(individual, fitness_metric)) for individual in self.population]

    def create_population(self):
        """
        Creates a random population of solutions. The largest limitation is
        """
        population = []

        # Loop through
        for i in tqdm(range(self.population_size)):
            team = pd.DataFrame()
            position_slots = copy.deepcopy(self.positions)
            budget = self.budget
            df = self.df
            team = self.select_random_team(team, df, position_slots, budget)
            population.append(team)

        return population

    def evolve(self):
        """
        Evolves the population
        Ends by replacing the population with the new
        one.
        :return:
        """
        # Evolve population
        self.parents = self.select_parents()  # Parents selection
        self.offsprings = self.pmx(self.pmx_prob)  # Create offsprings through pmx crossover
        self.offsprings = self.mutate_population(self.offsprings, self.mutation_prob)  # Mutates the new population
        self.replace_population()

    def select_parents(self):
        """
        Selects parents based on fitness proportionate selection.
        Performs a windowing on fitness beforehand to increase
        the chance for better solutions to be picked. If there
        is to little diversity in the population, population is forced
        to evolve, due to a divison by 0 error.
        :return: parents
        """
        # Perform windowing on fitness
        window_fitness = [(self.evaluation[i] - min(self.evaluation)) for i, item in enumerate(self.evaluation)]

        # If there is too little diversity in population, mutate population
        while sum(window_fitness) == 0:
            self.population = self.mutate_population(self.population, 1)
            self.evaluation = self.evaluate_population(self.population)
            window_fitness = [(self.evaluation[i] - min(self.evaluation)) for i, item in enumerate(self.evaluation)]

        # Create selection wheel and select parents from it
        selection_wheel = [(fitness / sum(window_fitness)) for fitness in window_fitness]
        selection = np.random.choice(range(len(self.population)), len(selection_wheel),
                                     p=selection_wheel)  # Random selection
        parents = [self.population[parent] for parent in selection]
        return parents

    def pmx(self):
        pass

    def mutate_population(self, population, mutation_prob=1/24):
        """
        A probability that an individual will mutate with a random transfers.
        Sell x players, and buy x new ones
        """
        offspring = []

        for i, individual in enumerate(population):
            if random.random() < mutation_prob:

                # Randomly sell up to 3 players
                n = np.random.randint(low=1, high=3)
                players = individual.sample(n=n)
                current_budget = 1000-individual["now_cost"].sum() + players["now_cost"].sum()
                for _, player in players.iterrows():
                    self.randomly_select_players(player, budget=current_budget, n=n, position=player["position"])
                    if player["second_name"] not in self.injured and current_budget >= player["now_cost"]:
                        pass

                seq_idx = list(range(len(individual)))
                a1, a2 = random.sample(seq_idx[1:-1], 2)
                individual[a1], individual[a2] = individual[a2], individual[a1]

                # Updates offspring
                offspring.append(individual)
            else:
                offspring.append(individual)
        return offspring

    def replace_population(self):
        pass

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
    def randomly_select_players(df, budget, n, position=None):
        if position is not None:
            df = df.loc[df["now_cost"] <= budget]
            df = df.loc[df["position"] == position]
            player = df.sample(n=n)
        else:
            df = df.loc[df["now_cost"] <= budget]
            player = df.sample(n=n)
        return player.iloc[0]

    def select_random_team(self, team, df, position_slots, budget):
        while len(team) < 15:
            # Randomly select a player
            available_positions = self.check_position_slots(position_slots)
            position = np.random.choice(available_positions, 1)[0]
            player = self.randomly_select_players(df, budget, n=1, position=position)
            df.drop([df.loc[df["second_name"] == player["second_name"]].index.values[0]])

            # Check if injured or money left in budget
            if player["second_name"] not in self.injured and budget >= player["now_cost"] and position_slots[position] > 0:
                team = team.append(player)
                budget -= player["now_cost"]
                position_slots[position] = position_slots[position] - 1
        return team


def genetic_algorithm(team_selector_AI, generations=10):
    """Evolves a random population of FPL teams"""

    # Evaluate fitness
    fitness = team_selector_AI.evaluate_fitness()
    best_fitness_per_generation = []

    # Evolve population
    for generation in tqdm(range(generations)):
        best_fitness_per_generation.append(max(fitness))
        team_selector_AI.evolve()

if __name__ == '__main__':
    api_data = pick_team_AI.api_call()
    select_team_EA = SelectTeamEA(api_data, False)
    genetic_algorithm(select_team_EA)

