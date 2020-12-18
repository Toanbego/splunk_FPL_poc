from pulp import *
def LP_FPL(self):
    data = self.df

    # player = {str(i) for i in self.df["second_name"]}
    point = {str(i): self.df["total_points"][i] for i in range(len(self.df["total_points"]))}

    cost = {str(i): self.df['now_cost'][i] for i in range(len(self.df["now_cost"]))}
    gk = {str(i): 1 if self.df['position'][i] == 'Goalkeeper' else 0 for i in range(len(self.df["total_points"]))}
    defe = {str(i): 1 if self.df['position'][i] == 'Defender' else 0 for i in range(len(self.df["total_points"]))}
    mid = {str(i): 1 if self.df['position'][i] == 'Midfielder' else 0 for i in range(len(self.df["total_points"]))}
    stri = {str(i): 1 if self.df['position'][i] == 'Forward' else 0 for i in range(len(self.df["total_points"]))}

    player = {str(i): 1 for i in range(data.shape[0])}

    prob = LpProblem("Fantasy_Football", LpMaximize)
    player_vars = LpVariable.dicts("Players", player, 0, 1, LpBinary)

    # objective function

    prob += lpSum([point[i] * player_vars[i] for i in player]), "Total Cost"

    # constraint
    prob += lpSum([player_vars[i] for i in player]) == 15, "Total 11 Players"
    prob += lpSum([cost[i] * player_vars[i] for i in player]) <= 100.0, "Total Cost"
    prob += lpSum([gk[i] * player_vars[i] for i in player]) == 1, "Only 1 Goalkeeper"
    prob += lpSum([defe[i] * player_vars[i] for i in player]) <= 4, "Less than 4 Defender"
    prob += lpSum([mid[i] * player_vars[i] for i in player]) <= 5, "Less than 5 Midfielder"
    prob += lpSum([stri[i] * player_vars[i] for i in player]) <= 3, "Less than 3 Forward"

    # solve
    status = prob.solve()
