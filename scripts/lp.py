def knapsack_01(self):
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


def simplex(self):
    player_data_json = self.data['elements']
    pdata = pd.DataFrame(player_data_json)
    to_drop = ['chance_of_playing_this_round', 'chance_of_playing_next_round', 'code', 'cost_change_event',
               'cost_change_event_fall', 'cost_change_start', 'cost_change_start_fall', 'dreamteam_count',
               'ep_this', 'event_points', 'form', 'ict_index', 'in_dreamteam',
               'news', 'photo', 'special', 'squad_number', 'status',
               'transfers_in', 'transfers_in_event', 'transfers_out', 'transfers_out_event', 'value_form',
               'value_season']
    pdata.drop(to_drop, axis=1, inplace=True)
    pdata['full_name'] = pdata.first_name + " " + pdata.second_name
    pdata['element_type_name'] = pdata.element_type.map(
        {x['id']: x['singular_name_short'] for x in data['element_types']})
    pdata = pdata.loc[:,
            ['full_name', 'first_name', 'second_name', 'element_type', 'element_type_name', 'id', 'team',
             'team_code', 'web_name',
             'saves', 'penalties_saved', 'clean_sheets', 'goals_conceded',
             'bonus', 'bps', 'creativity', 'ep_next', 'influence', 'threat',
             'goals_scored', 'assists', 'minutes', 'own_goals',
             'yellow_cards', 'red_cards', 'penalties_missed',
             'selected_by_percent', 'now_cost', 'points_per_game', 'total_points']]
    pdata['team'] = pdata.team.map({x['id']: x['name'] for x in data['teams']})

    prob = pulp.LpProblem('FantasyTeam', LpMaximize)
    decision_variables = []
    for rownum, row in pdata.iterrows():
        variable = str('x' + str(rownum))
        variable = pulp.LpVariable(str(variable), lowBound=0, upBound=1, cat='Integer')  # make variables binary
        decision_variables.append(variable)

    print("Total number of decision_variables: " + str(len(decision_variables)))
    # Returns: Total number of decision_variables: 501

    total_points = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                formula = row['total_points'] * player
                total_points += formula

    prob += total_points
    print("Optimization function: " + str(total_points))

    avail_cash = 830
    total_paid = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                formula = row['now_cost'] * player
                total_paid += formula

    prob += (total_paid <= avail_cash)

    avail_gk = 1
    total_gk = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                if row['element_type_name'] == 'GKP':
                    formula = 1 * player
                    total_gk += formula
    prob += (total_gk == avail_gk)
    print(total_gk)

    avail_def = 4
    total_def = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                if row['element_type_name'] == 'DEF':
                    formula = 1 * player
                    total_def += formula
    prob += (total_def == avail_def)
    print((total_def))

    avail_mid = 4
    total_mid = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                if row['element_type_name'] == 'MID':
                    formula = 1 * player
                    total_mid += formula
    prob += (total_mid == avail_mid)
    print((total_mid))

    avail_fwd = 2
    total_fwd = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                if row['element_type_name'] == 'FWD':
                    formula = 1 * player
                    total_fwd += formula
    prob += (total_fwd == avail_fwd)
    print(total_fwd)

    team_dict = {}
    for team in set(pdata.team_code):
        team_dict[str(team)] = dict()
        team_dict[str(team)]['avail'] = 3
        team_dict[str(team)]['total'] = ""
        for rownum, row in pdata.iterrows():
            for i, player in enumerate(decision_variables):
                if rownum == i:
                    if row['team_code'] == team:
                        formula = 1 * player
                        team_dict[str(team)]['total'] += formula

        prob += (team_dict[str(team)]['total'] <= team_dict[str(team)]['avail'])
    print(len(team_dict))
    hey = prob.objective
    print(hey)
    exit()
    prob.writeLP('FantasyTeam.lp')
    optimization_result = prob.solve()
    assert optimization_result == LpStatusOptimal
    print("Status:", LpStatus[prob.status])
    print("Optimal Solution to the problem: ", prob.objective)
    print("Individual decision_variables: ")
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    variable_name = []
    variable_value = []

    for v in prob.variables():
        variable_name.append(v.name)
        variable_value.append(v.varValue)

    df = pd.DataFrame({'variable': variable_name, 'value': variable_value})
    for rownum, row in df.iterrows():
        value = re.findall(r'(\d+)', row['variable'])
        df.loc[rownum, 'variable'] = int(value[0])

    df = df.sort_index('variable')

    # append results
    for rownum, row in pdata.iterrows():
        for results_rownum, results_row in df.iterrows():
            if rownum == results_row['variable']:
                pdata.loc[rownum, 'decision'] = results_row['value']

    pdata[pdata.decision == 1].now_cost.sum()  # Returns 830
    pdata[pdata.decision == 1].total_points.sum()  # Returns 2010.8606251232461
    pdata[pdata.decision == 1].sort_values('element_type').head(11)


def store_shit_here(self):
    pdata_series = pdata.corr()['total_points']
    pdata.pivot_table(index='element_type_name', values='total_points', aggfunc=np.mean)
    pdata.pivot_table(index='element_type_name', values='total_points', aggfunc=np.median)

    f = plt.figure(figsize=(16, 9))
    ax1 = f.add_subplot(2, 2, 1)
    ax2 = f.add_subplot(2, 2, 2, sharex=ax1, sharey=ax1)
    ax3 = f.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
    ax4 = f.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1)
    ax1.set_title('FWD')
    sns.distplot(pdata[pdata.element_type_name == 'FWD'].total_points, label='FWD', ax=ax1)
    ax1.axvline(np.mean(pdata[pdata.element_type_name == 'FWD'].total_points), color='red', label='mean')
    ax2.set_title('MID')
    sns.distplot(pdata[pdata.element_type_name == 'MID'].total_points, label='MID', ax=ax2)
    ax2.axvline(np.mean(pdata[pdata.element_type_name == 'MID'].total_points), color='red', label='mean')
    ax3.set_title('DEF')
    sns.distplot(pdata[pdata.element_type_name == 'DEF'].total_points, label='DEF', ax=ax3)
    ax3.axvline(np.mean(pdata[pdata.element_type_name == 'DEF'].total_points), color='red', label='mean')
    ax4.set_title('GKP')
    sns.distplot(pdata[pdata.element_type_name == 'GKP'].total_points, label='GKP', ax=ax4)
    ax4.axvline(np.mean(pdata[pdata.element_type_name == 'GKP'].total_points), color='red', label='mean')
    plt.show()

    impute_cols = ['saves', 'penalties_saved', 'clean_sheets', 'goals_conceded', 'bonus', 'bps',
                   'creativity', 'influence', 'threat', 'goals_scored', 'assists', 'minutes', 'own_goals',
                   'yellow_cards', 'red_cards', 'penalties_missed', 'points_per_game', 'total_points']
    positions = set(pdata.element_type_name)
    costs = set(pdata.now_cost)
    medians = {};
    stds = {}

    for i in positions:
        medians['{}'.format(i)] = {}
        stds['{}'.format(i)] = {}
        for c in costs:
            medians['{}'.format(i)]['{}'.format(c)] = {}
            stds['{}'.format(i)]['{}'.format(c)] = {}
            for j in impute_cols:
                if pdata[(pdata.total_points != 0) & (pdata.minutes != 0) & (pdata.element_type_name == str(i)) & (
                        pdata.now_cost == c)].shape[0] > 0:
                    median = np.median(pdata[(pdata.total_points != 0) & (pdata.minutes != 0) & (
                            pdata.element_type_name == i) & (pdata.now_cost == c)][j].astype(np.float32))
                    std = np.std(pdata[(pdata.total_points != 0) & (pdata.minutes != 0) & (
                            pdata.element_type_name == i) & (pdata.now_cost == c)][j].astype(np.float32))
                    medians['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = median
                    stds['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = std
                else:
                    medians['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = 0
                    stds['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = 0

    for idx, row in pdata[(pdata.total_points == 0) & (pdata.minutes == 0)].iterrows():
        for col in impute_cols:
            pdata.loc[idx, col] = medians[str(row['element_type_name'])][str(row['now_cost'])][str(col)]
            + np.abs(
                (np.random.randn() / 1.5) * stds[str(row['element_type_name'])][str(row['now_cost'])][str(col)])
