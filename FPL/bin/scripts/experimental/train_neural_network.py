import pick_team_AI
import pandas as pd
import numpy as np
import configparser
from tqdm import tqdm
import keras
import os
import sklearn.model_selection
from sklearn import preprocessing

config = configparser.ConfigParser()
config.read("config.ini")


def reformat_columns(df, player_df, element_types, teams_df):
    """
    Specific use case to reformat a dataframe to contain useful information.
    Return df with columns: name, position, team, opponent_team, was_home
    """
    df = df.loc[df["minutes"] > 0]
    df['team'] = df.element.map(player_df.set_index('id').team)
    df['position_name'] = df.element.map(player_df.set_index('id').element_type)
    df['position'] = df.position_name.map(element_types.set_index('id').singular_name)
    df['opponent_team'] = df.opponent_team.map(teams_df.set_index('id').name)
    return df


def encode_dataset(training_input, test_input):
    """
    One-hot encode data. Does not encode player names since it would create a massive dimensionality increase
    """
    ohe = preprocessing.OneHotEncoder()
    test_input = ohe.fit_transform(test_input)
    training_input = ohe.fit_transform(training_input)
    return training_input, test_input


def ordinal_encode_data(data):
    """ Ordinal encodes data. Does not include names in vector """
    encoder = preprocessing.OrdinalEncoder()
    encoder = encoder.fit_transform(data[['opponent_team', 'position', 'team', 'ict_index']])
    encoder[:, -1] /= 10
    return np.asarray(encoder)


def nominal_encoder(data):
    """Nominal Encodes data"""
    encoder = preprocessing.LabelEncoder()
    labels = encoder.fit_transform(data[['opponent_team', 'position', 'team', 'ict_index']])
    label_mappings = {index: label for index, label in enumerate(labels.classes_)}
    return label_mappings


def normalize_data(data):
    """ Normalize data to be between 0 and 1. Really stupid to do with encoding it seems """
    _min = np.min(data)
    _max = np.max(data)
    data = (data - _min) / (_max - _min)
    return data


class NeuralNetwork(pick_team_AI.TeamSelectorAI):
    def __init__(self, data, use_last_season):
        super().__init__(data, use_last_season)

        # Preprocessing attributes
        self.encoding = config['dataset']['ENCODING']
        self.features = ["name", "position",
                         "ict_index",
                         "team", "opponent_team",
                         "was_home", "total_points"]
        self.dummy_set_size = config['dataset'].getint('DUMMY_SET_SIZE')
        self.seasons = self.seasons[:config['dataset'].getint('NUMBER_OF_SEASONS')]

        # ML related attributes
        self.batch_size = config['network'].getint('BATCH_SIZE')
        self.optimizer_metric = config['network']['METRIC_TO_OPTIMIZE']
        self.learning_rate = config['network'].getfloat('LEARNING_RATE')
        self.epochs = config['network'].getint('EPOCHS')

        # Chooses optimizer
        self.optimizer = config['network']['OPTIMIZER']
        if self.optimizer == 'adadelta':
            self.optimizer = keras.optimizers.Adadelta()
        elif self.optimizer == 'adam':
            self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)

        self.model = None
        self.x, self.y = None, None
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

    def model_fcn(self):
        """
        Creates a Fully Connected neural network
        Input shape is 1x144
        Returns the expected reward for a set of actions.
        :return:
        """

        keras_model = keras.models.Sequential()
        keras_model.add(keras.layers.Dense(1024, activation='relu'
                                           ))
        keras_model.add(keras.layers.Dense(1024, activation='relu'
                                           ))

        keras_model.add(keras.layers.Dense(512, activation='relu'
                                           ))

        keras_model.add(keras.layers.Dropout(0.1))
        keras_model.add(keras.layers.Dense(512, activation='relu'
                                           ))
        keras_model.add(keras.layers.Dropout(0.1))

        keras_model.add(keras.layers.Dense(512, activation='relu'
                                           ))

        keras_model.add(keras.layers.Dense(256, activation='relu'
                                           ))

        keras_model.add(keras.layers.Dense(1, activation='relu'))

        keras_model.compile(loss=keras.losses.mean_absolute_error,
                            optimizer=self.optimizer,
                            metrics=[self.optimizer_metric])

        return keras_model

    def get_data_per_season(self, season="2019-20"):
        """
        Creates input for neural network
        """
        # Get Data frames for the given season
        historic_data_path = self.data_path + season
        teams_df = pd.read_csv(historic_data_path + r"\teams.csv")

        # Get player data for a given season
        players_season = pd.read_csv(historic_data_path + r"\players_raw.csv")
        players_season['team'] = players_season.team.map(teams_df.set_index('id').name)

        # Create data for each player per season
        i = 0

        x = pd.DataFrame()
        y = pd.DataFrame()

        for player in tqdm(os.listdir(historic_data_path + "/players")):
            path_to_player = historic_data_path + r"\players\\" + player + r"\gw.csv"
            player_stats_per_season = pd.read_csv(path_to_player)
            player_stats_per_season["name"] = player.split("_")[0] + " " + player.split("_")[1]
            player_stats_per_season["season"] = season
            player_stats_per_season = reformat_columns(player_stats_per_season,
                                                       players_season,
                                                       self.df_element_types, teams_df)[self.features]
            # Create array in the beginning, or add new data to existing array
            if i == 0:
                x = player_stats_per_season[self.features[:-1]]
                y = player_stats_per_season[self.features[-1]]

            elif i > 0:
                x = pd.concat([x, player_stats_per_season[self.features[:-1]]])
                y = pd.concat([y, player_stats_per_season[self.features[-1]]])

            i += 1

            if self.dummy_set_size > 0:
                if i > self.dummy_set_size:
                    break

        return x, y

    def extract_dataset(self):
        """
        Creates a dataset from the available seasons of data. One datasample is one gameweek for a player.
        """

        first_season = self.seasons[0]

        for season in self.seasons:
            x_temp, y_temp = self.get_data_per_season(season)
            if season == first_season:
                self.x, self.y = x_temp, y_temp

            else:
                self.x = pd.concat((self.x, x_temp))
                self.y = pd.concat((self.y, y_temp))

    def preprocess_data(self):
        """
        Preprocessing steps:
        1. Encode data
        2. If Ordinal encoding, normalize data afterwards
        3. Split into training and test sets
        """
        x = []
        if self.encoding == 'ORDINAL_ENCODING':
            x = ordinal_encode_data(self.x)
            x = normalize_data(x)

        elif self.encoding == 'NOMINAL_ENCODING':
            x = nominal_encoder(self.x)

        elif self.encoding == 'ONEHOT_ENCODING':
            x = self.onehot_encode_with_pandas()
            x[:, 0] = self.normalize_data(x[:, 0])
            x = np.asarray(x).astype('float')
        y = self.y.values

        # Split dataset into training and validation set
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(x, y,
                                                                                                        test_size=0.01,
                                                                                                        random_state=42)

    def train(self):
        """ Train model with training data in a supervised manner """

        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs)

    def setup_model(self):
        """ Setup and compile Neural Net model """
        self.model = self.model_fcn()

    def validate(self, x, y):
        """ Validate network """
        pass

    def onehot_encode_with_pandas(self):
        """ Onehot encodes dataset. Does not include names in vector"""
        data = pd.get_dummies(self.x[self.features[1:-1]])
        return data.values


if __name__ == '__main__':
    api_data = pick_team_AI.api_call()
    model = NeuralNetwork(api_data, False)
    model.extract_dataset()
    model.setup_model()
    model.preprocess_data()
    model.train()
    # model.validate(x_test, y_test)
