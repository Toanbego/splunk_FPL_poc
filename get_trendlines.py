#!/usr/bin/python
# import exec_anaconda
# exec_anaconda.exec_anaconda()
import numpy as np
import requests as req
import json
import datetime
import pandas as pd
import os, sys


def create_7_days_of_data():
    df = pd.read_csv("historic_data/2020-09-07.csv")

    for day in range(0, 7):

        date = datetime.date.today() - datetime.timedelta(days=day)
        df["now_cost"] += np.random.choice([0.0, 0.1, 0.2, 0.3], size=len(df["now_cost"]), p=[0.6, 0.1, 0.05, 0.25])
        df.to_csv(f"historic_data/{date}.csv", index=False)


def create_trend_matrix():
    files_list = os.listdir("historic_data/")[:]
    print(files_list)


if __name__ == '__main__':
    create_trend_matrix()