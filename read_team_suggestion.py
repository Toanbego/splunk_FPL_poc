import os


def read_team_suggestion_data():
    path = r'C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\data\team_suggestion'
    file = os.listdir(path)[-1]
    f = open(f"{path}/{file}", 'r')
    # print(file, path)
    print(f.read())


def checkpoint_mark():
    """Used to checkpoint that data was written this day. Mostly used for debugging. Currently not in use"""
    import datetime
    with open(r"checkpoint/API_call_timestamp.txt", "w") as f:
        f.writelines(f"Timestamp: {datetime.date.today()} - {datetime.datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    read_team_suggestion_data()
    checkpoint_mark()


