import os


def read_team_suggestion_data():
    path = "C:/Program Files/Splunk/etc/apps/FPL/bin"
    file = os.listdir(f"{path}/team_suggestion")[-1]

    f = open(path+f"/team_suggestion/{file}", 'r')
    print(f.read())


# def checkpoint_mark():
#     """Used to checkpoint that data was written this day. Mostly used for debugging. Currently not in use"""
#     import datetime
#     with open(r"checkpoint/API_call_timestamp.txt", "w") as f:
#         f.writelines(f"Timestamp: {datetime.date.today()} - {datetime.datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    read_team_suggestion_data()
    # checkpoint_mark()


