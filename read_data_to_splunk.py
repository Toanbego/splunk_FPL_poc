import os


def read_elements_data():
    path = r'C:\Program Files\Splunk\etc\apps\Fantasy_PL\bin\data\elements'
    file = os.listdir(path)[-1]
    f = open(f"{path}/{file}", 'r')
    print(f.read())


if __name__ == '__main__':
    read_elements_data()


