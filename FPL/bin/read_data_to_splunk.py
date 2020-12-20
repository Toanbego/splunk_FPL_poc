import os


def read_elements_data():
    print("Hello World")
    path = r"C:\Program Files\Splunk\etc\apps\FPL\bin"

    file = os.listdir(path + r'\elements')[-1]
  
    f = open(path + r"/elements/" + file, 'r')

    print(f.read())


if __name__ == '__main__':
    read_elements_data()


