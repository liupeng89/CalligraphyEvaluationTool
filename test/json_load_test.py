import json


def main():
    data_path = "../ProgrammWebScrapy.json"

    with open(data_path, 'rb') as f:
        lines = f.read()

        for line in lines:
            if "\n" in line:
                print(repr(line))





if __name__ == '__main__':
    main()