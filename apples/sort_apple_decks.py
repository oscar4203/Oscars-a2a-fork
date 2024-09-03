# Description: Sorts and eports all the apples and exports each set into a csv file.

# Standard Libraries
import csv
import re
from collections import defaultdict

# Third-party Libraries

# Local Modules


# Function to clean and split the set names
def extract_sets(set_string):
    return re.findall(r'\[([^\]]+)\]', set_string)


def main(base_filename: str) -> None:
    # Read the CSV file
    with open(f"{base_filename}-all.csv", mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        # Dictionary to hold sets and their corresponding rows
        sets_dict = defaultdict(list)

        for row in reader:
            sets = extract_sets(row[2])
            for set_name in sets:
                sets_dict[set_name].append(row)

    # Write to new CSV files based on sets
    for set_name, rows in sets_dict.items():
        filename = f"{base_filename}-{set_name}.csv"
        with open(filename, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    main("green_apples")
    main("red_apples")
