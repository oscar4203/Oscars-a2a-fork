# Description: Sorts and eports all the apples and exports each set into a csv file.

# Standard Libraries
import csv
import re
from collections import defaultdict
import os

# Third-party Libraries

# Local Modules


# Function to clean and split the set names
def extract_sets(set_string):
    return re.findall(r'\[([^\]]+)\]', set_string)

# Function to read the CSV file and extract sets
def read_csv(file_path: str):
    with open(file_path, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        # Dictionary to hold sets and their corresponding rows
        sets_dict = defaultdict(list)

        for row in reader:
            sets = extract_sets(row[2])
            for set_name in sets:
                sets_dict[set_name].append(row)

    return header, sets_dict

# Function to write the sets to separate CSV files
def write_csv(file_path: str, header: list, rows: list):
    with open(file_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

def main() -> None:
    # Define the base directory for data files
    data_dir = "../data/apples"

    # Process green apples
    base_filename_green = "green_apples"
    input_filepath_green = os.path.join(data_dir, f"{base_filename_green}-all.csv")
    header_green, sets_dict_green = read_csv(input_filepath_green)

    # Write to new green apples CSV files based on sets
    for set_name, rows in sets_dict_green.items():
        output_filename_green = os.path.join(data_dir, f"{base_filename_green}-{set_name}.csv")
        write_csv(output_filename_green, header_green, rows)
        print(f"Wrote {len(rows)} green apples to {output_filename_green}")

    # Process red apples
    base_filename_red = "red_apples"
    input_filepath_red = os.path.join(data_dir, f"{base_filename_red}-all.csv")
    header_red, sets_dict_red = read_csv(input_filepath_red)

    # Write to new red apples CSV files based on sets
    for set_name, rows in sets_dict_red.items():
        output_filename_red = os.path.join(data_dir, f"{base_filename_red}-{set_name}.csv")
        write_csv(output_filename_red, header_red, rows)
        print(f"Wrote {len(rows)} red apples to {output_filename_red}")


if __name__ == "__main__":
    main()
