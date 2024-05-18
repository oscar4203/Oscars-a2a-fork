# Description: Formats the source apple txt files from 'Apples to Apples' game into a csv file.

# Standard Libraries
import csv

# Third-party Libraries

# Local Modules


# Function for opening the file and reading the data
def read_file(file_path: str):
    # Open the file and read the data
    with open(file_path, 'r') as file:
        data = file.read()

    # Log the operation
    print(f"Reading data from {file_path}")
    return data


# Function for converting the green and red apple text data to a csv file
def convert_apple_txt_to_csv(data, csv_file_path: str, apple: str, header: bool=False, extra: bool=False) -> None:
    # Split the data into lines
    lines = data.split('\n')

    # Open the txt file for writing
    with open(csv_file_path, 'w') as file:
        # Use the csv writer
        csv_write = csv.writer(file)

        if header:
            # Write the header
            if apple == "green":
                if extra:
                    csv_write.writerow(["green apples/adjectives", "synonyms", "set"])
                else:
                    csv_write.writerow(["green apples/adjectives"])
            elif apple == "red":
                if extra:
                    csv_write.writerow(["red apples/nouns", "description", "set"])
                else:
                    csv_write.writerow(["red apples/nouns"])

        # Write the data
        for line in lines:
            # Check if the line is empty
            if not line:
                continue

            # Check if the line follows the expected format
            if " - " in line:
                # Split the line into columns based on the first " - "
                if apple == "green":
                    adjective, synonyms_set = line.split(" - ")

                    # Split the synonyms and set columns
                    synonyms, set_pack = synonyms_set.split(") ", 1)

                    if extra:
                        # Clean up and split the synonyms into a list
                        synonyms = synonyms[1:].strip()

                        # Clean up the set pack
                        set_pack = set_pack.strip()

                        # Write the columns to the csv file
                        csv_write.writerow([adjective, synonyms, set_pack])

                        # Log the output
                        print(f"Writing adjective[{adjective}] and synonyms[{synonyms}] from set[{set_pack}] to the csv file")
                    else:
                        # Write the columns to the csv file
                        csv_write.writerow([adjective])

                        # Log the output
                        print(f"Writing adjective[{adjective}] to the csv file")
                if apple == "red":
                    noun, description = line.split(" - ", 1)

                    # Split the description and set columns
                    description, set_pack = description.split(" [", 1)

                    if extra:
                        # Clean up the description
                        description = description.strip()

                        # Clean up the set pack
                        set_pack = set_pack.strip()

                        # Add the "[" back to the set pack
                        set_pack = "[" + set_pack

                        # Write the columns to the csv file
                        csv_write.writerow([noun, description, set_pack])

                        # Log the output
                        print(f"Writing noun[{noun}] and description[{description}] from set[{set_pack}] to the csv file")
                    else:
                        # Write the columns to the csv file
                        csv_write.writerow([noun])

                        # Log the output
                        print(f"Writing noun[{noun}] to the csv file")
            else:
                print(f"Line does not follow the expected format, skipping: {line}")

    # Notify when the opration is complete
    print(f"Apple data has been successfully written to {csv_file_path}")


# Main function
def main():
    # Green apple file paths
    green_txt_source_file_path = "../apples/green_apples_source.txt"
    green_csv_file_path = "../apples/green_apples.csv"

    # Read the green apple data from the file
    data = read_file(green_txt_source_file_path)

    # Convert and format the green apple text data to a csv file
    convert_apple_txt_to_csv(data, green_csv_file_path, "green", header=True, extra=True)

    # Red apple file paths
    red_txt_source_file_path = "../apples/red_apples_source.txt"
    red_csv_file_path = "../apples/red_apples.csv"

    # Read the red apple data from the file
    data = read_file(red_txt_source_file_path)

    # Convert and format the red apple text data to a csv file
    convert_apple_txt_to_csv(data, red_csv_file_path, "red", header=True, extra=True)


if __name__ == "__main__":
    main()
