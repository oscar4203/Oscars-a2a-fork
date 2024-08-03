# Description: Small script to count the number of times each unique player has won a game

# Standard Libraries
import csv

# Third-party Libraries

# Local Modules

def count_winners(filename: str) -> dict[str, int]:
    # Initialize the winners dictionary
    winners = {}

    # Open the file
    try:
        with open(filename, "r") as file:
            # Create a CSV reader object
            reader = csv.DictReader(file)

            # Check if the file is empty
            if not reader.fieldnames:
                print("CSV file is empty or has no header")
                return winners

            # Iterate through the rows
            for row in reader:
                # Get the winning player
                winner = row["Winner"]

                # Check if the 'Winner' column exists
                if winner is None:
                    print("No 'Winner' column found in CSV")
                    return winners

                # If the player is not in the dictionary, add them
                if winner not in winners:
                    winners[winner] = 0

                # Increment the player's win count
                winners[winner] += 1
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return winners


def main():
    # Define the filename
    filename = "./logs/winners.csv"

    # Get the winners dictionary
    try:
        winners = count_winners(filename)

        # Print the winners
        for player, wins in winners.items():
            print(f"{player}: {wins} wins")

        # If there are no winners
        if not winners:
            print("No winners found")
    except csv.Error:
        print("Error reading CSV file")
    except Exception as e:
        print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    main()
