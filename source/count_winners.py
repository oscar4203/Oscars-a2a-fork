# Description: Small script to count the number of times each unique player has won a game

# Standard Libraries
import csv
import argparse

# Third-party Libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

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


def abbreviate_name(name: str) -> str:
    # Keep only capital letters and hyphens, ignore spaces
    return "".join(
        char for char in name
            if char.isupper()
            or char == '-'
            or char.isdigit()
        )


def plot_winners(winners: dict[str, int]) -> None:
    # Check if there are any winners
    if not winners:
        print("No winners found")
        return

    # Get the players and wins
    players = list(winners.keys())
    wins = list(winners.values())
    total_games = sum(wins)

    # Abbreviate player names for x-axis
    abbreviated_names = [abbreviate_name(player) for player in players]

    # Define a list of standard, contrasting colors
    standard_colors = ["red", "cyan", "orange", "purple", "lime", "brown", "pink", "gray"]

    # Repeat the colors if there are more players than colors
    colors = (standard_colors * (len(players) // len(standard_colors) + 1))[:len(players)]

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 4])

    # Legend plot
    legend_ax = fig.add_subplot(gs[0, :])
    handles = [Rectangle((0,0),1,1, color=color) for color in colors[:len(players)]]
    legend_ax.legend(handles, players, title="Players", loc="center", fontsize=12, title_fontsize=14)
    legend_ax.axis('off')

    # Bar plot
    bar_ax = fig.add_subplot(gs[1, 0])
    bar_ax.bar(abbreviated_names, wins, color=colors)
    bar_ax.set_xlabel("Players", fontsize=16, fontweight="bold")
    bar_ax.set_ylabel("Wins", fontsize=16, fontweight="bold")
    bar_ax.set_title("Total Wins per Unique Player", fontsize=18, fontweight="bold")
    bar_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Use the default black color for the legend handle
    black_patch = mpatches.Patch(color="black", label=f"Total Games: {total_games}")
    bar_ax.legend(handles=[black_patch], loc="upper right", fontsize=14)

    # Pie chart
    pie_ax = fig.add_subplot(gs[1, 1])
    pie_result = pie_ax.pie(wins, labels=abbreviated_names, colors=colors, autopct='%1.1f%%', startangle=140)
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_fontsize(14)
    else:
        wedges, texts = pie_result
    pie_ax.set_title("Proportion of Wins per Player", fontsize=18, fontweight="bold")

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()


def main(filename: str) -> None:
    # Get the winners dictionary
    try:
        winners = count_winners(filename)

        # Print the winners
        print(f"Total games: {sum(winners.values())}")
        for player, wins in winners.items():
            print(f"{player}: {wins} wins")

        # If there are no winners
        if not winners:
            print("No winners found")

        # Plot the winners
        plot_winners(winners)
    except csv.Error:
        print("Error reading CSV file")
    except Exception as e:
        print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count winners from a CSV file.")
    parser.add_argument(
        "filename",
        nargs="?",
        default="./logs/winners.csv",
        help="Path to the CSV file containing winners data"
    )
    args = parser.parse_args()
    main(args.filename)
