# Description: Script to count the number of times each unique player has won a game

# Standard Libraries
import os
import csv
import argparse

# Third-party Libraries
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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
                winner = row["Game Winner"]

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
    # Define the archetype words and their abbreviations
    archetypes = {
        "Literalist": "Lit",
        "Contrarian": "Con",
        "Comedian": "Com"
    }

    # Split the name into words
    words = name.split()

    # Initialize the result list
    result = []

    for word in words:
        if word in archetypes:
            # If the word is an archetype, use its abbreviation
            result.append(archetypes[word])
        else:
            # Otherwise, keep only capital letters, hyphens, and digits
            result.append("".join(
                char for char in word
                    if char.isupper()
                    or char == '-'
                    or char.isdigit()
            ))

    # Join the result list into a single string
    return "".join(result)


def percent_ai_won(winners: dict[str, int]) -> float:
    # Get the total number of games
    total_games = sum(winners.values())

    # Get the number of games won by AI agents
    ai_wins = sum(
        wins
        for player, wins in winners.items()
        if "AI" in player
    )

    # Calculate the percentage of games won by AI agents
    return ai_wins / total_games * 100 if total_games > 0 else 0


def create_plot_for_winners(winners: dict[str, int], points_to_win: int, total_games: int) -> Figure:
    # Check if there are any winners
    if not winners:
        print("No winners found")
        raise ValueError("No winners found")

    # Get the players and wins
    players = list(winners.keys())
    wins = list(winners.values())

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
    bar_ax.set_title("Total Wins per Unique Player", fontsize=18, fontweight="bold")
    bar_ax.set_xlabel("Players", fontsize=16, fontweight="bold")
    bar_ax.set_ylabel("Wins", fontsize=16, fontweight="bold")
    bar_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Use the default black color for the legend handle
    black_patch_points = mpatches.Patch(color="black", label=f"Points to Win: {points_to_win}")
    black_patch_games = mpatches.Patch(color="black", label=f"Total Games: {total_games}")
    bar_ax.legend(handles=[black_patch_points, black_patch_games], loc="upper right", fontsize=14)

    # Pie chart
    pie_ax = fig.add_subplot(gs[1, 1])
    pie_result = pie_ax.pie(wins, labels=abbreviated_names, colors=colors, autopct='%1.1f%%', startangle=140)
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_fontsize(14)

    # Add a title to the pie chart
    pie_ax.set_title("Proportion of Wins per Player", fontsize=18, fontweight="bold")

    # Calculate the percentage of games won by AI agents
    ai_wins = sum(wins[i] for i, player in enumerate(players) if "AI" in player)
    percent_ai = ai_wins / total_games * 100 if total_games > 0 else 0

    # Add the percentage of games won by AI agents as a title below the pie chart
    pie_ax.text(0.5, -0.1, f"AI Wins: {percent_ai:.2f}%", ha='center', va='center', fontsize=14, transform=pie_ax.transAxes)

    # Adjust layout
    plt.tight_layout()

    return fig


def save_plot(plot_figure: Figure, output_filepath: str) -> None:
    # Save the plot to a file
    plot_figure.savefig(output_filepath)


def main(filepath: str, points_to_win: int, total_games: int) -> None:
    # Get the winners dictionary
    try:
        winners = count_winners(filepath)

        # Print the winners
        print(f"Points to win: {points_to_win}")
        print(f"Total games: {total_games}")
        for player, wins in winners.items():
            print(f"{player}: {wins} wins")

        # If there are no winners
        if not winners:
            print("No winners found")

        # Generate output filename
        base_name = os.path.splitext(filepath)[0]
        output_filepath = f"{base_name}.png"

        # Create a plot of the winners
        plot = create_plot_for_winners(winners, points_to_win, total_games)

        # Save the plot to a file
        save_plot(plot, output_filepath)

        # Display the plot
        plt.show()

    except csv.Error:
        print("Error reading CSV file")
        raise csv.Error
    except Exception as e:
        print(f"An error occurred in main: {e}")
        raise e


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Count winners from a CSV file.")

    # Add an argument for the filename as input
    parser.add_argument("filepath", nargs="?", default="./logs/winners.csv", help="Path to the CSV file containing winners data")
    parser.add_argument("points_to_win", help="Total number of points to win (1-10).")
    parser.add_argument("total_games", help="Total number of games to play (1-1000).")

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(args.filepath, args.points_to_win, args.total_games)
