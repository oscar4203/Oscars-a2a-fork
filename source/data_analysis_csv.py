# Description: Script to count the number of times each unique player has won a game

# Standard Libraries
import argparse
import logging
import numpy as np
import csv

# Third-party Libraries
from scipy.stats import binomtest, norm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgba
import matplotlib.patheffects as path_effects

# Local Modules
from source.data_classes import GameLog
from source.game_logger import LOGGING_BASE_DIRECTORY
from source.data_analysis import abbreviate_name, save_plot, print_table, create_legend, create_game_settings_box, \
    create_bar_plot, create_pie_chart, create_box_plot, create_line_graph


def prepare_players(round_winners_csv: str) -> tuple[list[str], list[str]]:
    """
    Prepares the plot data including players list, player strings, abbreviated names, and colors.

    Args:
        game_log (GameLog): GameLog object containing all the game data.

    Returns:
        Tuple[List[str], List[str], List[str], Dict[str, str]]: Tuple containing players list, player strings,
    """
    # Compile a list of all unique players in the csv file
    players_string: list[str] = []

    with open(round_winners_csv, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for row in csv_reader:
            # Skip the header row
            if row[0] == "round_winner":
                continue

            # Add the player to the list if they are not already in it
            if row[0] not in players_string:
                players_string.append(row[0])

    # Abbreviate player names for x-axis
    abbreviated_names = [abbreviate_name(player) for player in players_string]

    return players_string, abbreviated_names


def calculate_round_wins_per_game(round_winners_csv: str) -> dict[str, list[int]]:
    """
    Calculate the number of round wins per game for each AI agent.

    Args:
        round_winners_csv (str): Path to the CSV file containing the round winners data.

    Returns:
        dict: A dictionary with agent objects as keys and lists of round wins per game as values.
    """
    # Initialize the dictionary with each player having a list of zeros for each game
    round_wins_per_game: dict[str, list[int]] = {}

    with open(round_winners_csv, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for row in csv_reader:
            # Skip the header row
            if row[0] == "round_winner":
                continue

            # Add the player to the dictionary if they are not already in it
            if row[0] not in round_wins_per_game:
                round_wins_per_game[row[0]] = [0]

            # Increment the round wins for the player in the current game
            round_wins_per_game[row[0]][-1] += 1

    return round_wins_per_game


def calculate_confidence_intervals(win_rates: dict[str, float], total_games: int, confidence_level: float = 0.95) -> dict:
    pass


def perform_statistical_tests(win_rates: dict[str, float], total_games: int) -> dict:
    pass

def create_vector_line_graph(game_state_csv: str) -> Figure:
    pass


def create_round_winners_plot(game_state_csv: str) -> Figure:
    pass


def create_game_winners_plot(game_state_csv: str) -> Figure:
    pass


def prepare_round_win_stats(game_state_csv: str) -> list[list]:
    pass


def prepare_game_win_stats(game_state_csv: str) -> list[list]:
    pass


def prepare_players_and_colors(game_state_csv: str) -> tuple[list[str], list[str], list[str], list[str]]:
    pass


def prepare_plot_data(game_state_csv: str, win_counts: dict) -> tuple:
    pass


def create_vector_line_graph(ax: Axes, ai_agent: str, opponents: list[str], game_state_csv: str) -> None:
    pass


def create_round_winners_plot(game_state_csv: str, change_players_between_games: bool,
                            cycle_starting_judges: bool, reset_models_between_games: bool,
                            use_extra_vectors: bool) -> Figure:
    pass


def create_game_winners_plot(game_state_csv: str, change_players_between_games: bool,
                            cycle_starting_judges: bool, reset_models_between_games: bool,
                            use_extra_vectors: bool) -> Figure:
    pass


def create_heatmap(game_state_csv: str) -> Figure:
    # Prepare common plot data
    players_string, abbreviated_names = prepare_players(game_state_csv)

    # Initialize the heatmap data matrix
    num_players = len(players_string)
    heatmap_data = np.zeros((num_players, num_players), dtype=int)

    # Populate the heatmap data matrix
    for game in game_log.game_states:
        for round in game.round_states:
            judge: str = round.current_judge.get_name()
            winner: str = round.round_winner.get_name() if round.round_winner is not None else "No Winner"
            judge_index = players_string.index(judge)
            winner_index = players_string.index(winner)
            heatmap_data[judge_index, winner_index] += 1

    # Create the heatmap figure
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(heatmap_data, cmap="YlGnBu")

    # Add color bar
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(num_players))
    ax.set_yticks(np.arange(num_players))
    ax.set_xticklabels(abbreviated_names, fontsize=16, rotation=45, ha="left")
    ax.set_yticklabels(abbreviated_names, fontsize=16)

    # Add title
    ax.set_title("Heatmap of Judges' Choices", pad=40, fontsize=20, fontweight="bold")

    # Add subtitle aligned with the heatmap
    ax.text(0.5, 1.3, "[x-axis: winners | y-axis: judges]", ha="center", va="center", transform=ax.transAxes, fontsize=16, fontweight="bold")

    # Annotate each cell with the numeric value
    for i in range(num_players):
        for j in range(num_players):
            text = ax.text(j, i, str(heatmap_data[i, j]), va="center", ha="center", color="white", fontsize=20, fontweight="bold")
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground="black"), path_effects.Normal()])

    # Set ticks at the edges of the cells
    ax.set_xticks(np.arange(-0.5, num_players, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_players, 1), minor=True)

    # Add grid lines at the minor ticks
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    # Adjust layout
    plt.tight_layout()

    return fig


def create_vector_history_plot(game_state_csv: str) -> Figure:
    pass


def main(game_state_csv: str) -> None:
    # Generate judge heatmap output filepath
    judge_heatmap_output_filepath = game_state_csv.replace("game_state", "judge_heatmap").replace(".csv", ".png")

    # Create a plot of the judge heatmap
    judge_heatmap_plot = create_heatmap(game_state_csv)

    # Save the plot to a file
    save_plot(judge_heatmap_plot, judge_heatmap_output_filepath)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Count round winners from a CSV file.")

    # Add an argument for the filename as input
    parser.add_argument("round_winners", help="Round winners .csv file.")

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(args.round_winners)
