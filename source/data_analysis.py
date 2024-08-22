# Description: Script to count the number of times each unique player has won a game

# Standard Libraries
import os
import csv
import argparse

# Third-party Libraries
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

# Local Modules
from source.agent import Agent
from source.data_classes import GameLog


def count_winners(filename: str, header: str) -> dict[str, int]:
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
                winner = row[header]

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


def print_winners_table(game_log: GameLog, game_winners: dict[str, int], round_winners: dict[str, int]) -> None:
    # Prepare the data for the table
    table_data = []
    for player in game_log.all_game_players:
        player_name = player.get_name()
        game_wins = game_winners.get(player_name, 0)
        round_wins = round_winners.get(player_name, 0)
        table_data.append([player_name, game_wins, round_wins])

    # Define the headers
    headers = ["Player", "Game Wins", "Round Wins"]

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


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


def prepare_plot_data(game_log: GameLog) -> tuple[list[str], list[str], list[str], list[str]]:
    # Get the players and wins
    players: list[Agent] = [player for player in game_log.all_game_players]
    players_string: list[str] = [player.get_name() for player in players]

    # Abbreviate player names for x-axis
    abbreviated_names = [abbreviate_name(player) for player in players_string]

    # Define a list of standard, contrasting colors
    standard_colors = ["red", "cyan", "orange", "purple", "lime", "brown", "pink", "gray"]

    # Repeat the colors if there are more players than colors
    colors = (standard_colors * (len(players_string) // len(standard_colors) + 1))[:len(players_string)]

    return players_string, abbreviated_names, standard_colors, colors


def create_legend(ax: Axes, colors: list[str], labels: list[str]) -> None:
    # Remove the axis borders
    ax.axis("off")

    # Add a title to the legend
    ax.set_title("Legend", fontsize=18, fontweight="bold")
    
    # Create a legend for the plot
    handles = [Rectangle((0,0),1,1, color=color) for color in colors]
    ax.legend(handles, labels, title="Players", loc="upper center", fontsize=12, title_fontsize=14)


def create_game_settings_box(ax: Axes, points_to_win: int, total_games: int,
                             change_players_between_games: bool, cycle_starting_judges: bool,
                             reset_models_between_games: bool, use_extra_vectors: bool,
                             use_losing_red_apples: bool) -> None:
    # Remove the axis borders
    ax.axis("off")

    # Add a title to the box
    ax.set_title("Game Settings", fontsize=18, fontweight="bold")

    # Define the settings text
    settings_text = (f"Points to Win: {points_to_win}\n"
                    f"Total Games: {total_games}\n\n"
                    f"change_players_between_games = {change_players_between_games}\n"
                    f"cycle_starting_judges = {cycle_starting_judges}\n"
                    f"reset_models_between_games = {reset_models_between_games}\n"
                    f"use_extra_vectors = {use_extra_vectors}\n"
                    f"use_losing_red_apples = {use_losing_red_apples}")

    # Add the settings text closer to the title
    ax.text(0.5, 0.45, settings_text, fontsize=12, fontweight="bold",
            horizontalalignment="center", verticalalignment="top")


def create_bar_plot(ax: Axes, names: list[str], values: list[int], colors: list[str], title: str, xlabel: str, ylabel: str) -> None:
    ax.bar(names, values, color=colors)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def create_pie_chart(ax: Axes, data: list[int], labels: list[str], colors: list[str], title: str, total_rounds_games: int) -> None:
    # Filter out data and labels that correspond to 0%
    filtered_data = []
    filtered_labels = []
    filtered_colors = []
    for i, value in enumerate(data):
        if value > 0:
            filtered_data.append(value)
            filtered_labels.append(labels[i])
            filtered_colors.append(colors[i])

    # Create pie chart
    pie_result = ax.pie(filtered_data, labels=filtered_labels, colors=filtered_colors,
                        autopct='%1.1f%%', startangle=140)
    
    # Set font weight and size for pie chart
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_fontsize(14)
    
    # Add a title to the pie chart
    ax.set_title(title, fontsize=18, fontweight="bold")

    # Calculate the percentage of games won by AI agents
    ai_wins = sum(data[i] for i, label in enumerate(labels) if "AI" in label)
    percent_ai = ai_wins / total_rounds_games * 100 if total_rounds_games > 0 else 0

    # Add the percentage of games won by AI agents as a title below the pie chart
    ax.text(0.5, -0.1, f"AI Wins: {percent_ai:.2f}%", ha="center", va="center", fontsize=14, transform=ax.transAxes)


def create_box_plot(ax: Axes, data: list[list[int]], labels: list[str], colors: list[str], title: str, xlabel: str, ylabel: str) -> None:
    # Convert colors to RGBA format if necessary
    rgba_colors = [to_rgba(color) for color in colors]

    # Extend colors to match the length of data
    extended_colors = [rgba_colors[i % len(rgba_colors)] for i in range(len(data))]
    
    # Create box plot
    box = ax.boxplot(data, patch_artist=True)
    
    # Apply RGBA colors to the box plot
    for patch, color in zip(box["boxes"], extended_colors):
        patch.set_facecolor(color)

    # Set titles and labels
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    
    # Ensure the number of ticks matches the number of labels
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Add grid for better readability
    ax.grid(True)


def create_figure(title: str, players_string: list[str], colors: list[str], game_log: GameLog, 
                  change_players_between_games: bool, cycle_starting_judges: bool, 
                  reset_models_between_games: bool, use_extra_vectors: bool, use_losing_red_apples: bool, 
                  bar_plot_data, pie_plot_data, box_plot_data, bar_plot_labels, pie_plot_labels, box_plot_labels) -> Figure:
    # Create a figure with GridSpec
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[5, 4], figure=fig)

    # Add a custom title with underlining
    custom_title = f"{title}\n{'_' * len(title)}\n"
    fig.suptitle(custom_title, fontsize=24, fontweight="bold")

    # Legend
    legend_ax = fig.add_subplot(gs[0, 2])
    create_legend(legend_ax, colors, players_string)

    # Game settings box
    game_settings_ax: Axes = fig.add_subplot(gs[1, 2])
    create_game_settings_box(game_settings_ax, game_log.points_to_win, game_log.total_games,
                             change_players_between_games, cycle_starting_judges, reset_models_between_games,
                             use_extra_vectors, use_losing_red_apples)

    # Bar plot
    bar_ax: Axes = fig.add_subplot(gs[0, 0])
    create_bar_plot(bar_ax, *bar_plot_data, *bar_plot_labels)

    # Pie chart
    pie_ax: Axes = fig.add_subplot(gs[0, 1])
    create_pie_chart(pie_ax, *pie_plot_data, *pie_plot_labels)

    # Box plot
    box_ax: Axes = fig.add_subplot(gs[1, 0:2])
    create_box_plot(box_ax, *box_plot_data, *box_plot_labels)

    # Adjust layout
    plt.tight_layout()

    return fig


def create_round_winners_plot(round_winners_dict: dict[str, int], game_log: GameLog, change_players_between_games: bool,
                            cycle_starting_judges: bool, reset_models_between_games: bool, use_extra_vectors: bool, use_losing_red_apples: bool) -> Figure:
    # Check if there are any winners
    if not round_winners_dict:
        print("No winners found")
        raise ValueError("No winners found")

    # Prepare common plot data
    players_string, abbreviated_names, standard_colors, colors = prepare_plot_data(game_log)

    # Create a list of wins for all players, defaulting to 0 if a player has no wins
    round_winners = [round_winners_dict.get(player, 0) for player in players_string]

    # Prepare data and labels for Bar plot
    bar_plot_data = (abbreviated_names, round_winners, colors)
    bar_plot_labels = ("Total Wins per Player", "Players", "Round Wins")

    # Prepare data and labels for Pie chart
    total_rounds = sum(round_winners)
    pie_plot_data = (round_winners, abbreviated_names, colors)
    pie_plot_labels = ("Round Win Rates", total_rounds)

    # Prepare the data and labels for Box plot
    round_wins_per_game_dict: dict["Agent", list[int]] = game_log.get_round_wins_per_game()
    round_wins_per_game: list[list[int]] = [
        round_wins_per_game_dict.get(player, [0] * len(game_log.game_states))
        for player in game_log.all_game_players
    ]

    box_plot_data = (round_wins_per_game, abbreviated_names, colors)
    box_plot_labels = ("Distribution of Wins Across Games", "Players", "Round Wins")

    return create_figure("Apples to Apples - Round Winners", players_string, colors, game_log, 
                         change_players_between_games, cycle_starting_judges, reset_models_between_games, 
                         use_extra_vectors, use_losing_red_apples, bar_plot_data, pie_plot_data, box_plot_data, 
                         bar_plot_labels, pie_plot_labels, box_plot_labels)


def create_game_winners_plot(game_winners_dict: dict[str, int], game_log: GameLog, change_players_between_games: bool,
                            cycle_starting_judges: bool, reset_models_between_games: bool, use_extra_vectors: bool, use_losing_red_apples: bool) -> Figure:
    # Check if there are any winners
    if not game_winners_dict:
        print("No winners found")
        raise ValueError("No winners found")

    # Prepare common plot data
    players_string, abbreviated_names, standard_colors, colors = prepare_plot_data(game_log)

    # Create a list of wins for all players, defaulting to 0 if a player has no wins
    game_winners = [game_winners_dict.get(player, 0) for player in players_string]

    # Prepare data and labels for Bar plot
    bar_plot_data = (abbreviated_names, game_winners, colors)
    bar_plot_labels = ("Total Wins per Player", "Players", "Game Wins")

    # Prepare data and labels for Pie chart
    pie_plot_data = (game_winners, abbreviated_names, colors)
    pie_plot_labels = ("Game Win Rates", game_log.total_games)

    # Prepare the data and labels for Box plot
    rounds_per_game_dict: dict[int, int] = game_log.get_rounds_per_game()
    rounds_per_game: list[int] = [rounds_per_game_dict.get(i, 0) for i in range(game_log.total_games)]
    box_plot_data = (rounds_per_game, [f"Game {i + 1}" for i in range(game_log.total_games)], colors)
    box_plot_labels = ("Distribution of Rounds Across Games", "Games", "Rounds")

    return create_figure("Apples to Apples - Game Winners", players_string, colors, game_log, 
                         change_players_between_games, cycle_starting_judges, reset_models_between_games, 
                         use_extra_vectors, use_losing_red_apples, bar_plot_data, pie_plot_data, box_plot_data, 
                         bar_plot_labels, pie_plot_labels, box_plot_labels)


def save_plot(plot_figure: Figure, output_filepath: str) -> None:
    # Save the plot to a file
    plot_figure.savefig(output_filepath)


def main(game_log: GameLog, change_players_between_games: bool,
            cycle_starting_judges: bool, reset_models_between_games: bool,
            use_extra_vectors: bool, use_losing_red_apples: bool) -> None:
    # Get the winners dictionary
    try:
        round_winners = count_winners(game_log.round_winners_csv_filepath, "round_winner")
        game_winners = count_winners(game_log.game_winners_csv_filepath, "game_winner")

        # Print the game info
        print(f"\n|| DATA ANALYSIS ||")
        print(f"\nPoints to win: {game_log.points_to_win}")
        print(f"Total games: {game_log.total_games}", end="\n\n")

        # Print the winners table
        print_winners_table(game_log, game_winners, round_winners)

        # Generate round winners output filename
        round_winners_base_name = os.path.splitext(game_log.round_winners_csv_filepath)[0]
        round_winners_output_filepath = f"{round_winners_base_name}.png"

        # Create a plot of the round winners
        round_winners_plot = create_round_winners_plot(
            round_winners, game_log,
            change_players_between_games,
            cycle_starting_judges,
            reset_models_between_games,
            use_extra_vectors,
            use_losing_red_apples
            )
        
        # Save the plot to a file
        save_plot(round_winners_plot, round_winners_output_filepath)

        # Display the plot
        plt.show()

        # Generate game winners output filename
        game_winners_base_name = os.path.splitext(game_log.game_winners_csv_filepath)[0]
        game_winners_output_filepath = f"{game_winners_base_name}.png"

        # Create a plot of the game winners
        game_winners_plot = create_game_winners_plot(
            game_winners, game_log,
            change_players_between_games,
            cycle_starting_judges,
            reset_models_between_games,
            use_extra_vectors,
            use_losing_red_apples
            )

        # Save the plot to a file
        save_plot(game_winners_plot, game_winners_output_filepath)

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
    parser.add_argument("game_log", help="GameLog object with all GameState and RoundState data.")
    parser.add_argument("change_players_between_games", help="Change players between games (y/n).")
    parser.add_argument("cycle_starting_judges", help="Cycle starting judges between games (y/n).")
    parser.add_argument("reset_models_between_games", help="Reset models between games (y/n).")
    parser.add_argument("use_extra_vectors", help="Use extra vectors (y/n).")
    parser.add_argument("use_losing_red_apples", help="Use losing red apples (y/n).")

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(
        args.game_log,
        args.change_players_between_games,
        args.cycle_starting_judges,
        args.reset_models_between_games,
        args.use_extra_vectors,
        args.use_losing_red_apples
    )
