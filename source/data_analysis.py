# Description: Script to count the number of times each unique player has won a game

# Standard Libraries
import argparse
import logging
from typing import cast
import numpy as np
import csv

# Third-party Libraries
from tabulate import tabulate
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
from source.agent import Agent, AIAgent
from source.data_classes import GameLog
from source.game_logger import LOGGING_BASE_DIRECTORY


def calculate_win_counts(game_log: GameLog) -> tuple[dict, dict]:
    """
    Calculate the win counts for each AI agent for both rounds and games.

    Args:
        game_log (GameLog): The game log containing the results.

    Returns:
        tuple: A tuple containing two dictionaries:
            - The first dictionary has agent names as keys and round win counts as values.
            - The second dictionary has agent names as keys and game win counts as values.
    """
    round_win_counts = {}
    game_win_counts = {}

    for game_state in game_log.game_states:
        game_winner = game_state.game_winner

        if game_winner:
            game_win_counts[game_winner.get_name()] = game_win_counts.get(game_winner.get_name(), 0) + 1

        for round_state in game_state.round_states:
            round_winner = round_state.round_winner
            if round_winner:
                round_win_counts[round_winner.get_name()] = round_win_counts.get(round_winner.get_name(), 0) + 1

    return round_win_counts, game_win_counts


def calculate_win_rates(game_log: GameLog) -> tuple[dict, dict]:
    """
    Calculate the win rates for each AI agent for both rounds and games.

    Args:
        game_log (GameLog): The game log containing the results.

    Returns:
        tuple: A tuple containing two dictionaries:
            - The first dictionary has agent names as keys and round win rates as values.
            - The second dictionary has agent names as keys and game win rates as values.
    """
    round_win_counts, game_win_counts = calculate_win_counts(game_log)
    total_rounds = sum(len(game_state.round_states) for game_state in game_log.game_states)
    total_games = game_log.total_games

    round_win_rates = {agent: count / total_rounds for agent, count in round_win_counts.items()}
    game_win_rates = {agent: count / total_games for agent, count in game_win_counts.items()}

    return round_win_rates, game_win_rates


def calculate_round_wins_per_game(game_log: GameLog) -> dict[Agent, list[int]]:
    """
    Calculate the number of round wins per game for each AI agent.

    Args:
        game_log (GameLog): The game log containing the results.

    Returns:
        dict: A dictionary with agent objects as keys and lists of round wins per game as values.
    """
    # Initialize the dictionary with each player having a list of zeros for each game
    round_wins_per_game: dict[Agent, list[int]] = {player: [0] * len(game_log.game_states) for player in game_log.all_game_players}

    # Iterate through each game
    for game_index, game in enumerate(game_log.game_states):
        logging.debug(f"Collecting round wins from Game {game.current_game}")

        # Iterate through each round in the game
        for round_state in game.round_states:
            round_winner = round_state.round_winner
            logging.debug(f"Round {round_state.current_round} winner: {round_winner.get_name() if round_winner else None}")

            # If the round has no winner, skip it
            if round_winner is None:
                logging.debug(f"Skipping round {round_state.current_round} because it has no winner.")
                continue

            # Increment the round win count for the winning player
            round_wins_per_game[round_winner][game_index] += 1
            for player, wins in round_wins_per_game.items():
                logging.debug(f"Round wins for {player.get_name()}: {wins}")

    return round_wins_per_game


def calculate_standard_deviation(round_wins_per_game: dict[Agent, list[int]]) -> dict:
    """
    Calculate the standard deviation for the win rates.

    Args:
        win_rates (dict): A dictionary with agent names as keys and win rates as values.
        total_games (int): The total number of games played.

    Returns:
        dict: A dictionary with agent names as keys and standard deviations as values.
    """
    std_devs = {}
    for agent, win_counts in round_wins_per_game.items():
        std_dev = np.std(win_counts)
        std_devs[agent.get_name()] = std_dev
    return std_devs


def calculate_confidence_intervals(win_rates: dict[Agent, float], total_games: int, confidence_level: float = 0.95) -> dict:
    """
    Calculate confidence intervals for the win rates.

    Args:
        win_rates (dict): A dictionary with agent names as keys and win rates as values.
        total_games (int): The total number of games played.
        confidence_level (float): The confidence level for the intervals.

    Returns:
        dict: A dictionary with agent names as keys and confidence intervals as values.
    """
    confidence_intervals = {}
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    for agent, win_rate in win_rates.items():
        win_count = win_rate * total_games
        interval = z * np.sqrt((win_rate * (1 - win_rate)) / total_games)
        confidence_intervals[agent] = (win_rate - interval, win_rate + interval)
    return confidence_intervals


def perform_statistical_tests(win_rates: dict[Agent, float], total_games: int) -> dict:
    """
    Perform statistical tests to determine if the win rates are due to chance.

    Args:
        win_rates (dict): A dictionary with agent names as keys and win rates as values.
        total_games (int): The total number of games played.

    Returns:
        dict: A dictionary with agent names as keys and p-values as values.
    """
    p_values = {}
    for agent, win_rate in win_rates.items():
        win_count = int(win_rate * total_games)  # Convert win_count to an integer
        p_value = binomtest(win_count, total_games, 1/len(win_rates), alternative="greater").pvalue
        p_values[agent] = p_value
    return p_values


def prepare_round_win_stats(game_log: GameLog) -> list[list]:
    # Calculate win rates
    round_win_rates, _ = calculate_win_rates(game_log)

    # Calculate win counts
    round_win_counts, _ = calculate_win_counts(game_log)

    # Calculate round wins per game for each player
    round_wins_per_game_dict = calculate_round_wins_per_game(game_log)

    # Calculate the average number of round wins per game for each player
    round_wins_per_game_avg = {
        player.get_name(): sum(wins) / len(wins) if len(wins) > 0 else 0
        for player, wins in round_wins_per_game_dict.items()
    }

    # Calculate standard deviations for round win rates
    round_std_devs = calculate_standard_deviation(round_wins_per_game_dict)

    # Calculate confidence intervals for round win rates
    round_conf_intervals = calculate_confidence_intervals(round_win_rates, sum(len(game_state.round_states) for game_state in game_log.game_states))

    # Perform statistical tests for round win rates
    round_p_values = perform_statistical_tests(round_win_rates, sum(len(game_state.round_states) for game_state in game_log.game_states))

    # Prepare the data for the round winners table
    round_table_data = []
    for player in game_log.all_game_players:
        player_name = player.get_name()
        round_win_count = round_win_counts.get(player_name, 0)
        round_win_rate = round_win_rates.get(player_name, 0)
        round_conf_interval = round_conf_intervals.get(player_name, (0, 0))
        round_p_value = round_p_values.get(player_name, 1)
        mean_wins_per_game = round_wins_per_game_avg.get(player_name, 0)
        round_std_dev = round_std_devs.get(player_name, 0)
        round_table_data.append([
            player_name,
            round_win_count, f"{round_win_rate:.2%}", f"{round_conf_interval[0]:.2%} to {round_conf_interval[1]:.2%}", f"{round_p_value:.5f}", f"{mean_wins_per_game:.3f}", f"{round_std_dev:.3f}"
        ])

    # Sort the round table data by player name alphabetically
    round_table_data = sorted(round_table_data, key=lambda x: x[0])

    return round_table_data


def prepare_game_win_stats(game_log: GameLog) -> list[list]:
    # Calculate win rates
    _, game_win_rates = calculate_win_rates(game_log)

    # Calculate win counts
    _, game_win_counts = calculate_win_counts(game_log)

    # Calculate confidence intervals for game win rates
    game_conf_intervals = calculate_confidence_intervals(game_win_rates, game_log.total_games)

    # Perform statistical tests for game win rates
    game_p_values = perform_statistical_tests(game_win_rates, game_log.total_games)

    # Prepare the data for the game winners table
    game_table_data = []
    for player in game_log.all_game_players:
        player_name = player.get_name()
        game_win_count = game_win_counts.get(player_name, 0)
        game_win_rate = game_win_rates.get(player_name, 0)
        game_conf_interval = game_conf_intervals.get(player_name, (0, 0))
        game_p_value = game_p_values.get(player_name, 1)
        game_table_data.append([
            player_name,
            game_win_count, f"{game_win_rate:.2%}", f"{game_conf_interval[0]:.2%} to {game_conf_interval[1]:.2%}", f"{game_p_value:.5f}"
        ])

    # Sort the game table data by player name alphabetically
    game_table_data = sorted(game_table_data, key=lambda x: x[0])

    return game_table_data


def print_table(data, headers, title):
    print(title)
    print(tabulate(data, headers=headers, tablefmt="grid"))


def save_to_csv(data, headers, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


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


def prepare_players_and_colors(game_log: GameLog) -> tuple[list[Agent], list[str], list[str], list[str]]:
    """
    Prepares the plot data including players list, player strings, abbreviated names, and colors.

    Args:
        game_log (GameLog): GameLog object containing all the game data.

    Returns:
        Tuple[List[Agent], List[str], List[str], Dict[str, str]]: Tuple containing players list, player strings,
    """
    # Get all the players
    players: list[Agent] = game_log.all_game_players
    players_string: list[str] = [player.get_name() for player in players]

    # Abbreviate player names for x-axis
    abbreviated_names = [abbreviate_name(player) for player in players_string]

    # Define a list of standard, contrasting colors
    standard_colors = ["red", "cyan", "orange", "purple", "lime", "brown", "pink", "yellow", "gray", "black"]

    # Repeat the colors if there are more players than colors
    colors = [standard_colors[i % len(standard_colors)] for i in range(len(players))]

    return players, players_string, abbreviated_names, colors


def prepare_plot_data(game_log: GameLog, win_counts: dict) -> tuple:
    """
    Prepare data for plotting.

    Args:
        game_log (GameLog): The game log containing the results.
        win_counts (dict): A dictionary with agent names as keys and win counts as values.

    Returns:
        tuple: Prepared data for plotting.
    """
    players, players_string, abbreviated_names, colors = prepare_players_and_colors(game_log)
    wins = [win_counts.get(player, 0) for player in players_string]
    return abbreviated_names, wins, colors


def create_legend(ax: Axes, full_names: list[str], colors: list[str]) -> None:
    # Remove the axis borders
    ax.axis("off")

    # Add a title to the legend
    ax.set_title("Legend", fontsize=18, fontweight="bold")

    # Create a legend for the plot
    handles = [Rectangle((0,0),1,1, color=color) for color in colors]
    ax.legend(handles, full_names, title="Players", loc="upper center", fontsize=12, title_fontsize=14)


def create_game_settings_box(ax: Axes, points_to_win: int, total_games: int,
                             change_players_between_games: bool, cycle_starting_judges: bool,
                             reset_models_between_games: bool, use_extra_vectors: bool) -> None:
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
                    f"use_extra_vectors = {use_extra_vectors}")

    # Add the settings text closer to the title
    ax.text(0.5, 0.45, settings_text, fontsize=12, fontweight="bold",
            horizontalalignment="center", verticalalignment="top")


def create_bar_plot(ax: Axes, abbrev_names: list[str], wins: list[int], colors: list[str], title: str, xlabel: str, ylabel: str) -> None:
    # Create bar plot
    ax.bar(abbrev_names, wins, color=colors)

    # Set titles and labels
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add grid for better readability
    ax.grid(True)


def create_pie_chart(ax: Axes, abbrev_names: list[str], wins: list[int], colors: list[str], title: str, total_rounds_games: int) -> None:
    # Filter out data and labels that correspond to 0%
    filtered_wins = []
    filtered_names = []
    filtered_colors = []
    for i, value in enumerate(wins):
        if value > 0:
            filtered_wins.append(value)
            filtered_names.append(abbrev_names[i])
            filtered_colors.append(colors[i])

    # Create pie chart
    pie_result = ax.pie(filtered_wins, labels=filtered_names, colors=filtered_colors,
                        autopct="%1.1f%%", startangle=140)

    # Set font weight and size for pie chart
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_fontsize(14)

    # Add a title to the pie chart
    ax.set_title(title, fontsize=18, fontweight="bold")

    # Calculate the percentage of games won by AI agents
    ai_wins = sum(wins[i] for i, label in enumerate(abbrev_names) if "AI" in label)
    percent_ai = ai_wins / total_rounds_games * 100 if total_rounds_games > 0 else 0

    # Add the percentage of games won by AI agents as a title below the pie chart
    ax.text(0.5, -0.1, f"AI Wins: {percent_ai:.2f}%", ha="center", va="center", fontsize=14, transform=ax.transAxes)


def create_box_plot(ax: Axes, abbrev_names: list[str], wins_per_game: list[list[int]], colors: list[str], title: str, xlabel: str, ylabel: str) -> None:
    # Convert colors to RGBA format if necessary
    rgba_colors = [to_rgba(color) for color in colors]

    # Extend colors to match the length of data
    extended_colors = [rgba_colors[i % len(rgba_colors)] for i in range(len(wins_per_game))]

    # Create box plot
    box = ax.boxplot(wins_per_game, patch_artist=True)

    # Apply RGBA colors to the box plot
    for patch, color in zip(box["boxes"], extended_colors):
        patch.set_facecolor(color)
        patch.set_linewidth(2)

    # Set line width for other elements
    for element in ["whiskers", "caps", "medians", "fliers"]:
        plt.setp(box[element], linewidth=2)

    # Set titles and labels
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")

    # Ensure the number of ticks matches the number of labels
    ax.set_xticks(range(1, len(abbrev_names) + 1))
    ax.set_xticklabels(abbrev_names, rotation=45, ha="right")

    # Add grid for better readability
    ax.grid(True)


def create_line_graph(ax: Axes, rounds_per_game: list[int], game_labels: list[str], colors: list[str],
                      title: str, xlabel: str, ylabel: str) -> None:
    # Create a line graph
    ax.plot(game_labels, rounds_per_game, marker='o', color='b', linestyle='-')

    # Set titles and labels
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")

    # Ensure y-axis only shows whole non-negative integers
    ax.set_ylim(bottom=0)  # Set the lower limit to 0
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure only whole numbers are shown

    # Add grid for better readability
    ax.grid(True)


def create_vector_line_graph(ax: Axes, ai_agent: AIAgent, opponents: list[Agent], game_log: GameLog) -> None:
    # Initialize the dictionary
    vector_dict: dict["Agent", dict[str, list[np.ndarray]]] = game_log.get_slope_and_bias_history_by_player(ai_agent)

    # Create a secondary y-axis for bias
    ax_bias: Axes = cast(Axes, ax.twinx())

    # Collect data points for the line graph and plot them
    for opponent in opponents:
        # Get the slope and bias vectors
        if isinstance(opponent, AIAgent):
            slope_target, bias_target = opponent.get_self_slope_and_bias_vectors()
        else:
            vector_shape = vector_dict[opponent]["slope"][0].shape
            slope_target = [np.zeros(vector_shape) for _ in range(len(vector_dict[opponent]["slope"]))]
            bias_target = [np.zeros(vector_shape) for _ in range(len(vector_dict[opponent]["bias"]))]
        slope_predict, bias_predict = vector_dict[opponent]["slope"], vector_dict[opponent]["bias"]

        # Calculate Euclidean distances for slope and bias
        eucledian_distance_slope: list[float] = [float(np.linalg.norm(sp - st)) for sp, st in zip(slope_predict, slope_target)]
        eucledian_distance_bias: list[float] = [float(np.linalg.norm(bp - bt)) for bp, bt in zip(bias_predict, bias_target)]

        # Plot the slope data points
        x_data = range(len(vector_dict[opponent]["slope"]))
        y_data_slope = eucledian_distance_slope
        ax.plot(x_data, y_data_slope, label=f"Slope - {opponent.get_name()}", linestyle='-', marker='o')

        # Plot the bias data points
        y_data_bias = eucledian_distance_bias
        ax_bias.plot(x_data, y_data_bias, label=f"Bias - {opponent.get_name()}", linestyle='--', marker='x')

    # Set titles and labels
    ax.set_title(f"AI Agent: {ai_agent.get_name()} - Slope and Bias")
    ax.set_xlabel("Game Round")
    ax.set_ylabel("Euclidean Distance (Slope)")
    ax_bias.set_ylabel("Euclidean Distance (Bias)")

    # Add legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_bias.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    # Add grid for better readability
    ax.grid(True)


def create_round_winners_plot(game_log: GameLog, change_players_between_games: bool,
                            cycle_starting_judges: bool, reset_models_between_games: bool,
                            use_extra_vectors: bool) -> Figure:
    """
    Create a plot for round winners.

    Args:
        game_log (GameLog): The game log containing the results.
        change_players_between_games (bool): Whether players change between games.
        cycle_starting_judges (bool): Whether starting judges cycle.
        reset_models_between_games (bool): Whether models reset between games.
        use_extra_vectors (bool): Whether extra vectors are used.

    Returns:
        Figure: The created plot figure.
    """
    # Get the winners dictionary
    round_winners_dict = calculate_win_counts(game_log)[0]

    # Check if there are any winners
    if not round_winners_dict:
        print("No winners found")
        raise ValueError("No winners found")

    # Prepare common plot data
    players, players_string, abbreviated_names, colors = prepare_players_and_colors(game_log)

    # Create a list of wins for all players, defaulting to 0 if a player has no wins
    round_winners = [round_winners_dict.get(player, 0) for player in players_string]

    # Prepare data and labels for Bar plot
    bar_plot_data = (abbreviated_names, round_winners, colors)
    bar_plot_labels = ("Total Wins per Player", "Players", "Round Wins")

    # Prepare data and labels for Pie chart
    total_rounds = sum(round_winners)
    pie_plot_data = (abbreviated_names, round_winners, colors)
    pie_plot_labels = ("Round Win Rates", total_rounds)

    # Prepare the data and labels for Box plot TODO - try to make 0 visible on the box plot, try to make it look better
    round_wins_per_game_dict: dict[Agent, list[int]] = calculate_round_wins_per_game(game_log)
    round_wins_per_game: list[list[int]] = [
        round_wins_per_game_dict.get(player, [0] * len(game_log.game_states))
        for player in game_log.all_game_players
    ]

    box_plot_data = (abbreviated_names, round_wins_per_game, colors)
    box_plot_labels = ("Distribution of Wins Across Games", "Players", "Round Wins")

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[5, 4], figure=fig)

    # Add a custom title with underlining
    title = "Apples to Apples - Round Winners"
    custom_title = f"{title}\n{'_' * len(title)}\n"
    fig.suptitle(custom_title, fontsize=24, fontweight="bold")

    # Legend
    legend_ax = fig.add_subplot(gs[0, 2])
    create_legend(legend_ax, players_string, colors)

    # Game settings box
    game_settings_ax: Axes = fig.add_subplot(gs[1, 2])
    create_game_settings_box(game_settings_ax, game_log.points_to_win, game_log.total_games,
                             change_players_between_games, cycle_starting_judges, reset_models_between_games,
                             use_extra_vectors)

    # Create subplots for the bar, pie, and box plots
    bar_ax = fig.add_subplot(gs[0, 0])
    pie_ax = fig.add_subplot(gs[0, 1])
    box_ax = fig.add_subplot(gs[1, 0:2])

    # Create plots
    create_bar_plot(bar_ax, *bar_plot_data, *bar_plot_labels)
    create_pie_chart(pie_ax, *pie_plot_data, *pie_plot_labels)
    create_box_plot(box_ax, *box_plot_data, *box_plot_labels)

    # Adjust layout
    plt.tight_layout()

    return fig


def create_game_winners_plot(game_log: GameLog, change_players_between_games: bool,
                            cycle_starting_judges: bool, reset_models_between_games: bool,
                            use_extra_vectors: bool) -> Figure:
    """
    Create a plot for game winners.

    Args:
        game_log (GameLog): The game log containing the results.
        change_players_between_games (bool): Whether players change between games.
        cycle_starting_judges (bool): Whether starting judges cycle.
        reset_models_between_games (bool): Whether models reset between games.
        use_extra_vectors (bool): Whether extra vectors are used.

    Returns:
        Figure: The created plot figure.
    """
    # Get the winners dictionary
    game_winners_dict = calculate_win_counts(game_log)[1]

    # Check if there are any winners
    if not game_winners_dict:
        print("No winners found")
        raise ValueError("No winners found")

    # Prepare common plot data
    players, players_string, abbreviated_names, colors = prepare_players_and_colors(game_log)

    # Create a list of wins for all players, defaulting to 0 if a player has no wins
    game_winners = [game_winners_dict.get(player, 0) for player in players_string]

    # Prepare data and labels for Bar plot
    bar_plot_data = (abbreviated_names, game_winners, colors)
    bar_plot_labels = ("Total Wins per Player", "Players", "Game Wins")

    # Prepare data and labels for Pie chart
    pie_plot_data = (abbreviated_names, game_winners, colors)
    pie_plot_labels = ("Game Win Rates", game_log.total_games)

    # Prepare the data and labels for Box plot TODO - fix box plot data to be a bar chart with total rounds per game
    rounds_per_game_dict: dict[int, int] = game_log.get_rounds_per_game()
    rounds_per_game: list[int] = [rounds_per_game_dict.get(i + 1, 0) for i in range(game_log.total_games)]
    line_plot_data = (rounds_per_game, [f"{i + 1}" for i in range(game_log.total_games)], colors)
    line_plot_labels = ("Distribution of Rounds Across Games", "Games", "Rounds")

    # Calculate min, max, and average rounds per game
    min_rounds = min(rounds_per_game)
    max_rounds = max(rounds_per_game)
    avg_rounds = sum(rounds_per_game) / len(rounds_per_game)

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[5, 4], figure=fig)

    # Add a custom title with underlining
    title = "Apples to Apples - Game Winners"
    custom_title = f"{title}\n{'_' * len(title)}\n"
    fig.suptitle(custom_title, fontsize=24, fontweight="bold")

    # Legend
    legend_ax = fig.add_subplot(gs[0, 2])
    create_legend(legend_ax, players_string, colors)

    # Game settings box
    game_settings_ax: Axes = fig.add_subplot(gs[1, 2])
    create_game_settings_box(game_settings_ax, game_log.points_to_win, game_log.total_games,
                             change_players_between_games, cycle_starting_judges, reset_models_between_games,
                             use_extra_vectors)

    # Create subplots for the bar, pie, and line plots
    bar_ax = fig.add_subplot(gs[0, 0])
    pie_ax = fig.add_subplot(gs[0, 1])
    line_ax = fig.add_subplot(gs[1, 0:2])

    # Create plots
    create_bar_plot(bar_ax, *bar_plot_data, *bar_plot_labels)
    create_pie_chart(pie_ax, *pie_plot_data, *pie_plot_labels)
    create_line_graph(line_ax, *line_plot_data, *line_plot_labels)

    # Add min, max, and average rounds per game as text annotations
    stats_text = f"Min Rounds: {min_rounds}\nMax Rounds: {max_rounds}\nAvg Rounds: {avg_rounds:.2f}"
    line_ax.text(0.95, 0.05, stats_text, transform=line_ax.transAxes, fontsize=12,
                 verticalalignment="bottom", horizontalalignment="right", bbox=dict(facecolor="white", alpha=0.5))

    # Adjust layout
    plt.tight_layout()

    return fig


def create_heatmap(game_log: GameLog) -> Figure:
    # Prepare common plot data
    players, players_string, abbreviated_names, colors = prepare_players_and_colors(game_log)

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


def create_vector_history_plot(game_log: GameLog) -> Figure:
    # Identify all AI agents
    ai_agents = [player for player in game_log.all_game_players if isinstance(player, AIAgent)]
    num_ai_agents = len(ai_agents)

    # Determine the number of rows and columns for the GridSpec
    num_cols = 2
    num_rows = (num_ai_agents + num_cols - 1) // num_cols

    # Create a figure with GridSpec
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(nrows=num_rows, ncols=num_cols, figure=fig)

    # Add a custom title with underlining
    title = "Apples to Apples - Vector History"
    custom_title = f"{title}\n{'_' * len(title)}\n"
    fig.suptitle(custom_title, fontsize=24, fontweight="bold")

    # Create subplots for each AI agent
    for i, ai_agent in enumerate(ai_agents):
        row = i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(gs[row, col])

        # Get the opponents for the AI agent
        opponents = [player for player in game_log.all_game_players if player != ai_agent]
        create_vector_line_graph(ax, ai_agent, opponents, game_log)

    # Adjust layout
    plt.tight_layout()

    return fig


def save_plot(plot_figure: Figure, output_filepath: str) -> None:
    try:
        # Save the plot to a file
        plot_figure.savefig(output_filepath)
    except Exception as e:
        print(f"An error occurred while saving the plot: {e}")
        raise e


def main(game_log: GameLog, change_players_between_games: bool,
            cycle_starting_judges: bool, reset_models_between_games: bool,
            use_extra_vectors: bool) -> None:
    # Print the game info
    print(f"\n|| DATA ANALYSIS ||")
    print(f"\nPoints to win: {game_log.points_to_win}")
    print(f"Total games: {game_log.total_games}", end="\n\n")

    # Prepare stats
    round_table_data = prepare_round_win_stats(game_log)
    game_table_data = prepare_game_win_stats(game_log)

    # Define headers
    round_headers = ["Player", "Win Count", "Win Rate", "Confidence Interval", "P-Value", "Mean Wins per Game", "Std Dev"]
    game_headers = ["Player", "Win Count", "Win Rate", "Confidence Interval", "P-Value"]

    # Print tables
    print_table(round_table_data, round_headers, "ROUND WINNERS TABLE")
    print_table(game_table_data, game_headers, "\nGAME WINNERS TABLE")

    # Generate csv output filepaths
    round_stats_output_filepath = f"{LOGGING_BASE_DIRECTORY}{game_log.naming_scheme}/round_win_stats-{game_log.naming_scheme}.csv"
    game_stats_output_filepath = f"{LOGGING_BASE_DIRECTORY}{game_log.naming_scheme}/game_win_stats-{game_log.naming_scheme}.csv"

    # Save to CSV
    save_to_csv(round_table_data, round_headers, round_stats_output_filepath)
    save_to_csv(game_table_data, game_headers, game_stats_output_filepath)

    # Generate round winners output filepath
    round_winners_output_filepath = f"{LOGGING_BASE_DIRECTORY}{game_log.naming_scheme}/round_winners-{game_log.naming_scheme}.png"

    # Create a plot of the round winners
    round_winners_plot = create_round_winners_plot(
        game_log, change_players_between_games, cycle_starting_judges,
        reset_models_between_games,use_extra_vectors
        )

    # Save the plot to a file
    save_plot(round_winners_plot, round_winners_output_filepath)

    # Display the plot
    plt.show()

    # Generate game winners output filepath
    game_winners_output_filepath = f"{LOGGING_BASE_DIRECTORY}{game_log.naming_scheme}/game_winners-{game_log.naming_scheme}.png"

    # Create a plot of the game winners
    game_winners_plot = create_game_winners_plot(
        game_log, change_players_between_games, cycle_starting_judges,
        reset_models_between_games, use_extra_vectors
        )

    # Save the plot to a file
    save_plot(game_winners_plot, game_winners_output_filepath)

    # Display the plot
    plt.show()

    # Generate judge heatmap output filepath
    judge_heatmap_output_filepath = f"{LOGGING_BASE_DIRECTORY}{game_log.naming_scheme}/judge_heatmap-{game_log.naming_scheme}.png"

    # Create a plot of the judge heatmap
    judge_heatmap_plot = create_heatmap(game_log)

    # Save the plot to a file
    save_plot(judge_heatmap_plot, judge_heatmap_output_filepath)

    # Display the plot
    plt.show()

    # Generate vector history output filepath
    vector_history_output_filepath = f"{LOGGING_BASE_DIRECTORY}{game_log.naming_scheme}/vector_history-{game_log.naming_scheme}.png"


    # Create a plot of the vector history
    vector_history_plot = create_vector_history_plot(game_log)

    # Save the plot to a file
    save_plot(vector_history_plot, vector_history_output_filepath)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Count round winners and game winners from the resulting game log.")

    # Add an argument for the filename as input
    parser.add_argument("game_log", help="GameLog object with all GameState and RoundState data.")
    parser.add_argument("change_players_between_games", help="Change players between games (y/n).")
    parser.add_argument("cycle_starting_judges", help="Cycle starting judges between games (y/n).")
    parser.add_argument("reset_models_between_games", help="Reset models between games (y/n).")
    parser.add_argument("use_extra_vectors", help="Use extra vectors (y/n).")

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(
        args.game_log,
        args.change_players_between_games,
        args.cycle_starting_judges,
        args.reset_models_between_games,
        args.use_extra_vectors
    )
