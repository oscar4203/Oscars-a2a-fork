# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse
import time

# Third-party Libraries

# Local Modules
from source.embeddings import Embedding
from source.apples_to_apples import ApplesToApples
from source.game_logger import configure_logging
from source.data_analysis import main as data_analysis_main
from source.data_classes import GameLog


class GameDriver:
    def __init__(self, training_mode: bool, number_of_players: int, points_to_win: int, total_games: int) -> None:
        # Set the game state for training mode
        if training_mode:
            number_of_players = 2 # Override the number of players for training mode
            max_cards_in_hand = 25
        else: # Set the game state for non-training mode
            max_cards_in_hand = 7

        # Initialize the GameLog
        self.game_log: GameLog = GameLog()
        self.game_log.intialize_input_args(number_of_players, max_cards_in_hand, points_to_win, total_games)

    def load_keyed_vectors(self, use_custom_loader: bool) -> None:
        start = time.perf_counter()
        self.embedding = Embedding("./apples/GoogleNews-vectors-negative300.bin", custom=use_custom_loader)
        end = time.perf_counter()
        print("Loaded Vectors in", end-start, "seconds.")


def range_type(min_value, max_value):
    def range_checker(value):
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError(f"Value must be between {min_value} and {max_value}")
        return ivalue
    return range_checker


def get_user_input_y_or_n(prompt: str) -> str:
    while True:
        response = input(prompt)
        if response in ["y", "n"]:
            return response
        print("Invalid input. Type in either 'y' or 'n'.")


def main() -> None:
    # Define the command line arguments
    parser = argparse.ArgumentParser(
        prog="'Apples to Apples' card game",
        usage="python apples_to_apples.py <number_of_players> <points_to_win> <total_games> "\
              "[green_expansion] [red_expansion] [-A] [-V] [-T] [-D]"\
              "\n\nExample: python apples_to_apples.py 4 5 10 green_apples_extension.csv red_apples_extension.csv -A -V -T -D"\
              "\nFor help: python apples_to_apples.py -h",
        description="Configure and run the 'Apples to Apples' game. Specify the number of players, "\
                    "points to win, and total games to play. Include optional green and red apple expansions. "\
                    "Use the -A flag to load all available card packs. "\
                    "Use the -V flag to use the custom vector loader. "\
                    "Use the -T flag to run the program in training mode. "\
                    "Use the -D flag to enable debug mode for detailed logging."
    )

    # Add the command line arguments
    parser.add_argument("number_of_players", type=range_type(3, 10), help="Total number of players (3-10).")
    parser.add_argument("points_to_win", type=range_type(1, 10), help="Total number of points to win (1-10).")
    parser.add_argument("total_games", type=range_type(1, 1000), help="Total number of games to play (1-1000).")
    parser.add_argument("green_expansion", type=str, nargs='?', default='', help="Filename to a green apple expansion (optional).")
    parser.add_argument("red_expansion", type=str, nargs='?', default='', help="Filename to a red apple expansion (optional).")
    parser.add_argument("-A", "--load_all_packs", action="store_true", help="Load all available card packs")
    parser.add_argument("-V", "--vector_loader", action="store_true", help="Use the custom vector loader")
    parser.add_argument("-T", "--training_mode", action="store_true", help="Train a user specified model archetype")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug mode for detailed logging")

    # Parse the command line arguments
    args = parser.parse_args()

    # Configure and initialize the logging module
    configure_logging(args.debug)

    # Log the command line arguments
    logging.info(f"Command line arguments: {args}")
    logging.info(f"Number of players: {args.number_of_players}")
    logging.info(f"Points to win: {args.points_to_win}")
    logging.info(f"Total games to be played: {args.total_games}")
    logging.info(f"Green card expansion file: {args.green_expansion}")
    logging.info(f"Red card expansion file: {args.red_expansion}")
    logging.info(f"Load all card packs: {args.load_all_packs}")
    logging.info(f"Use custom vector loader: {args.vector_loader}")
    logging.info(f"Training mode: {args.training_mode}")
    logging.info(f"Debug mode: {args.debug}")

    # Create the game driver object
    print("Starting 'Apples to Apples' game driver.")
    logging.info("Starting 'Apples to Apples' game driver.")
    game_driver = GameDriver(args.training_mode, args.number_of_players, args.points_to_win, args.total_games)

    # Load the keyed vectors
    game_driver.load_keyed_vectors(args.vector_loader)

    # Create the game object
    a2a_game = ApplesToApples(game_driver.embedding, args.training_mode, args.load_all_packs, args.green_expansion, args.red_expansion)

    # Set the static game log
    a2a_game.initalize_game_log(game_driver.game_log)

    # Initialize all between game option variables
    change_players_between_games = "n"
    cycle_starting_judges = "n"
    reset_models_between_games = "n"
    use_extra_vectors = "y"
    reset_cards_between_games = "n"
    print_in_terminal = "y"

    # Prompt the user on whether they want to change players between games
    if not args.training_mode:
        change_players_between_games = get_user_input_y_or_n("Do you want to change players between games? (y/n): ")

        # Prompt the user on whether they want to cycle the starting judge between games
        if change_players_between_games == "n":
            cycle_starting_judges = get_user_input_y_or_n("Do you want to cycle the starting judge between games? (y/n): ")

        # Prompt the user on whether they want to reset the opponent model vectors between games
        reset_models_between_games = get_user_input_y_or_n("Do you want to reset the opponent models between games? (y/n): ")

        # Prompt the user on whether they want to include the synonym and description vectors inthe model
        use_extra_vectors = get_user_input_y_or_n("Do you want to include the extra synonym and description vectors in the model training? (y/n): ")

        # Prompt the user on whether they want to print the game info and results in the terminal
        print_in_terminal = get_user_input_y_or_n("Do you want to print the game info and results in the terminal? (y/n): ")

    # Prompt the user on whether they want to reset the training cards between games
    if args.training_mode:
        reset_cards_between_games = get_user_input_y_or_n("Do you want to reset the training cards between games? (y/n): ")

    # Set the game options
    a2a_game.set_game_options(
        change_players_between_games == 'y',
        cycle_starting_judges == 'y',
        reset_models_between_games == 'y',
        use_extra_vectors == 'y',
        reset_cards_between_games == 'y',
        print_in_terminal == 'y'
    )

    # Log the game options
    logging.info(f"Change players between games: {change_players_between_games == 'y'}")
    logging.info(f"Cycle starting judges: {cycle_starting_judges == 'y'}")
    logging.info(f"Reset models between games: {reset_models_between_games == 'y'}")
    logging.info(f"Use extra vectors: {use_extra_vectors == 'y'}")
    logging.info(f"Reset training cards between games: {reset_cards_between_games == 'y'}")
    logging.info(f"Print in terminal: {print_in_terminal == 'y'}")

    # Start the game timer
    start = time.perf_counter()

    # Start the first game
    a2a_game.new_game()

    # Continue playing games until the total number of games is reached
    while game_driver.game_log.get_current_game_number() < game_driver.game_log.total_games:
        # Start the next game
        a2a_game.new_game()

    # End the game timer
    end = time.perf_counter()

    # Format the total elapsed time
    total_time = end - start
    hours = int(total_time // 3600)
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    # Print and log the total time elapsed
    print(f"Total time elapsed: {hours} hour(s), {minutes} minute(s), {seconds} second(s)")
    logging.info(f"Total time elapsed: {hours} hour(s), {minutes} minute(s), {seconds} second(s)")

    # Run the winner counter and plot the results, if not in training mode
    if not args.training_mode:
        data_analysis_main(
            game_driver.game_log,
            change_players_between_games == 'y',
            cycle_starting_judges == 'y',
            reset_models_between_games == 'y',
            use_extra_vectors == 'y'
        )

if __name__ == "__main__":
    main()
