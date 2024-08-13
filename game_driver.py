# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse
from datetime import datetime

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules
from source.w2vloader import VectorsW2V
from source.apples_to_apples import ApplesToApples
from source.game_logger import configure_logging
from source.data_analysis import main as data_analysis_main
from source.data_classes import GameState


class GameDriver:
    def __init__(self, training_mode: bool, number_of_players: int, points_to_win: int, total_games: int, green_expansion: str = '', red_expansion: str = '') -> None:
        # Set the game state for training mode
        if training_mode:
            number_of_players = 2
            max_cards_in_hand = 25
        else: # Set the game state for non-training mode
            max_cards_in_hand = 7

        # Initialize the game state
        self.game_state: GameState = GameState(
                        number_of_players, [],
                        max_cards_in_hand,
                        points_to_win,
                        total_games, 0, 0,
                        None, None, [],
                        None, None, None)
        self.green_expansion_filename: str = green_expansion
        self.red_expansion_filename: str = red_expansion

    def load_keyed_vectors(self, use_custom_loader: bool) -> None:
        # if use_custom_loader:
        #     print_and_log("Loading keyed vectors using custom loader...")
        #     self.keyed_vectors = VectorsW2V("./apples/GoogleNews-vectors-negative300.bin")
        # else:
            message = "Loading keyed vectors..."
            print(message)
            logging.info(message)
            self.keyed_vectors = KeyedVectors.load_word2vec_format("./apples/GoogleNews-vectors-negative300.bin", binary=True)


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
    # Take a snapshot of the current date and time
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M)")

    # Define the command line arguments
    parser = argparse.ArgumentParser(
        prog="'Apples to Apples' card game",
        usage="python apples_to_apples.py <number_of_players> <points_to_win> <total_games> "\
              "[green_expansion] [red_expansion] [-V] [-T] [-D]",
        description="Configure and run the 'Apples to Apples' game. Specify the number of players, "\
                    "points to win, and total games to play. Include optional green and red card expansions. "\
                    "Use the -V flag to use the custom vector loader. "\
                    "Use the -T flag to run the program in training mode. "\
                    "Use the -D flag to enable debug mode for detailed logging."
    )

    # Add the command line arguments
    parser.add_argument("number_of_players", type=range_type(3, 8), help="Total number of players (3-8).")
    parser.add_argument("points_to_win", type=range_type(1, 10), help="Total number of points to win (1-10).")
    parser.add_argument("total_games", type=range_type(1,1000), help="Total number of games to play (1-1000).")
    parser.add_argument("green_expansion", type=str, nargs='?', default='', help="Filename to a green card expansion (optional).")
    parser.add_argument("red_expansion", type=str, nargs='?', default='', help="Filename to a red card expansion (optional).")
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
    logging.info(f"Use custom vector loader: {args.vector_loader}")
    logging.info(f"Training mode: {args.training_mode}")
    logging.info(f"Debug mode: {args.debug}")

    # Create the game driver object
    print("Starting 'Apples to Apples' game driver.")
    logging.info("Starting 'Apples to Apples' game driver.")
    game_driver = GameDriver(args.training_mode, args.number_of_players, args.points_to_win, args.total_games, args.green_expansion, args.red_expansion)

    # Load the keyed vectors
    game_driver.load_keyed_vectors(args.vector_loader)

    # Create the game object
    game = ApplesToApples(game_driver.keyed_vectors, args.training_mode, args.green_expansion, args.red_expansion)

    # Set the static game state
    game.initalize_game_state(game_driver.game_state)

    # Initialize all between game option variables
    change_players_between_games = "n"
    cycle_starting_judges = "n"
    reset_models_between_games = "n"
    use_extra_vectors = "n"
    use_losing_red_apples = "n"

    # Prompt the user on whether they want to change players between games
    if not args.training_mode:
        change_players_between_games = get_user_input_y_or_n("Do you want to change players between games? (y/n): ")

    # Prompt the user on whether they want to cycle the starting judge between games
    if change_players_between_games == "n" and not args.training_mode:
        cycle_starting_judges = get_user_input_y_or_n("Do you want to cycle the starting judge between games? (y/n): ")

    # Prompt the user on whether they want to reset the opponent model vectors between games
    reset_models_between_games = get_user_input_y_or_n("Do you want to reset the opponent models between games? (y/n): ")

    # Prompt the user on whether they want to include the synonym and description vectors inthe model
    use_extra_vectors = get_user_input_y_or_n("Do you want to include the synonym and description vectors in the model training? (y/n): ")

    # Prompt the user on whether they want to include the losing red apples in the model training
    use_losing_red_apples = get_user_input_y_or_n("Do you want to include the losing red apples in the model training? (y/n): ")

    # Set the game options
    game.set_game_options(
        change_players_between_games == 'y',
        cycle_starting_judges == 'y',
        reset_models_between_games == 'y',
        use_extra_vectors == 'y',
        use_losing_red_apples == 'y'
    )

    # Log the game options
    logging.info(f"Change players between games: {change_players_between_games == 'y'}")
    logging.info(f"Cycle starting judges: {cycle_starting_judges == 'y'}")
    logging.info(f"Reset models between games: {reset_models_between_games == 'y'}")
    logging.info(f"Use extra vectors: {use_extra_vectors == 'y'}")
    logging.info(f"Use losing red apples: {use_losing_red_apples == 'y'}")

    # Start the game, prompt the user for options
    while game.get_game_state().current_game < game.get_game_state().total_games:
        # Start a new game
        game.new_game()

    # Run the winner counter and plot the results
    data_analysis_main(game.winner_csv_filepath)

if __name__ == "__main__":
    main()
