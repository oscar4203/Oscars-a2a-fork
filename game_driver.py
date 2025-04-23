# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse
import time
import yaml
import os

# Third-party Libraries

# Local Modules
from src.embeddings.embeddings import Embedding
from src.apples_to_apples import ApplesToApples
from src.logging.game_logger import configure_logging
from src.data_analysis.data_analysis import main as data_analysis_main
from src.data_classes.data_classes import GameLog
from src.gui.gui_wrapper import GUIWrapper


class GameDriver:
    def __init__(self, training_mode: bool, number_of_players: int, points_to_win: int, total_games: int, config_path="config/config.yaml") -> None:
        # Load configuration
        self.config = self._load_config(config_path)

        # Get config values with defaults
        game_config = self.config.get("game", {})
        paths_config = self.config.get("paths", {})
        interaction_config = self.config.get("interaction", {})

        # Set the game state for training mode using config values
        if training_mode:
            number_of_players = game_config.get("training_num_players", 2) # Override from config
            max_cards_in_hand = game_config.get("training_max_cards_in_hand", 25) # From config
        else: # Set the game state for non-training mode
            max_cards_in_hand = game_config.get("default_max_cards_in_hand", 7) # From config

        # Store paths
        self.embedding_path = paths_config.get("embeddings", "./data/embeddings/GoogleNews-vectors-negative300.bin")
        self.log_path = paths_config.get("logs", "./logs")
        self.log_filename = self.config.get("logging", {}).get("log_filename", "a2a_game.log")

        # Store interaction defaults
        self.change_players_default = interaction_config.get("change_players_between_games_default", False)
        self.cycle_judges_default = interaction_config.get("cycle_starting_judges_default", False)
        self.reset_models_default = interaction_config.get("reset_models_between_games_default", False)
        self.reset_cards_default = interaction_config.get("reset_cards_between_games_default", False) # For training
        self.use_extra_vectors_default = self.config.get("model", {}).get("use_extra_vectors_default", True)


        # Initialize the GameLog
        self.game_log: GameLog = GameLog()
        self.game_log.intialize_input_args(number_of_players, max_cards_in_hand, points_to_win, total_games)

    def _load_config(self, config_path="config/config.yaml"):
        if not os.path.exists(config_path):
            # Handle missing config file (e.g., raise error, use defaults)
            logging.warning(f"Config file not found at {config_path}. Using default settings.")
            return {} # Return empty dict, defaults will be handled by .get()
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {config_path}")
            return config if config else {} # Return empty dict if file is empty
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML config file {config_path}: {e}")
            return {} # Return empty dict on error
        except Exception as e:
            logging.error(f"Failed to load config file {config_path}: {e}")
            return {}

    def load_keyed_vectors(self, use_custom_loader: bool) -> None:
        print("Loading Vectors...")
        logging.info(f"Loading word vectors from: {self.embedding_path}")
        start = time.perf_counter()
        try:
            self.embedding = Embedding(self.embedding_path, custom=use_custom_loader)
            end = time.perf_counter()
            print("Loaded Vectors in", end-start, "seconds.")
            logging.info(f"Loaded Vectors in {end-start} seconds.")
        except Exception as e:
            logging.error(f"Failed to load word vectors: {e}")
            raise


def range_type(min_value, max_value):
    def range_checker(value):
        try:
            ivalue = int(value)
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError(f"Value must be between {min_value} and {max_value}")
            return ivalue
        except ValueError:
             raise argparse.ArgumentTypeError(f"Value must be an integer.")
    return range_checker


def load_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        # Handle missing config file (e.g., raise error, use defaults)
        print(f"Warning: Config file not found at {config_path}")
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
        usage="python3 game_driver.py <number_of_players> <points_to_win> <total_games> "\
              "[green_expansion] [red_expansion] [-A] [-V] [-G] [-P] [-T] [-D]"\
              "\n\nExample: python3 game_driver.py 4 5 10 green_apples_extension.csv red_apples_extension.csv -A -V -T -D"\
              "\nFor help: python3 game_driver.py -h",
        description="Configure and run the 'Apples to Apples' game. Specify the number of players, "\
                    "points to win, and total games to play. Include optional green and red apple expansions. "\
                    "Use the -A flag to load all available card packs. "\
                    "Use the -V flag to use the custom vector loader (may not work on all systems). "\
                    "Use the -G flag to use the GUI wrapper for the game. "\
                    "Use the -P flag to print the game info and results in the terminal. "\
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
    parser.add_argument("-G", "--gui_wrapper", action="store_true", help="Use the GUI wrapper for the game")
    parser.add_argument("-P", "--print_in_terminal", action="store_true", help="Print the game info and prompts in the terminal")
    parser.add_argument("-T", "--training_mode", action="store_true", help="Train a user specified model archetype")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug mode for detailed logging")

    # Parse the command line arguments
    args = parser.parse_args()

    # Create the game driver object (this will load the config)
    print("Initializing 'Apples to Apples' game driver...")
    game_driver = GameDriver(args.training_mode, args.number_of_players, args.points_to_win, args.total_games)

    # Configure and initialize the logging module
    configure_logging(args.debug)
    logging.info("Starting 'Apples to Apples' game driver.")

    # Log the command line arguments
    logging.info(f"Command line arguments: {args}")
    # Log effective game settings after considering config and training mode
    logging.info(f"Effective number of players: {game_driver.game_log.total_number_of_players}")
    logging.info(f"Effective max cards in hand: {game_driver.game_log.max_cards_in_hand}")
    logging.info(f"Points to win: {game_driver.game_log.points_to_win}")
    logging.info(f"Total games to be played: {game_driver.game_log.total_games}")
    logging.info(f"Green card expansion file: {args.green_expansion}")
    logging.info(f"Red card expansion file: {args.red_expansion}")
    logging.info(f"Load all card packs: {args.load_all_packs}")
    logging.info(f"Use custom vector loader: {args.vector_loader}")
    logging.info(f"Use GUI wrapper: {args.gui_wrapper}")
    logging.info(f"Print in terminal: {args.print_in_terminal}")
    logging.info(f"Training mode: {args.training_mode}")
    logging.info(f"Debug mode: {args.debug}")
    logging.info(f"Using embedding path: {game_driver.embedding_path}")


    # Load the keyed vectors (using path from config stored in game_driver)
    game_driver.load_keyed_vectors(args.vector_loader)

    # Create the game object
    # Pass embedding object from game_driver
    a2a_game = ApplesToApples(game_driver.embedding, args.print_in_terminal, args.training_mode, args.load_all_packs, args.green_expansion, args.red_expansion)

    # Set the static game log
    a2a_game.initalize_game_log(game_driver.game_log)

    # The interaction variables are directly taken from the config defaults loaded in GameDriver

    # Set the game options directly from the config defaults stored in game_driver
    change_players = game_driver.change_players_default
    cycle_judges = game_driver.cycle_judges_default
    reset_models = game_driver.reset_models_default
    use_extra_vectors = game_driver.use_extra_vectors_default
    reset_cards = game_driver.reset_cards_default # This is relevant only in training mode

    # Apply logic that depended on prompts directly to config values
    # If players are not changed (based on config), then cycling judges (based on config) is relevant.
    # If players *are* changed (based on config), cycling judges doesn't apply in the same way.
    # The original logic was: if change_players_between_games == "n": prompt for cycle_starting_judges
    # We'll keep the cycle_judges setting from config regardless, but log its effective state.
    effective_cycle_judges = cycle_judges if not change_players else False # If players change, judge cycling is implicitly handled or irrelevant

    a2a_game.set_game_options(
        change_players,
        effective_cycle_judges, # Use the potentially adjusted value
        reset_models,
        use_extra_vectors,
        reset_cards # Pass the training-mode specific setting
    )

    # Log the final game options being used (sourced from config)
    logging.info(f"Option - Change players between games: {change_players}")
    # Log the effective judge cycling based on whether players change
    if change_players:
        logging.info(f"Option - Cycle starting judges: N/A (players change between games)")
    else:
        logging.info(f"Option - Cycle starting judges: {cycle_judges}") # cycle_starting_judges
    logging.info(f"Option - Reset models between games: {reset_models}")
    logging.info(f"Option - Use extra vectors: {use_extra_vectors}")
    if args.training_mode: # Only log reset_cards if in training mode
        logging.info(f"Option - Reset training cards between games: {reset_cards}")


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
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    # Print and log the total time elapsed
    print(f"Total time elapsed: {hours} hour(s), {minutes} minute(s), {seconds} second(s)")
    logging.info(f"Total time elapsed: {hours} hour(s), {minutes} minute(s), {seconds} second(s)")

    # Run the winner counter and plot the results, if not in training mode
    if not args.training_mode:
        # Consider passing analysis output path from config if needed by data_analysis_main
        analysis_output_path = game_driver.config.get('paths', {}).get('analysis_output', './analysis_results')
        data_analysis_main(
            game_driver.game_log,
            change_players, # change_players_between_games
            effective_cycle_judges, # Use the potentially adjusted value
            reset_models, # reset_models_between_games
            use_extra_vectors,
            # output_dir=analysis_output_path # Example: Pass path if needed
        )

if __name__ == "__main__":
    main()

