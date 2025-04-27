# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse
import time
import yaml
import os
import sys # Import sys (needed if you using streamlit)

# # Third-party Libraries
# try:
#     import streamlit.web.cli as stcli # Import streamlit cli runner
#     import streamlit as st # Import streamlit for session state
# except ImportError:
#     stcli = None
#     st = None


# Local Modules
from src.embeddings.embeddings import Embedding
from src.core.game import ApplesToApples
from src.logging.game_logger import configure_logging
from src.data_analysis.data_analysis import main as data_analysis_main
from src.data_classes.data_classes import GameLog, PathsConfig, GameConfig, ModelConfig, BetweenGameConfig

# Import the UI implementations
from src.ui.terminal.terminal_ui import TerminalUI
try:
    from src.ui.gui.pygame_ui import PygameUI
    import pygame
except ImportError:
    PygameUI = None
    pygame = None

# Always set these to None for now
PygameUI = None
pygame = None


# Standalone Config Loader
def load_config(config_path="config/config.yaml") -> dict:
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        # Use print because logging isn't configured yet.
        print(f"Warning: Config file not found at {config_path}. Using default settings.")
        return {} # Return empty dict, defaults will be handled by .get()
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Config loaded successfully, no need to print here.
        return config if config else {} # Return empty dict if file is empty
    except yaml.YAMLError as e:
        # Use print for errors during initial load.
        print(f"ERROR: Error parsing YAML config file {config_path}: {e}. Using default settings.")
        return {} # Return empty dict on error
    except Exception as e:
        # Use print for other errors during initial load.
        print(f"ERROR: Failed to load config file {config_path}: {e}. Using default settings.")
        return {}


class GameDriver:
    def __init__(self,
                 config: dict,
                 paths_config: PathsConfig,
                 training_mode: bool,
                 number_of_players: int,
                 points_to_win: int,
                 total_games: int) -> None:

        # Store the passed config and paths
        self.config = config
        self.paths_config = paths_config

        # Populate remaining config dataclasses
        raw_game_config = self.config.get("game", {})
        raw_model_config = self.config.get("model", {})
        raw_interaction_config = self.config.get("interaction", {})

        self.game_config = GameConfig(
            default_max_cards_in_hand=raw_game_config.get("default_max_cards_in_hand", 7),
            training_max_cards_in_hand=raw_game_config.get("training_max_cards_in_hand", 25),
            training_num_players=raw_game_config.get("training_num_players", 2)
        )
        self.model_config = ModelConfig(
            use_extra_vectors=raw_model_config.get("use_extra_vectors_default", True)
        )
        self.interaction_config = BetweenGameConfig(
            change_players=raw_interaction_config.get("change_players_between_games_default", False),
            cycle_starting_judges=raw_interaction_config.get("cycle_starting_judges_default", False),
            reset_models=raw_interaction_config.get("reset_models_between_games_default", False),
            reset_cards=raw_interaction_config.get("reset_cards_between_games_default", False)
        )
        # --- End Populate Dataclasses ---

        # Set the game state for training mode using config values
        if training_mode:
            effective_number_of_players = self.game_config.training_num_players
            effective_max_cards_in_hand = self.game_config.training_max_cards_in_hand
        else: # Set the game state for non-training mode
            effective_number_of_players = number_of_players # Use command line arg for non-training
            effective_max_cards_in_hand = self.game_config.default_max_cards_in_hand

        # Initialize the GameLog
        self.game_log: GameLog = GameLog()
        # Use effective values determined above
        self.game_log.intialize_input_args(effective_number_of_players, effective_max_cards_in_hand, points_to_win, total_games)

    def load_keyed_vectors(self, use_custom_loader: bool) -> None:
        print("Loading Vectors...")
        # --- Use path from dataclass ---
        logging.info(f"Loading word vectors from: {self.paths_config.embeddings}")
        start = time.perf_counter()
        try:
            # --- Use path from dataclass ---
            self.embedding = Embedding(self.paths_config.embeddings, custom=use_custom_loader)
            end = time.perf_counter()
            load_time = end - start
            print(f"Loaded Vectors in {load_time:.2f} seconds.")
            logging.info(f"Successfully loaded vectors in {load_time:.2f} seconds.")
        except FileNotFoundError:
             # --- Use path from dataclass ---
            logging.error(f"Embedding file not found at {self.paths_config.embeddings}. Please check the path in config.yaml.")
            print(f"ERROR: Embedding file not found at {self.paths_config.embeddings}. Exiting.")
            exit(1) # Exit if embeddings can't be loaded
        except Exception as e:
            logging.error(f"An error occurred while loading embeddings: {e}")
            print(f"ERROR: Failed to load embeddings. Check logs for details. Exiting.")
            exit(1)


def range_type(min_value, max_value):
    """Function for creating a range-limited integer type for argparse."""
    def range_checker(value):
        try:
            ivalue = int(value)
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError(f"Value must be between {min_value} and {max_value}")
            return ivalue
        except ValueError:
             raise argparse.ArgumentTypeError(f"Value must be an integer.")
    return range_checker


# def get_user_input_y_or_n(prompt: str) -> str:
#     """Prompts the user for a 'y' or 'n' response."""
#     while True:
#         response = input(prompt).lower().strip()
#         if response in ["y", "n"]:
#             return response
#         print("Invalid input. Type in either 'y' or 'n'.")

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
                    "The game runs in terminal mode by default. "\
                    "Use the -A flag to load all available card packs. "\
                    "Use the -V flag to use the custom vector loader (may not work on all systems). "\
                    "Use the -G flag to run the game in GUI mode (using Pygame). "\
                    "Use the -P flag to print the game info and results in the terminal (terminal mode only). "\
                    "Use the -T flag to run the program in training mode (forces terminal execution). "\
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
    parser.add_argument("-G", "--gui_mode", action="store_true", help="Use GUI mode for the game")
    parser.add_argument("-P", "--print_in_terminal", action="store_true", help="Print game info and prompts in terminal (for terminal mode)")
    parser.add_argument("-T", "--training_mode", action="store_true", help="Run in training mode (forces terminal execution)")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug mode for detailed logging")

    # Parse the command line arguments
    args = parser.parse_args()

    # Load Config and Create PathsConfig EARLY
    print("Loading configuration...")
    config = load_config()
    raw_paths_config = config.get("paths", {})
    raw_logging_config = config.get("logging", {})
    paths_config = PathsConfig(
        data_base=raw_paths_config.get("data_base", "./data"),
        embeddings=raw_paths_config.get("embeddings", "./data/embeddings/GoogleNews-vectors-negative300.bin"),
        apples_data=raw_paths_config.get("apples_data", "./data/apples"),
        model_archetypes=raw_paths_config.get("model_archetypes", "./data/agent_archetypes"),
        logging_base_directory=raw_paths_config.get("logs", "./logs"),
        logging_filename=raw_logging_config.get("log_filename", "a2a_game.log"),
        analysis_output=raw_paths_config.get("analysis_output", "./analysis_results")
    )

    configure_logging(args.debug, paths_config)
    logging.info("--- Logging configured ---")
    logging.info(f"Logging to: {os.path.join(paths_config.logging_base_directory, paths_config.logging_filename)}")

    # Create the GameDriver object
    print("Initializing 'Apples to Apples' game driver...")
    logging.info("Starting 'Apples to Apples' game driver.")
    game_driver = GameDriver(
        config=config,
        paths_config=paths_config,
        training_mode=args.training_mode,
        number_of_players=args.number_of_players,
        points_to_win=args.points_to_win,
        total_games=args.total_games
    )

    # Log the command line arguments
    logging.info(f"Command line arguments: {args}")
    logging.info(f"Effective number of players: {game_driver.game_log.total_number_of_players}")
    logging.info(f"Effective max cards in hand: {game_driver.game_log.max_cards_in_hand}")
    logging.info(f"Points to win: {game_driver.game_log.points_to_win}")
    logging.info(f"Total games to be played: {game_driver.game_log.total_games}")
    logging.info(f"Green card expansion file: {args.green_expansion}")
    logging.info(f"Red card expansion file: {args.red_expansion}")
    logging.info(f"Load all card packs: {args.load_all_packs}")
    logging.info(f"Use custom vector loader: {args.vector_loader}")
    logging.info(f"Use GUI mode: {args.gui_mode}")
    logging.info(f"Print in terminal: {args.print_in_terminal}")
    logging.info(f"Training mode: {args.training_mode}")
    logging.info(f"Debug mode: {args.debug}")
    logging.info(f"Using embedding path: {game_driver.paths_config.embeddings}")
    logging.info(f"Using model archetypes path: {game_driver.paths_config.model_archetypes}")
    logging.info(f"Logging base directory: {game_driver.paths_config.logging_base_directory}")
    logging.info(f"Log filename: {game_driver.paths_config.logging_filename}")

    # Load the keyed vectors (using path from config stored in game_driver)
    game_driver.load_keyed_vectors(args.vector_loader)

    # # Initialize the appropriate user interface based on args
    # # Training mode forces terminal interface
    # use_gui = args.gui_mode and not args.training_mode

    # if use_gui:
    #     # Check if Pygame is available
    #     if not pygame or not PygameUI:
    #         print("ERROR: Pygame is required for GUI mode (-G). Please install it (`pip install pygame`).")
    #         logging.error("Pygame or PygameUI not found, cannot start GUI mode.")
    #         exit(1)

    #     print("Starting GUI mode...")
    #     logging.info("Starting GUI mode.")
    #     game_interface = PygameUI()
    # else:
    #     # Terminal mode (default or forced by training mode)
    #     if args.training_mode and args.gui_mode:
    #         print("NOTE: GUI mode (-G) ignored in training mode (-T). Using terminal interface.")
    #         logging.warning("GUI mode (-G) ignored in training mode (-T). Using terminal interface.")
    #     elif not args.gui_mode:
    #         print("Starting terminal mode (default)...")
    #         logging.info("Starting terminal mode (default).")
    #     else:
    #         print("Starting terminal mode (forced by -T/training_mode)...")
    #         logging.info("Starting terminal mode (forced by -T/training_mode).")

    #     # Create terminal interface with print setting from args
    #     game_interface = TerminalUI(print_in_terminal=args.print_in_terminal)


    # Always use TerminalUI for now, regardless of args.gui_mode
    game_interface = TerminalUI(print_in_terminal=args.print_in_terminal)

    # Create the game object with the selected interface
    a2a_game = ApplesToApples(
        embedding=game_driver.embedding,
        interface=game_interface,
        paths_config=game_driver.paths_config,
        game_config=game_driver.game_config,
        training_mode=args.training_mode,
        load_all_packs=args.load_all_packs,
        green_expansion=args.green_expansion,
        red_expansion=args.red_expansion
    )

    # Set the game log
    a2a_game.initalize_game_log(game_driver.game_log)

    # Adjust cycle_starting_judges based on change_players
    adjusted_interaction_config = BetweenGameConfig(
        change_players=game_driver.interaction_config.change_players,
        cycle_starting_judges=game_driver.interaction_config.cycle_starting_judges if not game_driver.interaction_config.change_players else False,
        reset_models=game_driver.interaction_config.reset_models,
        reset_cards=game_driver.interaction_config.reset_cards
    )

    # Set the game options
    a2a_game.set_game_options(
        adjusted_interaction_config,
        game_driver.model_config
    )

    # Log the final game options being used (sourced from config)
    logging.info(f"Option - Change players between games: {adjusted_interaction_config.change_players}")
    # Log the effective judge cycling based on whether players change
    if adjusted_interaction_config.change_players:
        logging.info(f"Option - Cycle starting judges: N/A (players change between games)")
    else:
        logging.info(f"Option - Cycle starting judges: {adjusted_interaction_config.cycle_starting_judges}")
    logging.info(f"Option - Reset models between games: {adjusted_interaction_config.reset_models}")
    if args.training_mode: # Only log reset_cards if in training mode
        logging.info(f"Option - Reset training cards between games: {adjusted_interaction_config.reset_cards}")
    logging.info(f"Option - Use extra vectors: {game_driver.model_config.use_extra_vectors}")

    # Start the game timer
    start = time.perf_counter()

    # Start the game(s)
    # Play all games
    a2a_game.new_game()  # First game
    while game_driver.game_log.get_current_game_number() < game_driver.game_log.total_games:
        a2a_game.new_game()  # Subsequent games

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

    # Run data analysis if appropriate (only for terminal mode, not in training)
    if not args.training_mode and not args.gui_mode:
        logging.info("Starting data analysis.")
        data_analysis_main(
            paths_config=game_driver.paths_config,
            game_log=game_driver.game_log,
            change_players_between_games=adjusted_interaction_config.change_players,
            cycle_starting_judges=adjusted_interaction_config.cycle_starting_judges,
            reset_models_between_games=adjusted_interaction_config.reset_models,
            use_extra_vectors=game_driver.model_config.use_extra_vectors,
        )
        logging.info("Data analysis finished.")
    elif args.training_mode:
        logging.info("Skipping data analysis in training mode.")
    elif args.gui_mode:
        logging.info("Skipping data analysis in GUI mode.")

    # Shutdown Logging
    logging.info("Game simulation finished. Shutting down logging.")
    logging.shutdown()


if __name__ == "__main__":
    main()

