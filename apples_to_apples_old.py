# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse
from datetime import datetime

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules
from source.game_logger import configure_logging
from source.apples import GreenApple, RedApple, Deck
from source.agent import Agent, HumanAgent, RandomAgent, AIAgent
from source.model import Model, model_type_mapping
from source.game_logger import GameResults, PreferenceUpdates, format_naming_scheme, log_vectors, log_gameplay, log_winner, LOGGING_BASE_DIRECTORY
from source.w2vloader import VectorsW2V
from source.data_analysis import main as data_analysis_main


class ApplesToApples:
    def __init__(self, number_of_players: int, points_to_win: int, total_games: int, green_expansion: str = '', red_expansion: str = '') -> None:
        self.number_of_players: int = number_of_players
        self.players: list[Agent] = []
        self.points_to_win: int = points_to_win
        self.total_games: int = total_games
        self.green_expansion_filename: str = green_expansion
        self.red_expansion_filename: str = red_expansion
        self.green_apples_deck: Deck = Deck()
        self.red_apples_deck: Deck = Deck()
        self.max_cards_in_hand: int = 7
        # Initialize default game state values
        self.current_game: int = 0
        self.current_round: int = 0
        self.current_judge: Agent | None = None
        self.round_winner: Agent | None = None
        self.game_winner: Agent | None = None

    def load_keyed_vectors(self) -> None:
        self.keyed_vectors: KeyedVectors = KeyedVectors.load_word2vec_format("./apples/GoogleNews-vectors-negative300.bin", binary=True)
        # self.vectors = VectorsW2V("./apples/GoogleNews-vectors-negative300.bin")
        # embeddings.load()

    def set_between_game_options(self, change_players: bool, cycle_starting_judges: bool, reset_models: bool, train_on_extra_vectors: bool, train_on_losing_red_apples: bool) -> None:
        self.change_players_between_games = change_players
        self.cycle_starting_judges_between_games = cycle_starting_judges
        self.reset_models_between_games = reset_models
        self.train_on_extra_vectors = train_on_extra_vectors
        self.train_on_losing_red_apples = train_on_losing_red_apples

    def new_game(self) -> None:
        """
        Start a new game of 'Apples to Apples' and reset the game state.
        Optionally, initialize new players.
        """
        print("Starting new 'Apples to Apples' game.")
        logging.info("Starting new 'Apples to Apples' game.")

        # Increment the current game counter
        self.current_game += 1

        # Print and log the game message
        print(f"\n------------- GAME {self.current_game} of {self.total_games} -------------")
        logging.info(f"\n------------- GAME {self.current_game} of {self.total_games} -------------")

        # Reset the game state
        print("Resetting game state.")
        logging.info("Resetting game state.")
        self.__reset_game_state()

        # Initialize the decks
        print("Initializing decks.")
        logging.info("Initializing decks.")
        self.__initialize_decks()

        # Initialize the players
        if self.change_players_between_games or self.current_game == 1:
            print("Initializing players.")
            logging.info("Initializing players.")
            self.players = []
            self.__initialize_players()
        else:
            if self.reset_models_between_games:
                self.__reset_opponent_models()

        # Define the naming scheme and the winner csv filepath
        naming_scheme = format_naming_scheme(self.players, self.total_games, self.points_to_win)
        self.winner_csv_filepath = f"{LOGGING_BASE_DIRECTORY}{naming_scheme}/winners-{naming_scheme}.csv"

        # Choose the starting judge
        self.__choose_starting_judge()

        # Start the game loop
        self.__game_loop()

    def __reset_game_state(self) -> None:
        # Reset the game state for a new game
        self.current_round = 0
        self.current_judge = None
        self.game_winner = None

        # Reset the player points and judge status
        for player in self.players:
            player.reset_points()
            player.set_judge_status(False)

    def __new_round(self) -> None:
        # Reset the game state for a new round
        self.round_winner = None

        # Increment the round counter
        self.current_round += 1

        # Print and log the round message
        round_message = f"\n===================" \
                        f"\nROUND {self.current_round}:" \
                        f"\n===================\n"
        print(round_message)
        logging.info(round_message)

        # Print and log the player points
        for player in self.players:
            print(f"{player.get_name()}: {player.get_points()} points")
            logging.info(f"{player.get_name()}: {player.get_points()} points")

        # Assign the next judge, except for the first round
        if self.current_round > 1:
            self.__assign_next_judge()

    def __initialize_decks(self) -> None:
        # Initialize the decks
        self.green_apple_in_play: dict[Agent, GreenApple] | None = None
        self.red_apples_in_play: list[dict[Agent, RedApple]] = []
        self.discarded_green_apples: list[GreenApple] = []
        self.discarded_red_apples: list[RedApple] = []

        # Shuffle the decks
        self.__load_and_shuffle_deck(self.green_apples_deck, "Green Apples", "./apples/green_apples.csv", self.green_expansion_filename)
        self.__load_and_shuffle_deck(self.red_apples_deck, "Red Apples", "./apples/red_apples.csv", self.red_expansion_filename)

    def __load_and_shuffle_deck(self, deck: Deck, deck_name: str, base_file: str, expansion_file: str) -> None:
        # Load the base deck
        deck.load_deck(deck_name, base_file)
        print(f"Loaded {len(deck.get_apples())} {deck_name.lower()}.")
        logging.info(f"Loaded {len(deck.get_apples())} {deck_name.lower()}.")

        # Load the expansion deck, if applicable
        if expansion_file:
            deck.load_deck(f"{deck_name} Expansion", expansion_file)
            print(f"Loaded {len(deck.get_apples())} {deck_name.lower()} from the expansion.")
            logging.info(f"Loaded {len(deck.get_apples())} {deck_name.lower()} from the expansion.")

        # Shuffle the deck
        deck.shuffle()

    def __discard_apples_in_play(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Check if the green apple in play is None
        if self.green_apple_in_play is None:
            logging.error("The green apple in play is None.")
            raise ValueError("The green apple in play is None.")

        # Check if the red apples in play is None
        if self.red_apples_in_play is None:
            logging.error("The red apples in play is None.")
            raise ValueError("The red apples in play is None.")

        # Create red_apples_list using list comprehension
        red_apples_list = [list(red_apple.values())[0] for red_apple in self.red_apples_in_play]

        # Discard the green cards
        self.discarded_green_apples.append(self.green_apple_in_play[self.current_judge])
        self.green_apple_in_play = None

        # Discard the red cards
        self.discarded_red_apples.extend(red_apples_list)
        self.red_apples_in_play = []

    def __generate_unique_agent_name(self, base_name: str) -> str:
        # Unpack the existing names
        existing_names = [agent.get_name() for agent in self.players]

        # Generate a unique name
        i = 1
        while f"{base_name} {i}" in existing_names:
            i += 1
        return f"{base_name} {i}"

    def __initialize_players(self) -> None:
        # Display the number of players
        print(f"There are {self.number_of_players} players.")
        logging.info(f"There are {self.number_of_players} players.")

        # Create the players
        for i in range(self.number_of_players):
            # Prompt the user to select the player type
            print(f"\nWhat type is Agent {i + 1}?")
            logging.info(f"What type is Agent {i + 1}?")
            player_type: str = input("Please enter the player type (1: Human, 2: Random, 3: AI): ")
            logging.info(f"Please enter the player type (1: Human, 2: Random, 3: AI): {player_type}")

            # Validate the user input
            while player_type not in ['1', '2', '3']:
                player_type = input("Invalid input. Please enter the player type (1: Human, 2: Random, 3: AI): ")
                logging.error(f"Invalid input. Please enter the player type (1: Human, 2: Random, 3: AI): {player_type}")

            # Determine the player name
            if player_type == '1':
                # Validate the user input for a unique name
                new_agent_name: str = ""
                while True:
                    new_agent_name = input(f"Please enter the name for the Human Agent: ")
                    if new_agent_name not in [agent.get_name() for agent in self.players]:
                        break
                new_agent = HumanAgent(f"Human Agent - {new_agent_name}")
            elif player_type == '2':
                new_agent_name = self.__generate_unique_agent_name("Random Agent")
                new_agent = RandomAgent(new_agent_name)
            elif player_type == '3':
                # Validate the user input for the machine learning model
                ml_model_type: str = input("Please enter the machine learning model (1: Linear Regression, 2: Neural Network): ")
                logging.info(f"Please enter the machine learning model (1: Linear Regression, 2: Neural Network): {ml_model_type}")
                while ml_model_type not in ['1', '2']:
                    ml_model_type = input("Invalid input. Please enter the machine learning model (1: Linear Regression, 2: Neural Network): ")
                    logging.error(f"Invalid input. Please enter the machine learning model (1: Linear Regression, 2: Neural Network): {ml_model_type}")

                # Validate the user input for the pretrained archetype
                pretrained_archetype: str = input("Please enter the pretrained archetype (1: Literalist, 2: Contrarian, 3: Comedian): ")
                logging.info(f"Please enter the pretrained archetype (1: Literalist, 2: Contrarian, 3: Comedian): {pretrained_archetype}")
                while pretrained_archetype not in ['1', '2', '3']:
                    pretrained_archetype = input("Invalid input. Please enter the pretrained archetype (1: Literalist, 2: Contrarian, 3: Comedian): ")
                    logging.error(f"Invalid input. Please enter the pretrained archetype (1: Literalist, 2: Contrarian, 3: Comedian): {pretrained_archetype}")

                # Generate a unique name for the AI agent
                ml_model_type_class = model_type_mapping[ml_model_type]
                logging.debug(f"Model Type Class: {ml_model_type_class}")
                logging.debug(f"Model Type Name: {ml_model_type_class.__name__}")

                # Create pretrained model
                pretrained_archetype_string: str = ""
                if pretrained_archetype == '1':
                    pretrained_archetype_string = "Literalist"
                elif pretrained_archetype == '2':
                    pretrained_archetype_string = "Contrarian"
                elif pretrained_archetype == '3':
                    pretrained_archetype_string = "Comedian"
                logging.debug(f"Pretrained Model String: {pretrained_archetype_string}")

                # Create the AI agent
                new_agent_name = self.__generate_unique_agent_name(f"AI Agent - {ml_model_type_class.__name__} - {pretrained_archetype_string}")
                new_agent = AIAgent(new_agent_name, ml_model_type_class, pretrained_archetype_string, False)

            # Append the player object
            self.players.append(new_agent)
            logging.info(self.players[i])

            # Have each player pick up 7 red cards
            self.players[i].draw_red_apples(self.keyed_vectors, self.red_apples_deck, self.max_cards_in_hand, self.train_on_extra_vectors)

        # Initialize the models for the AI agents
        for player in self.players:
            if isinstance(player, AIAgent):
                player.initialize_models(self.keyed_vectors, self.players)
                logging.info(f"Initialized models for {new_agent.get_name()}.")

    def __choose_starting_judge(self) -> None:
        # Clear the judge status for all players
        for player in self.players:
            player.set_judge_status(False)

        # If cycle starting judge is True, choose the starting judge automatically
        if self.cycle_starting_judges_between_games:
            # Cycle through the judges to get the judge index
            judge_index = ((self.current_game - 1) % len(self.players)) # Subtract 1 since 0-based index and current_game starts at 1
        else: # If cycle starting judge is False, prompt the user to choose the starting judge
            if self.change_players_between_games:
                # Choose the starting judge
                choice = input(f"\nPlease choose the starting judge (1-{self.number_of_players}): ")
                logging.info(f"Please choose the starting judge (1-{self.number_of_players}): {choice}")

                # Validate the user input
                while not choice.isdigit() or int(choice) < 1 or int(choice) > self.number_of_players:
                    choice = input(f"Invalid input. Please enter a number (1-{self.number_of_players}): ")
                    logging.error(f"Invalid input. Please enter a number (1-{self.number_of_players}): {choice}")

                # Set the current judge index
                judge_index = int(choice)
            else:
                judge_index = 0

        # Assign the starting judge and set the judge status
        self.current_judge = self.players[judge_index]
        self.current_judge.set_judge_status(True)
        print(f"{self.current_judge.get_name()} is the starting judge.")
        logging.info(f"{self.current_judge.get_name()} is the starting judge.")

    def __assign_next_judge(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Clear the judge status for the current judge
        self.current_judge.set_judge_status(False)

        # Assign the next judge and set the judge status
        self.current_judge = self.players[(self.players.index(self.current_judge) + 1) % self.number_of_players]
        self.current_judge.set_judge_status(True)
        print(f"{self.current_judge.get_name()} is the next judge.")
        logging.info(f"{self.current_judge.get_name()} is the next judge.")

    def __judge_prompt(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Prompt the judge to draw a green card
        print(f"\n{self.current_judge.get_name()}, please draw a green card.")
        logging.info(f"{self.current_judge.get_name()}, please draw a green card.")

        # Set the green card in play
        self.green_apple_in_play = {self.current_judge: self.current_judge.draw_green_apple(self.keyed_vectors, self.green_apples_deck, self.train_on_extra_vectors)}

    def __player_prompt(self) -> list[RedApple]:
        red_apples: list[RedApple] = []
        # Prompt the players to select a red card
        for player in self.players:
            if player.get_judge_status():
                continue

            print(f"\n{player.get_name()}, please select a red card.")
            logging.info(f"{player.get_name()}, please select a red card.")

            # Check if the current judge is None
            if self.current_judge is None:
                logging.error("The current judge is None.")
                raise ValueError("The current judge is None.")

            # Check if the green apple in play is None
            if self.green_apple_in_play is None:
                logging.error("The green apple in play is None.")
                raise ValueError("The green apple in play is None.")

            # Set the red cards in play
            red_apple = player.choose_red_apple(self.current_judge, self.green_apple_in_play[self.current_judge])
            self.red_apples_in_play.append({player: red_apple})
            red_apples.append(red_apple)
            logging.info(f"Red card: {red_apple}")

            # Prompt the player to pick up a new red card
            if len(player.get_red_apples()) < self.max_cards_in_hand:
                player.draw_red_apples(self.keyed_vectors, self.red_apples_deck, self.max_cards_in_hand, self.train_on_extra_vectors)

        return red_apples

    def __train_ai_agents(self, game_results: GameResults) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Check if the green apple in play is None
        if self.green_apple_in_play is None:
            logging.error("The green apple in play is None.")
            raise ValueError("The green apple in play is None.")

        # Take a snapshot of the current date and time
        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M)")

        # Train all AI agents (if applicable)
        for player in self.players:
            if isinstance(player, AIAgent) and player != self.current_judge:

                player.train_opponent_models(
                    self.keyed_vectors, self.current_judge,
                    self.green_apple_in_play[self.current_judge],
                    game_results.winning_red_apple,
                    game_results.losing_red_apples,
                    self.train_on_extra_vectors,
                    self.train_on_losing_red_apples
                )

                judge_model: Model | None = player.get_opponent_model(self.current_judge)

                # Check that the judge model is not None
                if judge_model is None:
                    logging.error("The judge model is None.")
                    raise ValueError("The judge model is None.")

                current_slope = judge_model.get_slope_vector()
                current_bias = judge_model.get_bias_vector()
                preference_updates = PreferenceUpdates(player, self.current_round, date_time,
                                                game_results.winning_red_apple, self.green_apple_in_play[self.current_judge],
                                                current_bias, current_slope)
                log_vectors(game_results, preference_updates)

    def __reset_opponent_models(self) -> None:
        for player in self.players:
            if isinstance(player, AIAgent):
                player.reset_opponent_models()

    def __is_game_over(self) -> Agent | None:
        for player in self.players:
            if player.get_points() >= self.points_to_win:
                return player
        return None

    def __game_loop(self) -> None:
        # Start the game loop
        while self.game_winner is None:
            # Increment the round counter and print the round messages
            self.__new_round()

            # Prompt the judge to select a green card
            self.__judge_prompt()

            # Prompt the players to select a red card
            red_apples_this_round  = self.__player_prompt()

            # Check if the current judge is None
            if self.current_judge is None:
                logging.error("The current judge is None.")
                raise ValueError("The current judge is None.")

            # Check if the green apple in play is None
            if self.green_apple_in_play is None:
                logging.error("The green apple in play is None.")
                raise ValueError("The green apple in play is None.")

            # Prompt the judge to select the winning red card
            print(f"\n{self.current_judge.get_name()}, please select the winning red card.")
            logging.info(f"{self.current_judge.get_name()}, please select the winning red card.")
            winning_red_apple_dict: dict[str, RedApple] = self.current_judge.choose_winning_red_apple(
                self.green_apple_in_play[self.current_judge], self.red_apples_in_play)

            # Extract the winning red apple and losing red apples
            winning_red_apple: RedApple = list(winning_red_apple_dict.values())[0]
            losing_red_apples: list[RedApple] = red_apples_this_round.copy()
            losing_red_apples.remove(winning_red_apple)
            logging.info(f"Losing Apples: {losing_red_apples}")

            # Print and log the winning red card
            print(f"{self.current_judge.get_name()} chose the winning red card '{winning_red_apple}'.")
            logging.info(f"{self.current_judge.get_name()} chose the winning red card '{winning_red_apple}'.")

            # Award points to the winning player
            for player in self.players:
                logging.debug(f"Agent.name: {player.get_name()}, datatype: {type(player.get_name())}")
                logging.debug(f"Winning Red Card: {winning_red_apple_dict}, datatype: {type(winning_red_apple_dict)}")
                logging.debug(f"Winnning Red Card Keys: {winning_red_apple_dict.keys()}, datatype: {type(winning_red_apple_dict.keys())}")
                if player.get_name() in winning_red_apple_dict.keys():
                    self.round_winner = player
                    self.round_winner.set_points(self.round_winner.get_points() + 1)
                    print(f"{self.round_winner.get_name()} has won the round!")
                    logging.info(f"{self.round_winner} has won the round!")

             # Check if the game is over
            self.game_winner = self.__is_game_over()

            # Consolidate the gameplay results for the round and game
            results = GameResults(self.players, self.points_to_win, self.total_games, self.current_game, self.current_round,
                                  self.green_apple_in_play[self.current_judge], self.red_apples_in_play, winning_red_apple, losing_red_apples,
                                  self.current_judge, self.round_winner, self.game_winner)
            log_gameplay(results, True)

            # Train the AI agents
            self.__train_ai_agents(results)

            # Check if the game is over and print the winner message
            if self.game_winner is not None:
                # Prepare the winner message
                winner_text = f"# {self.game_winner.get_name()} has won the game! #"
                border = '#' * len(winner_text)
                message = f"\n{border}\n{winner_text}\n{border}\n"

                # Print and log the winner message
                print(message)
                logging.info(message)
                log_winner(results, True)

            # Reset the round state
            self.__discard_apples_in_play()


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
        usage="python apples_to_apples.py <# of players> <# of points to win> <# of games> "\
              "[green_expansion] [red_expansion] [-D]",
        description="Configure and run the 'Apples to Apples' game. Specify the number of players, "\
                    "points to win, and total games to play. Include optional green and red card expansions. "\
                    "Use the -D flag to enable debug mode for detailed logging."
    )

    # Add the command line arguments
    parser.add_argument("players", type=range_type(3, 8), help="Total number of players (3-8).")
    parser.add_argument("points", type=range_type(1, 10), help="Total number of points to win (1-10).")
    parser.add_argument("total_games", type=int, choices=range(1,1000), help="Total number of games to play (1-1000).")
    parser.add_argument("green_expansion", type=str, nargs='?', default='', help="Filename to a green card expansion (optional).")
    parser.add_argument("red_expansion", type=str, nargs='?', default='', help="Filename to a red card expansion (optional).")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug mode")

    # Parse the command line arguments
    args = parser.parse_args()

    # Configure and initialize the logging module
    configure_logging(args.debug)
    logging.info("Starting 'Apples to Apples' application.")

    # Log the command line arguments
    logging.info(f"Command line arguments: {args}")
    logging.info(f"Number of players: {args.players}")
    logging.info(f"Points to win: {args.points}")
    logging.info(f"Number of games to be played: {args.total_games}")
    logging.info(f"Green card expansion file: {args.green_expansion}")
    logging.info(f"Red card expansion file: {args.red_expansion}")

    # Create the game object
    game = ApplesToApples(args.players, args.points, args.total_games, args.green_expansion, args.red_expansion)

    # Load the keyed vectors
    game.load_keyed_vectors()

    # Initialize all between game option variables
    change_players_between_games = "n"
    cycle_starting_judges = "n"
    reset_models_between_games = "n"
    train_on_extra_vectors = "n"
    train_on_losing_red_apples = "n"

    # Prompt the user on whether they want to change players between games
    change_players_between_games = get_user_input_y_or_n("Do you want to change players between games? (y/n): ")

    # Prompt the user on whether they want to cycle the starting judge between games
    if change_players_between_games == "n":
        cycle_starting_judges = get_user_input_y_or_n("Do you want to cycle the starting judge between games? (y/n): ")

    # Prompt the user on whether they want to reset the opponent model vectors between games
    reset_models_between_games = get_user_input_y_or_n("Do you want to reset the opponent models between games? (y/n): ")

    # Prompt the user on whether they want to include the synonym and description vectors as part of the model
    train_on_extra_vectors = get_user_input_y_or_n("Do you want to include the synonym and description vectors as part of the model training? (y/n): ")

    # Prompt the user on whether they want to include the losing red apples as part of the model training
    train_on_losing_red_apples = get_user_input_y_or_n("Do you want to include the losing red apples as part of the model training? (y/n): ")

    # Set the between game options
    game.set_between_game_options(
        change_players_between_games == 'y',
        cycle_starting_judges == 'y',
        reset_models_between_games == 'y',
        train_on_extra_vectors == 'y',
        train_on_losing_red_apples == 'y'
    )

    # Start the game, prompt the user for options
    while game.current_game < game.total_games:
        if (not game.change_players_between_games) or (game.current_game == 0):
            game.new_game()
        else:
            print("--------------------OPTIONS--------------------")
            print("1.Restart the game (same players)\n2.Start a new game (new players)\n3.End Session.\n")
            response = input("Which option do you choose?: ")
            if response == '1':
                game.new_game()
            elif response == '2':
                game.new_game()
            elif response == '3':
                game.current_game = game.total_games
            else:
                print("Invalid response. Please select '1', '2', or '3'.")

    # Run the winner counter and plot the results
    data_analysis_main(game.winner_csv_filepath)


if __name__ == "__main__":
    main()
