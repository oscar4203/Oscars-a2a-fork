# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse

# Third-party Libraries
from gensim.models import KeyedVectors
from datetime import datetime

# Local Modules
from source.logging import configure_logging
from source.apples import GreenApple, RedApple, Deck
from source.agent import Agent, HumanAgent, RandomAgent, AIAgent, model_type_mapping
from source.model import Model
from source.logging import GameResults, log_gameplay, log_winner, PreferenceUpdates, log_vectors
from source.w2vloader import VectorsW2V


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
        self.__cards_in_hand: int = 7
        # Initialize default game state values
        self.winner: Agent | None = None
        self.round: int = 0
        self.current_judge: Agent | None = None

    def load_vectors(self) -> None:
        self.nlp_model: KeyedVectors = KeyedVectors.load_word2vec_format("./apples/GoogleNews-vectors-negative300.bin", binary=True)
        # self.vectors = VectorsW2V("./apples/GoogleNews-vectors-negative300.bin")
        # embeddings.load()

    def reset_vectors(self) -> None:
        for player in self.players:
            if isinstance(player, AIAgent):
                vector_size = player.__model_type._vector_size
                player.__model_type.__load_vectors(vector_size)

    def new_game(self, new_players: bool) -> None:
        """
        Start a new game of 'Apples to Apples' and reset the game state.
        Optionally, initialize new players.
        """
        # Reset the game state
        print("Resetting game state.")
        logging.info("Resetting game state.")
        self.__reset_game_state()

        # Initialize the decks
        print("Initializing decks.")
        logging.info("Initializing decks.")
        self.__initialize_decks()

        # Initialize the players
        if new_players:
            print("Initializing players.")
            logging.info("Initializing players.")
            self.players = []
            self.__initialize_players()

    def start_game(self, cycle_judges: bool, games_played: int) -> None:
        print("Starting new 'Apples to Apples' game.")
        logging.info("Starting new 'Apples to Apples' game.")

        if cycle_judges:
            for player in self.players:
                player.set_judge_status(False)

            #Automatically cycles through judges; used for automatic playing of multiple games
            judge = (games_played % len(self.players)) + 1
            self.current_judge = self.players[judge - 1]
            self.players[judge - 1].set_judge_status(True)
            print(f"{self.players[judge - 1].get_name()} is the starting judge.")
        else:
            # Choose the starting judge
            self.__choose_judge()

        # Start the game loop
        self.__game_loop()

    def __reset_game_state(self) -> None:
        # Reset the game state
        self.winner = None
        self.round = 0
        self.current_judge = None

        # Reset the player points
        for player in self.players:
            player.reset_points()

    def __initialize_decks(self) -> None:
        # Initialize the decks
        self.green_apples_in_play: dict[Agent, GreenApple] | None = None
        self.red_apples_in_play: list[dict[str, RedApple]] = []
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

    def __generate_unique_name(self, base_name: str) -> str:
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
            player_type = input("Please enter the player type (1: Human, 2: Random, 3: AI): ")
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
                new_agent_name = self.__generate_unique_name("Random Agent")
                new_agent = RandomAgent(new_agent_name)
            elif player_type == '3':
                # Validate the user input for the model type
                model_type: str = ""
                model_type = input("Please enter the model type (1: Linear Regression, 2: Neural Network): ")
                logging.info(f"Please enter the model type (1: Linear Regression, 2: Neural Network): {model_type}")
                while model_type not in ['1', '2']:
                    model_type = input("Invalid input. Please enter the model type (1: Linear Regression, 2: Neural Network): ")
                    logging.error(f"Invalid input. Please enter the model type (1: Linear Regression, 2: Neural Network): {model_type}")

                # Validate the user input for the pretrained model type
                pretrained_model_type: str = ""
                pretrained_model_type = input("Please enter the pretrained model type (1: Literalist, 2: Contrarian, 3: Comedian): ")
                logging.info(f"Please enter the pretrained model type (1: Literalist, 2: Contrarian, 3: Comedian): {pretrained_model_type}")
                while pretrained_model_type not in ['1', '2', '3']:
                    pretrained_model_type = input("Invalid input. Please enter the pretrained model type (1: Literalist, 2: Contrarian, 3: Comedian): ")
                    logging.error(f"Invalid input. Please enter the pretrained model type (1: Literalist, 2: Contrarian, 3: Comedian): {pretrained_model_type}")

                # Generate a unique name for the AI agent
                model_type_class = model_type_mapping[model_type]
                logging.debug(f"Model Type Class: {model_type_class}")
                logging.debug(f"Model Type Name: {model_type_class.__name__}")

                # Create pretrained model
                pretrained_model_string: str = ""
                if pretrained_model_type == '1':
                    pretrained_model_string = "Literalist"
                elif pretrained_model_type == '2':
                    pretrained_model_string = "Contrarian"
                elif pretrained_model_type == '3':
                    pretrained_model_string = "Comedian"
                logging.debug(f"Pretrained Model String: {pretrained_model_string}")

                # Create the AI agent
                new_agent_name = self.__generate_unique_name(f"AI Agent - {model_type_class.__name__} - {pretrained_model_string}")
                new_agent = AIAgent(new_agent_name, model_type_class, pretrained_model_string, False)

            # Append the player object
            self.players.append(new_agent)
            logging.info(self.players[i])

            # Have each player pick up 7 red cards
            self.players[i].draw_red_apples(self.red_apples_deck, self.__cards_in_hand)

        # Initialize the models for the AI agents
        for player in self.players:
            if isinstance(player, AIAgent):
                player.initialize_models(self.nlp_model, self.players)
                logging.info(f"Initialized models for {new_agent.get_name()}.")


    def __choose_judge(self) -> None:
        # Choose the starting judge
        choice = input(f"\nPlease choose the starting judge (1-{self.number_of_players}): ")
        logging.info(f"Please choose the starting judge (1-{self.number_of_players}): {choice}")

        # Validate the user input
        while not choice.isdigit() or int(choice) < 1 or int(choice) > self.number_of_players:
            choice = input(f"Invalid input. Please enter a number (1-{self.number_of_players}): ")
            logging.error(f"Invalid input. Please enter a number (1-{self.number_of_players}): {choice}")

        # Clear the judge status for all players
        for player in self.players:
            player.set_judge_status(False)

        # Set the current judge
        self.current_judge = self.players[int(choice) - 1]
        self.current_judge.set_judge_status(True)
        print(f"{self.players[int(choice) - 1].get_name()} is the starting judge.")

    def __assign_next_judge(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Calculate the next judge
        next_judge: Agent = self.players[(self.players.index(self.current_judge) + 1) % self.number_of_players]
        print(f"\n{next_judge.get_name()} is the next judge.")
        logging.info(f"{next_judge.get_name()} is the next judge.")

        # Assign the next judge
        self.current_judge.set_judge_status(False)
        next_judge.set_judge_status(True)
        self.current_judge = next_judge

    def __is_game_over(self) -> Agent | None:
        for player in self.players:
            if player.get_points() >= self.points_to_win:
                return player
        return None

    def __judge_prompt(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Prompt the judge to draw a green card
        print(f"\n{self.current_judge.get_name()}, please draw a green card.")
        logging.info(f"{self.current_judge.get_name()}, please draw a green card.")

        # Set the green card in play
        self.green_apples_in_play = {self.current_judge: self.current_judge.draw_green_apple(self.green_apples_deck)}

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

            # Check if the green apples in play is None
            if self.green_apples_in_play is None:
                logging.error("The green apples in play is None.")
                raise ValueError("The green apples in play is None.")

            # Set the red cards in play
            red_apple = player.choose_red_apple(self.current_judge, self.green_apples_in_play[self.current_judge])
            self.red_apples_in_play.append({player.get_name(): red_apple})
            red_apples.append(red_apple)
            logging.info(f"Red card: {red_apple}")

            # Prompt the player to pick up a new red card
            if len(player.get_red_apples()) < self.__cards_in_hand:
                player.draw_red_apples(self.red_apples_deck, self.__cards_in_hand)


        return red_apples

    def __game_loop(self) -> None:
        # Start the game loop
        start_time = datetime.now().strftime("%Y-%m-%d-%H-%M)")

        while self.winner is None:
            # Increment the round
            self.round += 1

            # Print and log the round message
            round_message = f"\n===================" \
                            f"\nROUND {self.round}:" \
                            f"\n===================\n"
            print(round_message)
            logging.info(round_message)

            # Print and log the player points
            for player in self.players:
                print(f"{player.get_name()}: {player.get_points()} points")
                logging.info(f"{player.get_name()}: {player.get_points()} points")

            # Prompt the judge to select a green card
            self.__judge_prompt()

            # Prompt the players to select a red card
            red_apples_this_round  = self.__player_prompt()

            # Check if the current judge is None
            if self.current_judge is None:
                logging.error("The current judge is None.")
                raise ValueError("The current judge is None.")

            # Check if the green apples in play is None
            if self.green_apples_in_play is None:
                logging.error("The green apples in play is None.")
                raise ValueError("The green apples in play is None.")

            # Prompt the judge to select the winning red card
            print(f"\n{self.current_judge.get_name()}, please select the winning red card.")
            logging.info(f"{self.current_judge.get_name()}, please select the winning red card.")
            winning_red_card_dict: dict[str, RedApple] = self.current_judge.choose_winning_red_apple(
                self.green_apples_in_play[self.current_judge], self.red_apples_in_play)

            # Extract the winning red card
            winning_red_card: RedApple = list(winning_red_card_dict.values())[0]

            losing_red_apples: list[RedApple] = red_apples_this_round.copy()
            losing_red_apples.remove(winning_red_card)
            logging.info(f"Losing Apples: {losing_red_apples}")

            # Print and log the winning red card
            print(f"{self.current_judge.get_name()} chose the winning red card '{winning_red_card}'.")
            logging.info(f"{self.current_judge.get_name()} chose the winning red card '{winning_red_card}'.")

            # Award points to the winning player
            for player in self.players:
                logging.debug(f"Agent.name: {player.get_name()}, datatype: {type(player.get_name())}")
                logging.debug(f"Winning Red Card: {winning_red_card_dict}, datatype: {type(winning_red_card_dict)}")
                logging.debug(f"Winnning Red Card Keys: {winning_red_card_dict.keys()}, datatype: {type(winning_red_card_dict.keys())}")
                if player.get_name() in winning_red_card_dict.keys():
                    player.set_points(player.get_points() + 1)
                    print(f"{player.get_name()} has won the round!")
                    logging.info(f"{player.get_name()} has won the round!")

            # Check for None values
            if self.green_apples_in_play is None:
                logging.error("The green apples in play is None.")
                raise ValueError("The green apples in play is None.")

            if self.red_apples_in_play is None:
                logging.error("The red apples in play is None.")
                raise ValueError("The red apples in play is None.")
            else:
                red_apples_list = []
                for red_apple in self.red_apples_in_play:
                    red_apples_list.append(list(red_apple.values())[0])

            # Log the gameplay results for the round
            results = GameResults(self.players, self.points_to_win, self.round, self.green_apples_in_play[self.current_judge],
                                  red_apples_list, winning_red_card, self.current_judge)
            log_gameplay(results, self.total_games, True)

            # Train all AI agents (if applicable)
            for player in self.players:
                if isinstance(player, AIAgent) and player != self.current_judge:

                    player.train_models(self.nlp_model, self.green_apples_in_play[self.current_judge], winning_red_card, losing_red_apples, self.current_judge)
                    judge_model: Model | None = player.get_opponent_models(self.current_judge)

                    # Check that the judge model is not None
                    if judge_model is None:
                        logging.error("The judge model is None.")
                        raise ValueError("The judge model is None.")

                    current_slope = judge_model.get_slope_vector()
                    current_bias = judge_model.get_bias_vector()
                    preference_updates = PreferenceUpdates(player, self.round, start_time,
                                                   winning_red_card, self.green_apples_in_play[self.current_judge],
                                                   current_bias, current_slope)
                    log_vectors(results, self.total_games, preference_updates)

            # Discard the green cards
            self.discarded_green_apples.append(self.green_apples_in_play[self.current_judge])
            self.green_apples_in_play = None

            # Discard the red cards
            self.discarded_red_apples.extend(red_apples_list)
            self.red_apples_in_play = []

             # Check if the game is over
            self.winner = self.__is_game_over()
            if self.winner is not None:
                # Prepare the winner message
                winner_text = f"# {self.winner.get_name()} has won the game! #"
                border = '#' * len(winner_text)
                message = f"\n{border}\n{winner_text}\n{border}\n"

                # Print and log the winner message
                print(message)
                logging.info(message)
                log_winner(results, self.total_games, True)

                break

            # Assign the next judge
            self.__assign_next_judge()


def range_type(min_value, max_value):
    def range_checker(value):
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError(f"Value must be between {min_value} and {max_value}")
        return ivalue
    return range_checker


def main() -> None:
    # Define the command line arguments
    parser = argparse.ArgumentParser(description="Apples to Apples game configuration.",
                                     usage="python apples_to_apples.py <# of players> <# of points to win> <# of games> [green_expansion] [red_expansion]")
    parser.add_argument("players", type=range_type(3, 8), help="Total number of players (3-8).")
    parser.add_argument("points", type=range_type(1, 10), help="Total number of points to win (1-10).")
    parser.add_argument("total_games", type=int, choices=range(1,1000), help="Total number of games to play (1-1000).")
    parser.add_argument("green_expansion", type=str, nargs='?', default='', help="Filename to a green card expansion (optional).")
    parser.add_argument("red_expansion", type=str, nargs='?', default='', help="Filename to a red card expansion (optional).")

    # Parse the command line arguments
    args = parser.parse_args()

    # Configure and initialize the logging module
    configure_logging()
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
    # game.load_vectors()


    response = 'y'

    # Load the vectors
    game.load_vectors()

    #For this program to automaically play games, choose 'n'. 'y' is for unique situations.
    while True:
        differing_players = input("Will you want to change which players will be playing between games? (y/n): ")
        if (differing_players == "y" or differing_players == "n"):
            break
        print("Invalid input. Type in either 'y' or 'n'.")

    while True:
        vectors_reset = input("Would you like to have the vectors reset in between games? (y/n): ")
        if (vectors_reset == "y" or vectors_reset == "n"):
            break
        print("Invalid input. Type in either 'y' or 'n'.")

    # Start the game
    print (f"------------- GAME 1 of {game.total_games} -------------------")
    game.new_game(True)

    if differing_players == "y":
        game.start_game(False, 0)
    else:
        game.start_game(True, 0)

    #First game is played above since it will be starting a brand new game
    games_played = 1

    # Start the game, prompt the user for options
    while games_played != game.total_games:

        print(f"------------- GAME {games_played+1} of {game.total_games} -------------------")

        if (differing_players == "n"):
            game.new_game(False)
            game.start_game(True, games_played)
            games_played += 1
        else:
            print("--------------------OPTIONS--------------------")
            print("1.Restart the game (same players)\n2.Start a new game (new players)\n3.End Session.\n")
            response = input("Which option do you choose?: ")
            if response == '1':
                game.new_game(False)
                game.start_game(False, games_played)
                games_played += 1
            elif response == '2':
                game.new_game(True)
                game.start_game(False, games_played)
                games_played += 1
            elif response == '3':
                games_played = game.total_games
            else:
                print("Invalid response. Please select '1', '2', or '3'.")


if __name__ == "__main__":
    main()
