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
from source.agent import Agent, HumanAgent, AIAgent
from source.model import Model, model_type_mapping
from source.game_logger import GameResults, PreferenceUpdates, log_vectors, log_training
from source.w2vloader import VectorsW2V
from apples_to_apples import range_type, get_user_input_y_or_n


class ApplesToApples:
    def __init__(self, number_of_rounds: int, green_expansion: str = '', red_expansion: str = '') -> None:
        # Static game parameters
        self.number_of_rounds: int = number_of_rounds
        self.green_expansion_filename: str = green_expansion
        self.red_expansion_filename: str = red_expansion
        self.green_apples_deck: Deck = Deck()
        self.red_apples_deck: Deck = Deck()
        self.__cards_in_hand: int = 25
        # Dynamic game parameters
        self.human: Agent = HumanAgent("Human Agent")
        self.agent: Agent | None = None
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

    def set_between_game_options(self, train_on_extra_vectors: bool, train_on_losing_red_apples: bool) -> None:
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

        # Reset the game state
        print("Resetting game state.")
        logging.info("Resetting game state.")
        self.__reset_game_state()

        # Initialize the decks
        print("Initializing decks.")
        logging.info("Initializing decks.")
        self.__initialize_decks()

        # Initialize the players
        print("Initializing players.")
        logging.info("Initializing players.")
        self.players: list[Agent] = []
        self.__initialize_players()

        # Choose the starting judge
        self.__choose_starting_judge()

        # Start the game loop
        self.__game_loop()

    def __reset_game_state(self) -> None:
        # Reset the game state
        self.current_round = 0
        self.current_judge = None
        self.game_winner = None

        # # Reset the player points and judge status
        # for player in self.players:
        #     player.reset_points()
        #     player.set_judge_status(False)

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

    def __initialize_players(self) -> None:
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

        # Create a new AI agent
        new_agent_name = f"AI Agent - {model_type_class.__name__} - {pretrained_model_string}"
        new_agent = AIAgent(new_agent_name, model_type_class, pretrained_model_string, True)

        # Append the player object
        self.agent = new_agent
        logging.info(self.agent)

        # Have the human player pick up 25 red cards
        self.human.draw_red_apples(self.red_apples_deck, self.__cards_in_hand)

        # Add the human player to a list
        human_list = [self.human]

        # Initialize the models for the AI agents
        if isinstance(self.agent, AIAgent):
            self.agent.initialize_models(self.keyed_vectors, human_list)
            logging.info(f"Initialized models for {new_agent.get_name()}.")

    def __choose_starting_judge(self) -> None:
        # Check that the agent is not None
        if self.agent is None:
            logging.error("The agent is None.")
            raise ValueError("The agent is None.")

        self.current_judge = self.agent
        self.agent.set_judge_status(True)
        print(f"{self.agent.get_name()} is the starting judge.")

    def __assign_next_judge(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

    def __judge_prompt(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Prompt the judge to draw a green card
        print(f"\n{self.current_judge.get_name()}, please draw a green card.")
        logging.info(f"{self.current_judge.get_name()}, please draw a green card.")

        # Set the green card in play
        self.green_apple_in_play = {self.current_judge: self.current_judge.draw_green_apple(self.green_apples_deck)}

    def __player_prompt(self) -> list[RedApple]:
        # # Check if the agent is None
        # if self.agent is None:
        #     logging.error("The agent is None.")
        #     raise ValueError("The agent is None.")
        red_apples: list[RedApple] = []

        print(f"\n{self.human.get_name()}, please select a red card for training purposes.")
        logging.info(f"{self.human.get_name()}, please select a red card for training purposes.")

        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Check if the green apples in play is None
        if self.green_apple_in_play is None:
            logging.error("The green apples in play is None.")
            raise ValueError("The green apples in play is None.")

        # Set the red cards in play
        red_apple = self.human.choose_red_apple(self.current_judge, self.green_apple_in_play[self.current_judge])
        self.red_apples_in_play.append({self.human: red_apple})
        red_apples.append(red_apple)
        logging.info(f"Red card: {red_apple}")

        # Prompt the player to pick up a new red card
        if len(self.human.get_red_apples()) < self.__cards_in_hand:
            self.human.draw_red_apples(self.red_apples_deck, self.__cards_in_hand)

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

        # # Take a snapshot of the current date and time
        # date_time = datetime.now().strftime("%Y-%m-%d-%H-%M)")

        # # Train all AI agents (if applicable)
        # for player in self.players:
        #     if isinstance(player, AIAgent) and player != self.current_judge:

        #         player.train_models(self.keyed_vectors, self.green_apple_in_play[self.current_judge],
        #                             game_results.winning_red_apple, game_results.losing_red_apples, self.current_judge)
        #         judge_model: Model | None = player.get_opponent_models(self.current_judge)

        #         # Check that the judge model is not None
        #         if judge_model is None:
        #             logging.error("The judge model is None.")
        #             raise ValueError("The judge model is None.")

        #         current_slope = judge_model.get_slope_vector()
        #         current_bias = judge_model.get_bias_vector()
        #         preference_updates = PreferenceUpdates(player, self.current_round, date_time,
        #                                         game_results.winning_red_apple, self.green_apple_in_play[self.current_judge],
        #                                         current_bias, current_slope)
        #         log_vectors(game_results, preference_updates)

        # Train AI agent on human selected apples
        if isinstance(self.agent, AIAgent):
            # Temporarily make the HumanAgent the judge
            self.current_judge = self.human
            self.human.set_judge_status(True)
            self.agent.set_judge_status(False)

            # Train the AI agent
            logging.debug(f"Training the AI agent on the human selected apples.")
            logging.debug(f"Green Apple: {self.green_apple_in_play[self.agent]}")
            logging.debug(f"Winning Red Apple: {game_results.winning_red_apple}")
            # logging.debug(f"Losing Red Apples: {losing_red_apples}")
            self.agent.train_opponent_models(
                self.keyed_vectors, self.current_judge,
                self.green_apple_in_play[self.agent],
                game_results.winning_red_apple,
                game_results.losing_red_apples,
                self.train_on_extra_vectors,
                self.train_on_losing_red_apples
            )

            # Reset the judge to the AI agent
            self.current_judge = self.agent
            self.agent.set_judge_status(True)
            self.human.set_judge_status(False)

    def __is_game_over(self) -> Agent | None:
        if self.current_round >= self.number_of_rounds:
            return self.agent
        return None

    def __game_loop(self) -> None:
        # Start the game loop
        while self.round_winner is None:
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

            # Check if the green apples in play is None
            if self.green_apple_in_play is None:
                logging.error("The green apples in play is None.")
                raise ValueError("The green apples in play is None.")

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
            print(f"{self.human.get_name()} chose the winning red card '{winning_red_apple}'.")
            logging.info(f"{self.human.get_name()} chose the winning red card '{winning_red_apple}'.")

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

            # Check if the agent is None
            if self.agent is None:
                logging.error("The agent is None.")
                raise ValueError("The agent is None.")

            # Put the agent into a list
            agent_list = [self.agent]

            # Check if the game is over
            self.round_winner = self.__is_game_over()

            # Log the training
            results = GameResults(agent_list, self.number_of_rounds, self.number_of_rounds, self.current_round, self.current_round,
                                  self.green_apple_in_play[self.current_judge], self.red_apples_in_play, winning_red_apple, losing_red_apples,
                                  self.current_judge, self.current_judge, self.current_judge)
            log_training(results, True)

            # Train the AI agents
            self.__train_ai_agents(results)

            # Check if the game is over and print the winner message
            if self.round_winner is not None:
                # Prepare the winner message
                winner_text = f"# {self.round_winner.get_name()} has won the game! #"
                border = '#' * len(winner_text)
                message = f"\n{border}\n{winner_text}\n{border}\n"

                # Print and log the winner message
                print(message)
                logging.info(message)

            # Reset the round state
            self.__discard_apples_in_play()


def main() -> None:
    # Define the command line arguments
    parser = argparse.ArgumentParser(
        prog="'Apples to Apples' agent training program",
        usage="python agent_training.py <# of rounds> [green_expansion] [red_expansion] [-D]",
        description="Configure and run the 'Apples to Apples' agent training program. "\
                    "Specify the number of rounds of training, and optional green and red card expansions. "
                    "Use the '-D' flag to enable debug mode for detailed logging."
    )

    # Add the command line arguments
    parser.add_argument("rounds", type=range_type(1, 100), help="Total number of rounds (1-100).")
    parser.add_argument("green_expansion", type=str, nargs='?', default='', help="Filename to a green card expansion (optional).")
    parser.add_argument("red_expansion", type=str, nargs='?', default='', help="Filename to a red card expansion (optional).")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug mode")

    # Parse the command line arguments
    args = parser.parse_args()

    # Configure and initialize the logging module
    configure_logging(args.debug)
    logging.info("Starting 'Apples to Apples' agent training application.")

    # Log the command line arguments
    logging.info(f"Command line arguments: {args}")
    logging.info(f"Number of rounds: {args.rounds}")
    logging.info(f"Green card expansion file: {args.green_expansion}")
    logging.info(f"Red card expansion file: {args.red_expansion}")

    # Create the game object
    game = ApplesToApples(args.rounds, args.green_expansion, args.red_expansion)

    # Load the keyed vectors
    game.load_keyed_vectors()

    # Initialize all between game option variables
    train_on_extra_vectors = 'n'
    train_on_losing_red_apples = 'n'

    # Prompt the user on whether they want to include the synonym and description vectors as part of the model
    train_on_extra_vectors = get_user_input_y_or_n("Do you want to include the synonym and description vectors as part of the model training? (y/n): ")

    # Prompt the user on whether they want to include the losing red apples as part of the model training
    train_on_losing_red_apples = get_user_input_y_or_n("Do you want to include the losing red apples as part of the model training? (y/n): ")

    # Set the between game options
    game.set_between_game_options(
        train_on_extra_vectors == 'y',
        train_on_losing_red_apples == 'y'
    )

    # Start the game
    game.new_game()


if __name__ == "__main__":
    main()
