# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules
from source.logging import configure_logging
from source.apples import GreenApple, RedApple, Deck
from source.agent import Agent, HumanAgent, AIAgent, model_type_mapping
# from source.model import Model, LRModel, NNModel
from source.logging import GameResults, log_training
from source.w2vloader import VectorsW2V


class ApplesToApples:
    def __init__(self, number_of_rounds: int, green_expansion: str = '', red_expansion: str = '') -> None:
        self.number_of_rounds: int = number_of_rounds
        self.green_expansion_filename: str = green_expansion
        self.red_expansion_filename: str = red_expansion
        self.green_apples_deck: Deck = Deck()
        self.red_apples_deck: Deck = Deck()
        self.__cards_in_hand: int = 25

        self.winner: Agent | None = None
        self.human: Agent = HumanAgent("Human Agent")
        self.agent: Agent | None = None
        self.round: int = 0
        self.current_judge: Agent | None = None
        self.green_apples_in_play: dict[Agent, GreenApple] | None = None
        self.red_apples_in_play: list[dict[str, RedApple]] = []
        self.discarded_green_apples: list[GreenApple] = []
        self.discarded_red_apples: list[RedApple] = []

    def load_vectors(self) -> None:
        self.nlp_model: KeyedVectors = KeyedVectors.load_word2vec_format("./apples/GoogleNews-vectors-negative300.bin", binary=True)
        # self.vectors = VectorsW2V("./apples/GoogleNews-vectors-negative300.bin")
        # embeddings.load()

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
            self.players: list[Agent] = []
            self.__initialize_players()

    def start_game(self) -> None:
        print("Starting new 'Apples to Apples' game.")
        logging.info("Starting new 'Apples to Apples' game.")

        # Choose the starting judge
        self.__choose_judge()

        # Start the game loop
        self.__game_loop()

    def __reset_game_state(self) -> None:
        # Reset the game state
        self.winner: Agent | None = None
        self.round: int = 0
        self.current_judge: Agent | None = None

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
            self.agent.initialize_models(self.nlp_model, human_list)
            logging.info(f"Initialized models for {new_agent.get_name()}.")

    def __choose_judge(self) -> None:
        # Check that the agent is not None
        if self.agent is None:
            logging.error("The agent is None.")
            raise ValueError("The agent is None.")

        self.current_judge = self.agent
        self.agent.set_judge_status(True)
        print(f"{self.agent.get_name()} is the starting judge.")

    def __is_game_over(self) -> Agent | None:
        if self.round >= self.number_of_rounds:
            return self.agent
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

    def __player_prompt(self) -> None:
        # # Check if the agent is None
        # if self.agent is None:
        #     logging.error("The agent is None.")
        #     raise ValueError("The agent is None.")

        print(f"\n{self.human.get_name()}, please select a red card for training purposes.")
        logging.info(f"{self.human.get_name()}, please select a red card for training purposes.")

        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Check if the green apples in play is None
        if self.green_apples_in_play is None:
            logging.error("The green apples in play is None.")
            raise ValueError("The green apples in play is None.")

        # Set the red cards in play
        red_apple = self.human.choose_red_apple(self.current_judge, self.green_apples_in_play[self.current_judge])
        self.red_apples_in_play.append({self.human.get_name(): red_apple})
        logging.info(f"Red card: {red_apple}")

        # Prompt the player to pick up a new red card
        if len(self.human.get_red_apples()) < self.__cards_in_hand:
            self.human.draw_red_apples(self.red_apples_deck, self.__cards_in_hand)

    def __game_loop(self) -> None:
        # Start the game loop
        while self.winner is None:
            # Increment the round
            self.round += 1

            # Print and log the round message
            round_message = f"\n===================" \
                            f"\nROUND {self.round}:" \
                            f"\n===================\n"
            print(round_message)
            logging.info(round_message)

            # Prompt the judge to select a green card
            self.__judge_prompt()

            # Prompt the players to select a red card
            self.__player_prompt()

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

            # Check for None values
            if self.agent is None:
                logging.error("The agent is None.")
                raise ValueError("The agent is None.")

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

            # Extract the winning red card
            winning_red_card: RedApple = list(winning_red_card_dict.values())[0]

            # Put the agent into a list
            agent_list = [self.agent]

            # Log the training
            results = GameResults(agent_list, self.number_of_rounds, self.round, self.green_apples_in_play[self.current_judge],
                                  red_apples_list, winning_red_card, self.current_judge)
            log_training(results, True)

            # Collect all the non-winning red cards
            losing_red_cards = []
            for red_apple in red_apples_list:
                if red_apple != winning_red_card:
                    losing_red_cards.append(red_apple)

            # Train AI agent on human selected apples
            if isinstance(self.agent, AIAgent):
                # Temporarily make the HumanAgent the judge
                self.current_judge = self.human
                self.human.set_judge_status(True)
                self.agent.set_judge_status(False)

                # Train the AI agent
                self.agent.train_models(self.nlp_model, self.green_apples_in_play[self.agent], winning_red_card, losing_red_cards, self.current_judge)

                # Reset the judge to the AI agent
                self.current_judge = self.agent
                self.agent.set_judge_status(True)
                self.human.set_judge_status(False)


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

                break


def range_type(min_value, max_value):
    def range_checker(value):
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError(f"Value must be between {min_value} and {max_value}")
        return ivalue
    return range_checker


def main() -> None:
    # Define the command line arguments
    parser = argparse.ArgumentParser(description="Apples to Apples agent training configuration.",
                                     usage="python agent_training.py <# of rounds> [green_expansion] [red_expansion]")
    parser.add_argument("rounds", type=range_type(3, 8), help="Total number of rounds (1-100).")
    parser.add_argument("green_expansion", type=str, nargs='?', default='', help="Filename to a green card expansion (optional).")
    parser.add_argument("red_expansion", type=str, nargs='?', default='', help="Filename to a red card expansion (optional).")

    # Parse the command line arguments
    args = parser.parse_args()

    # Configure and initialize the logging module
    configure_logging()
    logging.info("Starting 'Apples to Apples' agent training application.")

    # Log the command line arguments
    logging.info(f"Command line arguments: {args}")
    logging.info(f"Number of rounds: {args.rounds}")
    logging.info(f"Green card expansion file: {args.green_expansion}")
    logging.info(f"Red card expansion file: {args.red_expansion}")

    # Create the game object
    game = ApplesToApples(args.rounds, args.green_expansion, args.red_expansion)

    # Load the vectors
    game.load_vectors()

    # Start the game
    game.new_game(True)
    game.start_game()


if __name__ == "__main__":
    main()
