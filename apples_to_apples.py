# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules
from source.config import configure_logging
from source.apples import GreenApple, RedApple, Deck
from source.agent import Agent, HumanAgent, RandomAgent, AIAgent, agent_type_mapping
from source.results import GameResults, log_results
from source.w2vloader import VectorsW2V


class ApplesToApples:
    def __init__(self, number_of_players: int, points_to_win: int, green_expansion: str = '', red_expansion: str = '') -> None:
        self.number_of_players: int = number_of_players
        self.points_to_win: int = points_to_win
        self.green_expansion_filename: str = green_expansion
        self.red_expansion_filename: str = red_expansion
        self.green_apples_deck: Deck = Deck()
        self.red_apples_deck: Deck = Deck()
        self.winner: Agent | None = None
        self.players: list[Agent] = []
        self.round: int = 0
        self.current_judge: Agent | None = None
        self.green_apples_in_play: dict[Agent, GreenApple] | None = None
        self.red_apples_in_play: list[dict[str, RedApple]] = []
        self.discarded_green_apples: list[GreenApple] = []
        self.discarded_red_apples: list[RedApple] = []
        self.nlp_model: KeyedVectors = KeyedVectors.load_word2vec_format("./apples/GoogleNews-vectors-negative300.bin", binary=True)
        # self.vectors = VectorsW2V("./apples/GoogleNews-vectors-negative300.bin")
        # embeddings.load()

    def start(self) -> None:
        print("Starting 'Apples to Apples' game.")
        logging.info("Starting 'Apples to Apples' game.")
        logging.info("Initializing players.")

        # # Intro to the game
        # print("\nI assume I am player 1!")
        # logging.info("I assume I am player 1!")

        # Initialize the decks
        self.__initialize_decks()

        # Initialize the players
        self.__initialize_players()

        # Choose the starting judge
        self.__choose_judge()

        # Start the game loop
        self.__game_loop()

    def __initialize_decks(self) -> None:
        # Load the green apples deck
        self.green_apples_deck.load_deck("Green Apples", "./apples/green_apples.csv")
        print(f"Loaded {len(self.green_apples_deck.apples)} green apples.")
        logging.info(f"Loaded {len(self.green_apples_deck.apples)} green apples.")

        # Load the red apples deck
        self.red_apples_deck.load_deck("Red Apples", "./apples/red_apples.csv")
        print(f"Loaded {len(self.red_apples_deck.apples)} red apples.")
        logging.info(f"Loaded {len(self.red_apples_deck.apples)} red apples.")

        # Load the green apples expansion deck
        if self.green_expansion_filename:
            self.green_apples_deck.load_deck("Green Apples Expansion", self.green_expansion_filename)
            print(f"Loaded {len(self.green_apples_deck.apples)} green apples from the expansion.")
            logging.info(f"Loaded {len(self.green_apples_deck.apples)} green apples from the expansion.")

        # Load the red apples expansion deck
        if self.red_expansion_filename:
            self.red_apples_deck.load_deck("Red Apples Expansion", self.red_expansion_filename)
            print(f"Loaded {len(self.red_apples_deck.apples)} red apples from the expansion.")
            logging.info(f"Loaded {len(self.red_apples_deck.apples)} red apples from the expansion.")

        # Shuffle the decks
        self.green_apples_deck.shuffle()
        self.red_apples_deck.shuffle()

    def __generate_unique_name(self, base_name: str) -> str:
        # Unpack the existing names
        existing_names = [agent.name for agent in self.players]

        # Generate a unique name
        i = 1
        while f"{base_name} {i}" in existing_names:
            i += 1
        return f"{base_name} {i}"

    def __initialize_players(self) -> None:
        # Initialize the list of player types
        human_agents: list[HumanAgent] = []
        random_agents: list[RandomAgent] = []
        ai_agents: list[AIAgent] = []

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
                    if new_agent_name not in [agent.name for agent in self.players]:
                        break
                human_agents.append(HumanAgent(new_agent_name))
            elif player_type == '2':
                new_agent_name = self.__generate_unique_name("Random Agent")
                random_agents.append(RandomAgent(new_agent_name))
            elif player_type == '3':
                # Validate the user input for the model type
                model_type: str = ""
                model_type = input("Please enter the model type (1: Linear Regression, 2: Neural Network): ")
                logging.error(f"Please enter the model type (1: Linear Regression, 2: Neural Network): {model_type}")
                while model_type not in ['1', '2']:
                    model_type = input("Invalid input. Please enter the model type (1: Linear Regression, 2: Neural Network): ")
                    logging.error(f"Invalid input. Please enter the model type (1: Linear Regression, 2: Neural Network): {model_type}")

                # Convert the model type from an number to a descriptive string
                if model_type == '1':
                    model_type = "Linear Reg"
                elif model_type == '2':
                    model_type = "Neural Net"

                # Generate a unique name for the AI agent
                new_agent_name = self.__generate_unique_name(f"AI Agent - {model_type}")
                new_agent = AIAgent(new_agent_name)
                ai_agents.append(new_agent)

                # Create the agent object
                new_agent.initialize_models(self.nlp_model, self.players, model_type)
                # new_agent.initialize_models(self.vectors, self.players,model_type)
                logging.info(f"Initialized models for {new_agent.name}.")

            # Create the player object
            self.players.append(agent_type_mapping[player_type](new_agent_name))
            logging.info(self.players[i])

            # Have each player pick up 7 red cards
            self.players[i].draw_red_apples(self.red_apples_deck)

    def __choose_judge(self) -> None:
        # Choose the starting judge
        choice = input(f"\nPlease choose the starting judge (1-{self.number_of_players}): ")
        logging.info(f"Please choose the starting judge (1-{self.number_of_players}): {choice}")

        # Validate the user input
        while not choice.isdigit() or int(choice) < 1 or int(choice) > self.number_of_players:
            choice = input(f"Invalid input. Please enter a number (1-{self.number_of_players}): ")
            logging.error(f"Invalid input. Please enter a number (1-{self.number_of_players}): {choice}")

        # Set the judge
        self.current_judge = self.players[int(choice) - 1]
        self.players[int(choice) - 1].judge = True
        print(f"{self.players[int(choice) - 1].name} is the starting judge.")

    def __assign_next_judge(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Calculate the next judge
        next_judge: Agent = self.players[(self.players.index(self.current_judge) + 1) % self.number_of_players]
        print(f"\n{next_judge.name} is the next judge.")
        logging.info(f"{next_judge.name} is the next judge.")

        # Assign the next judge
        self.current_judge.judge = False
        next_judge.judge = True
        self.current_judge = next_judge

    def __is_game_over(self) -> Agent | None:
        for player in self.players:
            if player.points >= self.points_to_win:
                return player
        return None

    def __judge_prompt(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Prompt the judge to draw a green card
        print(f"\n{self.current_judge.name}, please draw a green card.")
        logging.info(f"{self.current_judge.name}, please draw a green card.")

        # Set the green card in play
        self.green_apples_in_play = {self.current_judge: self.current_judge.draw_green_apple(self.green_apples_deck)}

    def __player_prompt(self) -> None:
        # Prompt the players to select a red card
        for player in self.players:
            if player.judge:
                continue

            print(f"\n{player.name}, please select a red card.")
            logging.info(f"{player.name}, please select a red card.")

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
            self.red_apples_in_play.append({player.name: red_apple})
            logging.info(f"Red card: {red_apple}")

            # Prompt the player to pick up a new red card
            if len(player.red_apples) < 7:
                player.draw_red_apples(self.red_apples_deck)

    def __game_loop(self) -> None:
        # Start the game loop
        while self.winner is None:
            # Increment the round
            self.round += 1

            # Play the round
            print("\n===================")
            print(f"ROUND {self.round}:")
            print("===================")
            logging.info("===================")
            logging.info(f"ROUND {self.round}:")
            logging.info("===================")

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
            print(f"\n{self.current_judge.name}, please select the winning red card.")
            logging.info(f"{self.current_judge.name}, please select the winning red card.")
            winning_red_card = self.current_judge.choose_winning_red_apple(
                self.green_apples_in_play[self.current_judge], self.red_apples_in_play)

            # Award points to the winning player
            for player in self.players:
                logging.debug(f"Agent.name: {player.name}, datatype: {type(player.name)}")
                logging.debug(f"Winning Red Card: {winning_red_card}, datatype: {type(winning_red_card)}")
                logging.debug(f"Winnning Red Card Keys: {winning_red_card.keys()}, datatype: {type(winning_red_card.keys())}")
                if player.name in winning_red_card.keys():
                    player.points += 1
                    print(f"{player.name} has won the round!")
                    logging.info(f"{player.name} has won the round!")

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

            winning_red_card = list(winning_red_card.values())[0]

            # Log the results
            results = GameResults(self.players, self.points_to_win, self.round, self.green_apples_in_play[self.current_judge],
                                  red_apples_list, winning_red_card, self.current_judge)
            log_results(results)

            # Discard the green cards
            self.discarded_green_apples.append(self.green_apples_in_play[self.current_judge])
            self.green_apples_in_play = None

            # Discard the red cards
            self.discarded_red_apples.extend(red_apples_list)
            self.red_apples_in_play = []

             # Check if the game is over
            self.winner = self.__is_game_over()
            if self.winner is not None:
                print("\n##############################")
                print(f"# {self.winner.name} has won the game! #")
                print("##############################")
                logging.info("##############################")
                logging.info(f"# {self.winner.name} has won the game! #")
                logging.info("##############################")
                break

            # Assign the next judge
            self.__assign_next_judge()


def main() -> None:
    # Define the command line arguments
    parser = argparse.ArgumentParser(description="Apples to Apples game configuration.")
    parser.add_argument("players", type=int, choices=range(3, 9), help="Total number of players (3-8).")
    parser.add_argument("points", type=int, choices=range(1, 11), help="Total number of points to win (1-10).")
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
    logging.info(f"Green card expansion file: {args.green_expansion}")
    logging.info(f"Red card expansion file: {args.red_expansion}")

    # Create the game object
    game = ApplesToApples(args.players, args.points, args.green_expansion, args.red_expansion)

    # Start the game
    game.start()


if __name__ == "__main__":
    main()
