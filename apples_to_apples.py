# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse

# Third-party Libraries
from gensim.models import KeyedVectors
from datetime import datetime

# Local Modules
from source.config import configure_logging
from source.apples import GreenApple, RedApple, Deck
from source.agent import Agent, HumanAgent, RandomAgent, AIAgent, model_type_mapping
# from source.model import Model, LRModel, NNModel
from source.results import GameResults, log_results, JudgePreferences, log_preferences, PreferenceUpdates, log_preference_updates
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
                new_agent = HumanAgent(new_agent_name)
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

                # Generate a unique name for the AI agent
                model_type_class = model_type_mapping[model_type]
                logging.debug(f"Model Type Class: {model_type_class}")
                logging.debug(f"Model Type Name: {model_type_class.__name__}")
                new_agent_name = self.__generate_unique_name(f"AI Agent - {model_type_class.__name__}")
                new_agent = AIAgent(new_agent_name, model_type_class)

            # Append the player object
            self.players.append(new_agent)
            logging.info(self.players[i])

            # Have each player pick up 7 red cards
            self.players[i].draw_red_apples(self.red_apples_deck)

        # Initialize the models for the AI agents
        for player in self.players:
            if isinstance(player, AIAgent):
                player.initialize_models(self.nlp_model, self.players)
                logging.info(f"Initialized models for {new_agent.name}.")


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
        start_time = datetime.now().strftime("%Y-%m-%w-%H-%M-%S)")

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
                print(f"{player.name}: {player.points} points")
                logging.info(f"{player.name}: {player.points} points")

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
            winning_red_card_dict: dict[str, RedApple] = self.current_judge.choose_winning_red_apple(
                self.green_apples_in_play[self.current_judge], self.red_apples_in_play)

            # Award points to the winning player
            for player in self.players:
                logging.debug(f"Agent.name: {player.name}, datatype: {type(player.name)}")
                logging.debug(f"Winning Red Card: {winning_red_card_dict}, datatype: {type(winning_red_card_dict)}")
                logging.debug(f"Winnning Red Card Keys: {winning_red_card_dict.keys()}, datatype: {type(winning_red_card_dict.keys())}")
                if player.name in winning_red_card_dict.keys():
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

            # Extract the winning red card
            winning_red_card: RedApple = list(winning_red_card_dict.values())[0]

            # Log the results
            results = GameResults(self.players, self.points_to_win, self.round, self.green_apples_in_play[self.current_judge],
                                  red_apples_list, winning_red_card, self.current_judge)
            log_results(results)

            # #checks if there is an AI player in the game, and if so, logs that players preferences when
            # #they are a judge on a given round
            # for player in self.players:
            #     if (isinstance(player, AIAgent) and player.judge == True):
            #         judge_preferences = JudgePreferences(self.current_judge, self.round, player.self_model.bias_vector, player.self_model.slope_vector)
            #         log_preferences(judge_preferences)
            #==========================================================
            # opposing_players = self.players.copy()
            # opposing_players.remove(self.current_judge)
            # biases_list = []
            # slopes_list = []
            # for player in opposing_players:
            #     biases_list.append(player.self_model.bias_vector)
            #     slopes_list.append(player.self_model.slope_vector)


            # if(isinstance(self.current_judge, AIAgent)):
            #     opposing_players = self.current_judge.opponent_models
            #     opposing_models = opposing_players.values()
            #     opposing_agents = opposing_players.keys()
            #     biases_list = []
            #     slopes_list = []
            #     for player in opposing_models:
            #         biases_list.append(player.bias_vector)
            #         slopes_list.append(player.slope_vector)

                # preference_updates = PreferenceUpdates(opposing_models, opposing_agents, self.round, start_time, 
                #                                    winning_red_card, self.green_apples_in_play[self.current_judge], 
                #                                    biases_list, slopes_list, "")
                # log_preference_updates(preference_updates)

            # if model.judge != player in for loop, get player model


            #TEMPORARY UNTIL WE MERGE BOTH THIS CODE AND THE ONE ON ISAAC'S DESKTOP
            losing_red_cards = [value for d in self.red_apples_in_play for value in d.values()]
            # losing_red_cards = [d for d in losing_reds if winning_red_card not in d.values()]

            # list_of_values = [value for d in list_of_dicts for value in d.values()]

            # Train all AI agents (if applicable)
            for player in self.players:
                if isinstance(player, AIAgent) and player != self.current_judge:
                    # opposing_players = self.current_judge.opponent_models
                    # opposing_models = opposing_players.values()
                    # opposing_agents = opposing_players.keys()

                    player.train_models(self.nlp_model, self.green_apples_in_play[self.current_judge], winning_red_card,  self.current_judge, losing_red_cards)
                    judge_model = player.opponent_models[self.current_judge]
                    current_slope = judge_model.slope_vector
                    current_bias = judge_model.bias_vector
                    preference_updates = PreferenceUpdates(self.round, start_time, 
                                                   winning_red_card, self.green_apples_in_play[self.current_judge], 
                                                   current_bias, current_slope, "")
                    log_preference_updates(preference_updates)

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
                winner_text = f"# {self.winner.name} has won the game! #"
                border = '#' * len(winner_text)
                message = f"\n{border}\n{winner_text}\n{border}\n"

                # Print and log the winner message
                print(message)
                logging.info(message)

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
