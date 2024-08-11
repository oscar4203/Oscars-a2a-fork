# Description: 'Apples to Apples' game class.

# Standard Libraries
import logging

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules
from source.w2vloader import VectorsW2V
from source.apples import GreenApple, RedApple, Deck
from source.agent import Agent, HumanAgent, RandomAgent, AIAgent
from source.model import Model, model_type_mapping
from source.game_logger import print_and_log, format_naming_scheme, \
                            log_vectors, log_gameplay, log_winner, log_training, \
                            LOGGING_BASE_DIRECTORY
from source.data_classes import GameState


class ApplesToApples:
    def __init__(self, keyed_vectors: KeyedVectors, training_mode: bool, green_expansion: str = '', red_expansion: str = '') -> None:
        self.__keyed_vectors: KeyedVectors = keyed_vectors
        self.__training_mode: bool = training_mode
        self.__green_expansion_filename: str = green_expansion
        self.__red_expansion_filename: str = red_expansion
        self.__green_apples_deck: Deck = Deck()
        self.__red_apples_deck: Deck = Deck()

    def initalize_game_state(self, game_state: GameState) -> None:
        self.__game_state = game_state

    def get_game_state(self) -> GameState:
        return self.__game_state

    def set_game_options(self, change_players: bool, cycle_starting_judges: bool, reset_models: bool, train_on_extra_vectors: bool, train_on_losing_red_apples: bool) -> None:
        self.__change_players_between_games = change_players
        self.__cycle_starting_judges_between_games = cycle_starting_judges
        self.__reset_models_between_games = reset_models
        self.__train_on_extra_vectors = train_on_extra_vectors
        self.__train_on_losing_red_apples = train_on_losing_red_apples

    def new_game(self) -> None:
        """
        Start a new game of 'Apples to Apples' and reset the game state.
        Optionally, initialize new players.
        """
        print_and_log("Starting new 'Apples to Apples' game.")

        # Increment the current game counter
        self.__game_state.current_game += 1

        # Print and log the game message
        print_and_log(f"\n------------- GAME {self.__game_state.current_game} of {self.__game_state.total_games} -------------")

        # Reset the game state
        print_and_log("Resetting game state.")
        self.__reset_game_state()

        # Initialize the decks
        print_and_log("Initializing decks.")
        self.__initialize_decks()

        # Initialize the players for the first game
        if self.__game_state.current_game == 1:
            print_and_log("Initializing players.")
            self.__game_state.players = []
            self.__initialize_players()
        elif self.__game_state.current_game > 1:
            # Reset the opponent models for the AI agents, if applicable
            if self.__reset_models_between_games:
                self.__reset_opponent_models()

            # Prompt the user on whether to keep the same players, if applicable
            if self.__change_players_between_games:
                from game_driver import get_user_input_y_or_n
                keep_players = get_user_input_y_or_n("Do you want to keep the same players as last game? (y/n): ")
                if keep_players == "n":
                    self.__game_state.players = []
                    self.__initialize_players()

        # Define the naming scheme and the winner csv filepath
        naming_scheme = format_naming_scheme(self.__game_state.players, self.__game_state.total_games, self.__game_state.points_to_win)
        self.winner_csv_filepath = f"{LOGGING_BASE_DIRECTORY}{naming_scheme}/winners-{naming_scheme}.csv"

        # Choose the starting judge
        self.__choose_starting_judge()

        # Start the game loop
        self.__game_loop()

    def __reset_game_state(self) -> None:
        # Reset the game state for a new game
        self.__game_state.current_round = 0
        self.__game_state.current_judge = None
        self.__game_state.game_winner = None

        # TODO - check if need to skip for training mode
        # Reset the player points and judge status
        for player in self.__game_state.players:
            player.reset_points()
            player.set_judge_status(False)

    def __new_round(self) -> None:
        # Reset the game state for a new round
        self.__game_state.round_winner = None

        # Increment the round counter
        self.__game_state.current_round += 1

        # Print and log the round message
        round_message = f"\n===================" \
                        f"\nROUND {self.__game_state.current_round}:" \
                        f"\n===================\n"
        print_and_log(round_message)

        # Print and log the player points
        for player in self.__game_state.players:
            print_and_log(f"{player.get_name()}: {player.get_points()} points")

    def __end_of_round(self) -> None:
        # Discard the apples in play
        self.__discard_apples_in_play()

        # Assign the next judge if not in training mode
        if not self.__training_mode:
            self.__assign_next_judge()

    def __initialize_decks(self) -> None:
        # Initialize the decks
        self.__game_state.green_apple = None
        self.__game_state.red_apples = []
        self.discarded_green_apples: list[GreenApple] = []
        self.discarded_red_apples: list[RedApple] = []

        # Shuffle the decks
        self.__load_and_shuffle_deck(self.__green_apples_deck, "Green Apples", "./apples/green_apples.csv", self.__green_expansion_filename)
        self.__load_and_shuffle_deck(self.__red_apples_deck, "Red Apples", "./apples/red_apples.csv", self.__red_expansion_filename)

    def __load_and_shuffle_deck(self, deck: Deck, deck_name: str, base_file: str, expansion_file: str) -> None:
        # Load the base deck
        deck.load_deck(deck_name, base_file)
        print_and_log(f"Loaded {len(deck.get_apples())} {deck_name.lower()}.")

        # Load the expansion deck, if applicable
        if expansion_file:
            deck.load_deck(f"{deck_name} Expansion", expansion_file)
            print_and_log(f"Loaded {len(deck.get_apples())} {deck_name.lower()} from the expansion.")

        # Shuffle the deck
        deck.shuffle()

    def __discard_apples_in_play(self) -> None:
        # Check if the current judge is None
        if self.__game_state.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Check if the green apple in play is None
        if self.__game_state.green_apple is None:
            logging.error("The green apple in play is None.")
            raise ValueError("The green apple in play is None.")

        # Check if the red apples in play is None
        if self.__game_state.red_apples is None:
            logging.error("The red apples in play is None.")
            raise ValueError("The red apples in play is None.")

        # Create red_apples_list using list comprehension
        red_apples_list = [list(red_apple.values())[0] for red_apple in self.__game_state.red_apples]

        # Discard the green cards
        self.discarded_green_apples.append(self.__game_state.green_apple[self.__game_state.current_judge])
        self.__game_state.green_apple = None

        # Discard the red apples
        self.discarded_red_apples.extend(red_apples_list)
        self.__game_state.red_apples = []

    def __generate_unique_agent_name(self, base_name: str) -> str:
        # Unpack the existing names
        existing_names = [agent.get_name() for agent in self.__game_state.players]

        # Generate a unique name
        i = 1
        while f"{base_name} {i}" in existing_names:
            i += 1
        return f"{base_name} {i}"

    def __initialize_players(self) -> None:
        # Display the number of players
        print_and_log(f"There are {self.__game_state.number_of_players} players.")

        if self.__training_mode:
            # Validate the user input for the model type
            model_type: str = input("Please enter the model type (1: Linear Regression, 2: Neural Network): ")
            logging.info(f"Please enter the model type (1: Linear Regression, 2: Neural Network): {model_type}")
            while model_type not in ['1', '2']:
                model_type = input("Invalid input. Please enter the model type (1: Linear Regression, 2: Neural Network): ")
                logging.error(f"Invalid input. Please enter the model type (1: Linear Regression, 2: Neural Network): {model_type}")

            # Validate the user input for the pretrained model type
            pretrained_model_type: str = input("Please enter the pretrained model type (1: Literalist, 2: Contrarian, 3: Comedian): ")
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

            # Create the human agent
            human_agent = HumanAgent("Human Agent")

            # Have the human agent pick up max red apples in hand
            human_agent.draw_red_apples(self.__keyed_vectors, self.__red_apples_deck, self.__game_state.max_cards_in_hand, self.__train_on_extra_vectors)

            # Append the human agent and AI agent
            self.__game_state.players.append(human_agent) # APPEND HUMAN AGENT FIRST!!!
            self.__game_state.players.append(new_agent)

        else:
            # Create the players when not in training mode
            for i in range(self.__game_state.number_of_players):
                # Prompt the user to select the player type
                print_and_log(f"\nWhat type is Agent {i + 1}?")
                player_type: str = input("Please enter the player type (1: Human, 2: Random, 3: AI): ")

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
                        if new_agent_name not in [agent.get_name() for agent in self.__game_state.players]:
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
                self.__game_state.players.append(new_agent)
                logging.info(self.__game_state.players[i])

                # Have each player pick up max red apples in hand
                self.__game_state.players[i].draw_red_apples(self.__keyed_vectors, self.__red_apples_deck, self.__game_state.max_cards_in_hand, self.__train_on_extra_vectors)

        # Initialize the models for the AI agents
        for player in self.__game_state.players:
            if isinstance(player, AIAgent):
                player.initialize_models(self.__keyed_vectors, self.__game_state.players)
                logging.info(f"Initialized models for {new_agent.get_name()}.")

    def __choose_starting_judge(self) -> None:
        # Clear the judge status for all players
        for i, player in enumerate(self.__game_state.players):
            self.__game_state.players[i].set_judge_status(False)

        if self.__training_mode:
            # Choose the ai agent as the starting judge
            for player in self.__game_state.players:
                if isinstance(player, AIAgent):
                    judge_index = self.__game_state.players.index(player)
                    break
        else:
            # If cycle starting judge is True, choose the starting judge automatically
            if self.__cycle_starting_judges_between_games:
                # Cycle through the judges to get the judge index
                judge_index = ((self.__game_state.current_game - 1) % len(self.__game_state.players)) # Subtract 1 since 0-based index and current_game starts at 1
            else: # If cycle starting judge is False, prompt the user to choose the starting judge
                if self.__change_players_between_games:
                    # Choose the starting judge
                    choice = input(f"\nPlease choose the starting judge (1-{self.__game_state.number_of_players}): ")
                    logging.info(f"Please choose the starting judge (1-{self.__game_state.number_of_players}): {choice}")

                    # Validate the user input
                    while not choice.isdigit() or int(choice) < 1 or int(choice) > self.__game_state.number_of_players:
                        choice = input(f"Invalid input. Please enter a number (1-{self.__game_state.number_of_players}): ")
                        logging.error(f"Invalid input. Please enter a number (1-{self.__game_state.number_of_players}): {choice}")

                    # Set the current judge index
                    judge_index = int(choice) - 1 # Subtract 1 since 0-based index
                else:
                    judge_index = 0

        # Assign the starting judge and set the judge status
        self.__game_state.current_judge = self.__game_state.players[judge_index]
        self.__game_state.current_judge.set_judge_status(True)
        print_and_log(f"{self.__game_state.current_judge.get_name()} is the starting judge.")

    def __assign_next_judge(self) -> None:
        # Check if the current judge is None
        if self.__game_state.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Clear the judge status for the current judge
        self.__game_state.current_judge.set_judge_status(False)

        # Assign the next judge and set the judge status
        self.__game_state.current_judge = self.__game_state.players[(self.__game_state.players.index(self.__game_state.current_judge) + 1) % self.__game_state.number_of_players]
        self.__game_state.current_judge.set_judge_status(True)
        print_and_log(f"{self.__game_state.current_judge.get_name()} is the next judge.")

    def __judge_prompt(self) -> None:
        # Check if the current judge is None
        if self.__game_state.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Prompt the judge to draw a green card
        print_and_log(f"\n{self.__game_state.current_judge.get_name()}, please draw a green card.")

        # Set the green card in play
        self.__game_state.green_apple = {self.__game_state.current_judge: self.__game_state.current_judge.draw_green_apple(self.__keyed_vectors, self.__green_apples_deck, self.__train_on_extra_vectors)}

    def __player_prompt(self) -> list[RedApple]:
        red_apples: list[RedApple] = []
        # Prompt the players to select a red apple
        for player in self.__game_state.players:
            if player.get_judge_status():
                continue

            print_and_log(f"\n{player.get_name()}, please select a red apple.")

            # Check if the current judge is None
            if self.__game_state.current_judge is None:
                logging.error("The current judge is None.")
                raise ValueError("The current judge is None.")

            # Check if the green apple in play is None
            if self.__game_state.green_apple is None:
                logging.error("The green apple in play is None.")
                raise ValueError("The green apple in play is None.")

            # Set the red apples in play
            red_apple = player.choose_red_apple(
                self.__game_state.current_judge,
                self.__game_state.green_apple[self.__game_state.current_judge],
                self.__train_on_losing_red_apples)
            self.__game_state.red_apples.append({player: red_apple})
            red_apples.append(red_apple)
            logging.info(f"Chosen red apple: {red_apple}")

            # Prompt the player to pick up a new red apple
            if len(player.get_red_apples()) < self.__game_state.max_cards_in_hand:
                player.draw_red_apples(self.__keyed_vectors, self.__red_apples_deck, self.__game_state.max_cards_in_hand, self.__train_on_extra_vectors)

        return red_apples

    def __train_ai_agents(self, game_state: GameState) -> None:
        # Check if the current judge is None
        if game_state.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Check if the green apple in play is None
        if game_state.green_apple is None:
            logging.error("The green apple in play is None.")
            raise ValueError("The green apple in play is None.")

        # Check if the winning red apple is None
        if game_state.winning_red_apple is None:
            logging.error("The winning red apple is None.")
            raise ValueError("The winning red apple is None.")

        # Train all AI agents (if applicable)
        for player in self.__game_state.players:
            # Train the AI agent
            if isinstance(player, AIAgent):
                # In training mode, train on the human agent
                if self.__training_mode:
                    for agent in self.__game_state.players:
                        if isinstance(agent, HumanAgent):
                            player.train_opponent_judge_model(
                                agent,
                                game_state.green_apple[game_state.current_judge],
                                game_state.winning_red_apple,
                                game_state.losing_red_apples,
                                self.__train_on_extra_vectors,
                                self.__train_on_losing_red_apples
                            )

                            # Get the opponent judge model
                            opponent_judge_model: Model | None = player.get_opponent_model(agent)

                            # Check that the judge model is not None
                            if opponent_judge_model is None:
                                logging.error("The opponent judge model is None.")
                                raise ValueError("The opponent judge model is None.")

                            current_slope, current_bias = opponent_judge_model.get_current_slope_and_bias_vectors()
                            log_vectors(game_state, player, current_slope, current_bias, True)
                else:
                    # In non-training mode, train only if the player is not the current judge
                    if player != game_state.current_judge:
                        player.train_opponent_judge_model(
                            game_state.current_judge,
                            game_state.green_apple[game_state.current_judge],
                            game_state.winning_red_apple,
                            game_state.losing_red_apples,
                            self.__train_on_extra_vectors,
                            self.__train_on_losing_red_apples
                        )

                        # Get the opponent judge model
                        opponent_judge_model: Model | None = player.get_opponent_model(game_state.current_judge)

                        # Check that the judge model is not None
                        if opponent_judge_model is None:
                            logging.error("The opponent judge model is None.")
                            raise ValueError("The opponent judge model is None.")

                        current_slope, current_bias = opponent_judge_model.get_current_slope_and_bias_vectors()
                        log_vectors(game_state, player, current_slope, current_bias, True)

    def __reset_opponent_models(self) -> None:
        # TODO - check if need to skip for training mode
        for player in self.__game_state.players:
            if isinstance(player, AIAgent):
                player.reset_opponent_models()

    def __determine_round_winner(self, red_apples_this_round: list[RedApple]) -> tuple[RedApple, list[RedApple]]:
            # Check if the current judge is None
            if self.__game_state.current_judge is None:
                logging.error("The current judge is None.")
                raise ValueError("The current judge is None.")

            # Check if the green apple in play is None
            if self.__game_state.green_apple is None:
                logging.error("The green apple in play is None.")
                raise ValueError("The green apple in play is None.")

            # Prompt the judge to select the winning red apple
            print_and_log(f"\n{self.__game_state.current_judge.get_name()}, please select the winning red apple.")
            winning_red_apple_dict: dict[Agent, RedApple] = self.__game_state.current_judge.choose_winning_red_apple(
                self.__game_state.green_apple[self.__game_state.current_judge], self.__game_state.red_apples)

            # Extract the winning red apple
            winning_red_apple: RedApple = list(winning_red_apple_dict.values())[0]

            # Print and log the winning red apple
            print_and_log(f"{self.__game_state.current_judge.get_name()} chose the winning red apple '{winning_red_apple}'.")

            # Extract the losing red apples
            losing_red_apples: list[RedApple] = red_apples_this_round.copy()
            losing_red_apples.remove(winning_red_apple)
            logging.info(f"Losing Apples: {losing_red_apples}")

            # Extract the round winner
            round_winner: Agent = list(winning_red_apple_dict.keys())[0]

            # Verify the winning player is in the list of players
            if round_winner not in self.__game_state.players:
                logging.error(f"Round winner {round_winner} not in list of players.")
                raise ValueError(f"Round winner {round_winner} not in list of players.")

            # Set the round winner
            self.__game_state.round_winner = round_winner
            print_and_log(f"{self.__game_state.round_winner.get_name()} has won the round!")
            logging.debug(f"{self.__game_state.round_winner} has won the round!")

            # Award points to the winning player
            self.__game_state.round_winner.set_points(self.__game_state.round_winner.get_points() + 1)

            return winning_red_apple, losing_red_apples

    def __is_game_over(self) -> Agent | None:
        for player in self.__game_state.players:
            if player.get_points() >= self.__game_state.points_to_win:
                return player
        return None

    def __game_loop(self) -> None:
        # Start the game loop
        while self.__game_state.game_winner is None:
            # Increment the round counter and print the round messages
            self.__new_round()

            # Prompt the judge to select a green card
            self.__judge_prompt()

            # Prompt the players to select a red apple
            red_apples_this_round  = self.__player_prompt()

            # Determine the winning red apple and losing red apples
            self.__game_state.winning_red_apple, self.__game_state.losing_red_apples = self.__determine_round_winner(red_apples_this_round)

            # Check if the game is over
            self.__game_state.game_winner = self.__is_game_over()

            # Log the gameplay or training results
            if not self.__training_mode:
                log_gameplay(self.__game_state, True)
            else:
                log_training(self.__game_state, True)

            # Train the AI agents
            self.__train_ai_agents(self.__game_state)

            # End of the round cleanup
            self.__end_of_round()

            # Print the winner message
            if self.__game_state.game_winner is not None:
                # Prepare the winner message
                winner_text = f"# {self.__game_state.game_winner.get_name()} has won the game! #"
                border = '#' * len(winner_text)
                message = f"\n{border}\n{winner_text}\n{border}\n"

                # Print and log the winner message
                print_and_log(message)

                # Log the winner if not in training mode
                if not self.__training_mode:
                    log_winner(self.__game_state, True)


if __name__ == "__main__":
    pass
