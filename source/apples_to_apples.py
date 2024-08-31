# Description: 'Apples to Apples' game class.

# Standard Libraries
import logging

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules
from source.w2vloader import VectorsW2V
from source.embeddings import Embedding
from source.apples import GreenApple, RedApple, Deck
from source.agent import Agent, HumanAgent, RandomAgent, AIAgent
from source.model import model_type_mapping
from source.game_logger import log_game_state, log_round_winner, log_game_winner, log_training
from source.data_classes import RoundState, GameState, GameLog, ApplesInPlay


class ApplesToApples:
    def __init__(self, embedding: Embedding, training_mode: bool, green_expansion: str = '', red_expansion: str = '') -> None:
        self.embedding: Embedding = embedding
        self.__training_mode: bool = training_mode
        self.__green_expansion_filename: str = green_expansion
        self.__red_expansion_filename: str = red_expansion
        self.__green_apples_deck: Deck = Deck()
        self.__red_apples_deck: Deck = Deck()

    def initalize_game_log(self, game_log: GameLog) -> None:
        self.__game_log = game_log

    def set_game_options(self, change_players: bool, cycle_starting_judges: bool, reset_models: bool, use_extra_vectors: bool,
                         use_losing_red_apples: bool, reset_cards_between_games: bool, print_in_terminal: bool) -> None:
        self.__change_players_between_games = change_players
        self.__cycle_starting_judges_between_games = cycle_starting_judges
        self.__reset_models_between_games = reset_models
        self.__use_extra_vectors = use_extra_vectors
        self.__use_losing_red_apples = use_losing_red_apples
        self.__reset_cards_between_games = reset_cards_between_games
        self.__print_in_terminal = print_in_terminal

    def new_game(self) -> None:
        """
        Start a new game of 'Apples to Apples' and reset the game state.
        Optionally, initialize new players.
        """
        message = "\nStarting new 'Apples to Apples' game."
        if self.__print_in_terminal:
            print(message)
        logging.info(message)

        # Initialize the GameState and add the game
        game_state = GameState()
        self.__game_log.add_game(game_state)

        # Print and log the game message
        message = f"\n------------- GAME {self.__game_log.get_current_game_number()} of {self.__game_log.total_games} -------------"
        print(message)
        logging.info(message)

        # Reset the game state
        message = "Resetting game state."
        if self.__print_in_terminal:
            print(message)
        logging.info(message)

        # Initialize the decks
        self.__initialize_decks()

        # Initialize the players for the first game
        if self.__game_log.get_current_game_number() == 1:
            self.__initialize_players()
        elif self.__game_log.get_current_game_number() > 1:
            # Prompt the user on whether to keep the same players, if applicable
            if self.__change_players_between_games:
                from game_driver import get_user_input_y_or_n
                keep_players = get_user_input_y_or_n("Do you want to keep the same players as last game? (y/n): ")
                if keep_players == "n":
                    self.__initialize_players()
                elif keep_players == "y":
                    self.__game_log.copy_players_to_new_game()
                    message = "Keeping the same players as last game."
                    if self.__print_in_terminal:
                        print(message)
                    logging.info(message)
            else:
                # Copy players over to the new game
                self.__game_log.copy_players_to_new_game()

            # Reset the player points and judge status
            message = "Resetting player points and judge status."
            if self.__print_in_terminal:
                print(message)
            logging.info(message)
            self.__reset_player_points_and_judge_status()

            # Reset the opponent models for the AI agents, if applicable
            if self.__reset_models_between_games:
                self.__reset_opponent_models()

            # Reset the red apples in hand for all players, if applicable
            if self.__training_mode and self.__reset_cards_between_games:
                # Reset the red apples in hand for all players
                for player in self.__game_log.get_game_players():
                    if isinstance(player, HumanAgent):
                        player.reset_red_apples()
                        player.draw_red_apples(self.embedding, self.__red_apples_deck, self.__game_log.max_cards_in_hand, self.__use_extra_vectors)


        # Start the game loop
        self.__game_loop()

    def __reset_player_points_and_judge_status(self) -> None:
        # TODO - check if need to skip for training mode
        # Reset the player points and judge status
        for player in self.__game_log.get_game_players():
            player.reset_points()
            player.set_judge_status(False)

    def __initialize_decks(self) -> None:
        # Print and log the initialization message
        message = "Initializing decks."
        if self.__print_in_terminal:
            print(message)
        logging.info(message)

        # Shuffle the decks
        self.__load_and_shuffle_deck(self.__green_apples_deck, "Green Apples", "./apples/green_apples.csv", self.__green_expansion_filename)
        self.__load_and_shuffle_deck(self.__red_apples_deck, "Red Apples", "./apples/red_apples.csv", self.__red_expansion_filename)

    def __load_and_shuffle_deck(self, deck: Deck, deck_name: str, base_file: str, expansion_file: str) -> None:
        # Load the base deck
        deck.load_deck(deck_name, base_file)
        message = f"Loaded {len(deck.get_apples())} {deck_name.lower()}."
        if self.__print_in_terminal:
            print(message)
        logging.info(message)

        # Load the expansion deck, if applicable
        if expansion_file:
            deck.load_deck(f"{deck_name} Expansion", expansion_file)
            message = f"Loaded {len(deck.get_apples())} {deck_name.lower()} from the expansion."
            if self.__print_in_terminal:
                print(message)
            logging.info(message)

        # Shuffle the deck
        deck.shuffle()

    def __generate_unique_agent_name(self, base_name: str) -> str:
        # Unpack the existing names (from the game log, not just current game players)
        existing_names = [agent.get_name() for agent in self.__game_log.all_game_players]

        # Generate a unique name
        i = 1
        while f"{base_name} {i}" in existing_names:
            i += 1
        return f"{base_name} {i}"

    def __initialize_players(self) -> None:
        # Print and log the initialization message
        message = "Initializing players."
        print(message)
        logging.info(message)

        # Display the number of players
        message = f"There are {self.__game_log.total_number_of_players} players per game."
        if self.__print_in_terminal:
            print("\n" + message)
        logging.info(message)

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
            new_agent = AIAgent(new_agent_name, model_type_class, pretrained_model_string, self.__use_extra_vectors, self.__use_losing_red_apples, True)

            # Create the human agent
            human_agent = HumanAgent("Human Agent", self.__print_in_terminal)

            # Have the human agent pick up max red apples in hand
            human_agent.draw_red_apples(self.embedding, self.__red_apples_deck, self.__game_log.max_cards_in_hand, self.__use_extra_vectors)

            # Add the human agent and AI agent to the current game and game log
            self.__game_log.add_player_to_current_game(human_agent) # NOTE - APPEND HUMAN AGENT FIRST!!!
            self.__game_log.add_player_to_current_game(new_agent)
            self.__game_log.all_game_players.append(human_agent)
            self.__game_log.all_game_players.append(new_agent)
            logging.info(f"Added new human player {human_agent.get_name()}.")
            logging.info(f"Added new AI agent {new_agent.get_name()}.")

        else:
            # Create the players when not in training mode
            for i in range(self.__game_log.total_number_of_players):
                # Prompt the user to select the player type
                message = f"\nWhat type is Agent {i + 1}?"
                print(message)
                logging.info(message)
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
                        if new_agent_name not in [agent.get_name() for agent in self.__game_log.get_game_players()]:
                            break
                    new_agent = HumanAgent(f"Human Agent - {new_agent_name}", self.__print_in_terminal)
                elif player_type == '2':
                    new_agent_name = self.__generate_unique_agent_name("Random Agent")
                    new_agent = RandomAgent(new_agent_name, self.__print_in_terminal)
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
                    new_agent = AIAgent(new_agent_name, ml_model_type_class, pretrained_archetype_string, self.__use_extra_vectors, self.__use_losing_red_apples,
                                        training_mode=False, print_in_terminal=self.__print_in_terminal)

                # Append the player object to the current game and game log
                self.__game_log.add_player_to_current_game(new_agent)
                self.__game_log.all_game_players.append(new_agent)
                logging.info(f"Added new player {self.__game_log.get_game_players()[i]}")

                # Have each player pick up max red apples in hand
                self.__game_log.get_game_players()[i].draw_red_apples(self.embedding, self.__red_apples_deck, self.__game_log.max_cards_in_hand, self.__use_extra_vectors)

        # Initialize the models for the AI agents
        for player in self.__game_log.get_game_players():
            if isinstance(player, AIAgent):
                player.initialize_models(self.embedding, self.__game_log.get_game_players())
                logging.info(f"Initialized models for {new_agent.get_name()}.")

        # Format the naming scheme when new players are initialized
        self.__game_log.format_naming_scheme()

    def __choose_starting_judge(self) -> Agent:
        # Clear the judge status for all players
        for i, player in enumerate(self.__game_log.get_game_players()):
            self.__game_log.get_game_players()[i].set_judge_status(False)

        if self.__training_mode:
            # Choose the ai agent as the starting judge
            for player in self.__game_log.get_game_players():
                if isinstance(player, AIAgent):
                    judge_index = self.__game_log.get_game_players().index(player)
                    break
        else:
            # If cycle starting judge is True, choose the starting judge automatically
            if self.__cycle_starting_judges_between_games:
                # Cycle through the judges to get the judge index
                judge_index = ((self.__game_log.get_current_game_number() - 1) % self.__game_log.total_number_of_players) # Subtract 1 since 0-based index and current_game starts at 1
            else: # If cycle starting judge is False, prompt the user to choose the starting judge
                if self.__change_players_between_games:
                    # Choose the starting judge
                    choice = input(f"\nPlease choose the starting judge (1-{self.__game_log.get_number_of_players()}): ")
                    logging.info(f"Please choose the starting judge (1-{self.__game_log.get_number_of_players()}): {choice}")

                    # Validate the user input
                    while not choice.isdigit() or int(choice) < 1 or int(choice) > self.__game_log.get_number_of_players():
                        choice = input(f"Invalid input. Please enter a number (1-{self.__game_log.get_number_of_players()}): ")
                        logging.error(f"Invalid input. Please enter a number (1-{self.__game_log.get_number_of_players()}): {choice}")

                    # Set the current judge index
                    judge_index = int(choice) - 1 # Subtract 1 since 0-based index
                else:
                    judge_index = 0

        return self.__game_log.get_game_players()[judge_index]

    def __determine_next_judge(self) -> Agent:
        # If it's the first round, choose the starting judge
        if self.__game_log.get_current_round() == 0:
            # Choose the starting judge and set the judge status
            next_judge = self.__choose_starting_judge()
            next_judge.set_judge_status(True)

            # Print and log the starting judge message
            message = f"{next_judge.get_name()} is the starting judge."
            if self.__print_in_terminal:
                print("\n" + message)
            logging.info(message)
        else:
            # Determine the next judge
            if self.__training_mode:
                next_judge = self.__game_log.get_current_judge()
            else:
                # Get the current judge and remove the judge status
                current_judge = self.__game_log.get_current_judge()
                current_judge.set_judge_status(False)

                # Get the next judge and set the judge status
                next_judge = self.__game_log.get_game_players()[ \
                            (self.__game_log.get_game_players().index(self.__game_log.get_current_judge()) + 1)
                            % self.__game_log.get_number_of_players()]
                next_judge.set_judge_status(True)

            # Print and log the next judge message
            if not self.__training_mode:
                message = f"{next_judge.get_name()} is the next judge."
                if self.__print_in_terminal:
                    print(message)
                logging.info(message)

        return next_judge

    def __new_round(self) -> None:
        # Determine the next judge
        next_judge = self.__determine_next_judge()

        # Initialize the RoundState and add the round
        round_state = RoundState(current_judge=next_judge)
        self.__game_log.add_round(round_state)

        # Print and log the round message
        message = f"\n===================" \
                        f"\nROUND {self.__game_log.get_current_round()}:" \
                        f"\n===================\n"
        if self.__print_in_terminal:
            print(message)
        logging.info(message)

        # Print and log the player points
        if not self.__training_mode:
            for player in self.__game_log.get_game_players():
                message = f"{player.get_name()}: {player.get_points()} points"
                if self.__print_in_terminal:
                    print(message)
                logging.info(message)

    def ___prompt_judge_draw_green_apple(self) -> None:
        # Extract the current judge
        current_judge: Agent = self.__game_log.get_current_judge()

        # Prompt the judge to draw a green apple
        message = f"\n{current_judge.get_name()}, please draw a green apple."
        if self.__print_in_terminal:
            print(message)
        logging.info(message)

        # Set the green card in play
        green_apple_dict: dict[Agent, GreenApple] = current_judge.draw_green_apple(self.embedding, self.__green_apples_deck, self.__use_extra_vectors)
        self.__game_log.set_green_apple_in_play(green_apple_dict)
        self.__game_log.set_chosen_green_apple(green_apple_dict)
        
    def __prompt_players_select_red_apples(self) -> None:
        # Prompt the players to select a red apple
        for player in self.__game_log.get_game_players():
            if player.get_judge_status():
                continue

            # Prompt the player to select a red apple
            if self.__training_mode:
                message = f"\n{player.get_name()}, please select a good red apple."
            else:
                message = f"\n{player.get_name()}, please select a red apple."
            if self.__print_in_terminal:
                print(message)
            logging.info(message)

            # Prompt the player to pick a red apple
            red_apple_dict: dict[Agent, RedApple] = player.choose_red_apple(
                self.__game_log.get_current_judge(),
                self.__game_log.get_apples_in_play().get_green_apple()
            )

            # Append the red apple to the list of red apples in play
            self.__game_log.add_red_apple_in_play(red_apple_dict)
            logging.info(f"Chosen red apple: {red_apple_dict[player]}")

            # Prompt the player to select a bad red apple, if applicable
            if self.__training_mode and self.__use_losing_red_apples:
                message = f"\nThe green apple is '{self.__game_log.get_apples_in_play().get_green_apple()}'."
                if self.__print_in_terminal:
                    print(message)
                logging.info(message)
                message = f"\n{player.get_name()}, please select a bad red apple."
                if self.__print_in_terminal:
                    print(message)
                logging.info(message)

                # Prompt the player to select a bad red apple
                bad_red_apple_dict: dict[Agent, RedApple] = player.choose_red_apple(
                    self.__game_log.get_current_judge(),
                    self.__game_log.get_apples_in_play().get_green_apple()
                )

                # Append the bad red apple to the list of red apples in play
                self.__game_log.add_red_apple_in_play(bad_red_apple_dict)
                logging.info(f"Chosen bad red apple: {bad_red_apple_dict[player]}")

            # Prompt the player to pick up a new red apple
            if len(player.get_red_apples()) < self.__game_log.max_cards_in_hand:
                player.draw_red_apples(self.embedding, self.__red_apples_deck, self.__game_log.max_cards_in_hand, self.__use_extra_vectors)

    def __determine_round_winner(self) -> None:
            # Extract the current judge and apples in play
            current_judge: Agent = self.__game_log.get_current_judge()
            apples_in_play: ApplesInPlay = self.__game_log.get_apples_in_play()

            # Prompt the judge to select the winning red apple
            if not self.__training_mode:
                message = f"\n{current_judge.get_name()}, please select the winning red apple."
                if self.__print_in_terminal:
                    print(message)
                logging.info(message)

            # Determine the winning red apple
            if self.__training_mode:
                winning_red_apple_dict: dict[Agent, RedApple] = apples_in_play.red_apples[0]
            else:
                winning_red_apple_dict: dict[Agent, RedApple] = current_judge.choose_winning_red_apple(apples_in_play)

            # Set the winning red apple
            self.__game_log.set_winning_red_apple(winning_red_apple_dict)

            # Print and log the winning red apple
            winning_red_apple: RedApple = winning_red_apple_dict[self.__game_log.get_chosen_apples().get_red_apple_winner()]
            if not self.__training_mode:
                message = f"{current_judge.get_name()} chose the winning red apple '{winning_red_apple}'."
                if self.__print_in_terminal:
                    print(message)
                logging.info(message)

            # Set the losing red apples
            for red_apple_dict in apples_in_play.red_apples:
                if red_apple_dict != winning_red_apple_dict:
                    self.__game_log.add_losing_red_apple(red_apple_dict)
            logging.info(f"Losing Red Apples: {self.__game_log.get_chosen_apples().losing_red_apples}")

            # Extract the round winner
            round_winner: Agent = self.__game_log.get_chosen_apples().get_red_apple_winner()
            logging.info(f"Round Winner: {round_winner}")

            # Verify the round winner is in the list of players
            if round_winner not in self.__game_log.get_game_players():
                logging.error(f"Round winner {round_winner} not in list of players.")
                raise ValueError(f"Round winner {round_winner} not in list of players.")

            # Print and log the round winner
            if not self.__training_mode:
                message = f"***{round_winner.get_name()} has won the round!***"
                if self.__print_in_terminal:
                    print(f"\n{message}")
                logging.info(message)

            # Set the round winner and award the additional point
            self.__game_log.set_round_winner(round_winner)

    def __train_ai_agents(self) -> None:
        # Train all AI agents (if applicable)
        for player in self.__game_log.get_game_players():
            # Train the AI agent
            if isinstance(player, AIAgent):
                # In training mode, train the AI "player" on the human agent
                if self.__training_mode:
                    for agent in self.__game_log.get_game_players():
                        if isinstance(agent, HumanAgent):
                            player.train_self_judge_model(
                                self.__game_log.get_chosen_apples()
                            )

                            # TODO - Refactor to get vector logging
                            # # Get the opponent judge model
                            # opponent_judge_model: Model | None = player.get_opponent_model(agent)

                            # # Check that the judge model is not None
                            # if opponent_judge_model is None:
                            #     logging.error("The opponent judge model is None.")
                            #     raise ValueError("The opponent judge model is None.")

                            # current_slope, current_bias = opponent_judge_model.get_current_slope_and_bias_vectors()
                            # log_vectors(self.__game_log.get_current_game_state(), player, current_slope, current_bias, True)
                else:
                    # If not in training mode, train only if the player is not the current judge
                    if player != self.__game_log.get_current_judge():
                        player.train_opponent_judge_model(
                            self.__game_log.get_current_judge(),
                            self.__game_log.get_chosen_apples(),
                        )

                        # TODO - Refactor to get vector logging
                        # # Get the opponent judge model
                        # opponent_judge_model: Model | None = player.get_opponent_model(self.__game_log.get_current_round_judge())

                        # # Check that the judge model is not None
                        # if opponent_judge_model is None:
                        #     logging.error("The opponent judge model is None.")
                        #     raise ValueError("The opponent judge model is None.")

                        # current_slope, current_bias = opponent_judge_model.get_current_slope_and_bias_vectors()
                        # log_vectors(self.__game_log.get_current_game_state(), player, current_slope, current_bias, True)

    def __reset_opponent_models(self) -> None:
        # TODO - check if need to skip for training mode
        # Print and log the message
        message = "Resetting opponent models for all AI agents."
        if self.__print_in_terminal:
            print(message)
        logging.info(message)

        # Reset the opponent models for all AI agents
        for player in self.__game_log.get_game_players():
            if isinstance(player, AIAgent):
                player.reset_opponent_models()

    def __is_game_over(self) -> None:
        for player in self.__game_log.get_game_players():
            if player.get_points() >= self.__game_log.points_to_win:
                self.__game_log.set_game_winner(player)

    def __game_loop(self) -> None:
        # Start the game loop
        while self.__game_log.get_game_winner() is None:
            # Increment the round counter and print the round messages
            self.__new_round()

            # Prompt the judge to draw a green apple
            self.___prompt_judge_draw_green_apple()

            # Prompt the players to select red apples
            self.__prompt_players_select_red_apples()

            # Determine the winning red apple and losing red apples
            self.__determine_round_winner()

            # Log the round winner
            if not self.__training_mode:
                log_round_winner(self.__game_log.naming_scheme, self.__game_log.get_current_game_state(), True)

            # Check if the game is over
            self.__is_game_over()

            # Log the gameplay or training results
            if self.__training_mode:
                log_training(self.__game_log.naming_scheme, self.__game_log.get_current_game_state(), True)
            else:
                log_game_state(self.__game_log.naming_scheme, self.__game_log, True)

            # Train the AI agents
            self.__train_ai_agents()

            # Add the chosen apples to the discard pile
            self.__game_log.discard_chosen_apples(self.__game_log.get_chosen_apples())

            # Print the winner message
            current_game_winner: Agent | None = self.__game_log.get_game_winner()
            if current_game_winner is not None:
                # Prepare the winner message
                winner_text = f"# {current_game_winner.get_name()} has won the game! #"
                border = '#' * len(winner_text)
                message = f"\n{border}\n{winner_text}\n{border}\n"

                # Print and log the winner message
                message = message
                if self.__print_in_terminal:
                    print(message)
                logging.info(message)

                # Log the winner if not in training mode
                if not self.__training_mode:
                    log_game_winner(self.__game_log.naming_scheme, self.__game_log.get_current_game_state(), True)


if __name__ == "__main__":
    pass
