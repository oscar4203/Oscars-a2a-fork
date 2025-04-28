# Description: AI agent logic for the 'Apples to Apples' game.

# Standard Libraries
from typing import Optional, TYPE_CHECKING
import logging
import random
import numpy as np

# Third-party Libraries

# Local Modules
from src.embeddings.embeddings import Embedding
from src.agent_model.model import Model, LRModel, NNModel
from src.apples.apples import GreenApple, RedApple, Deck
from src.data_classes.data_classes import ApplesInPlay, ChosenApples, PathsConfig

# Type Checking to prevent circular imports
if TYPE_CHECKING:
    from src.interface.input.input_handler import InputHandler


class Agent:
    """Base class for the agents in the 'Apples to Apples' game"""
    def __init__(self, name: str) -> None:
        self._name: str = name
        self._points: int = 0
        self._judge_status: bool = False
        self._green_apple: GreenApple | None = None
        self._red_apples: list[RedApple] = []
        self._input_handler: Optional["InputHandler"] = None

    def __str__(self) -> str:
        # Retrieve the green apple
        if self._green_apple is not None:
            green_apple = self._green_apple.get_adjective()
        else:
            green_apple = None

        # Retrieve the red apples
        red_apples = [red_apple.get_noun() for red_apple in self._red_apples]

        return f"{self.__class__.__name__}(name={self._name}, points={self._points}, judge_status={self._judge_status}, " \
            f"green_apple={green_apple}, red_apples={red_apples})"

    def __repr__(self) -> str:
        """
        Return the string representation of the agent.
        Returns a more detailed string representation of the agent,
        and calls the __repr__ method for each apple, so they are more detailed too.
        """
        # Retrieve the green apple
        if self._green_apple is not None:
            green_apple = self._green_apple.get_adjective()
        else:
            green_apple = None

        # Retrieve the red apples
        red_apples = [red_apple.get_noun() for red_apple in self._red_apples]

        return f"Agent(name={self._name}, points={self._points}, judge_status={self._judge_status}, " \
            f"green_apple={green_apple}, red_apples={red_apples})"

    def set_input_handler(self, input_handler: "InputHandler") -> None:
        """Set the input handler for this agent."""
        self._input_handler = input_handler

    def is_human(self) -> bool:
        """
        Check if the agent is a human player.
        """
        return isinstance(self, HumanAgent)

    def get_name(self) -> str:
        return self._name

    def get_points(self) -> int:
        return self._points

    def get_judge_status(self) -> bool:
        return self._judge_status

    def get_green_apple(self) -> GreenApple | None:
        return self._green_apple

    def get_red_apples(self) -> list[RedApple]:
        return self._red_apples

    def set_points(self, points: int) -> None:
        self._points = points

    def set_judge_status(self, judge: bool) -> None:
        self._judge_status = judge

    def reset_points(self) -> None:
        """
        Reset the agent's points to zero.
        """
        self._points = 0

    def reset_red_apples(self) -> None:
        """
        Reset the agent's red apples to an empty list.
        """
        self._red_apples = []

    def draw_green_apple(self, embedding: Embedding, green_apple_deck: Deck, extra_vectors: bool) -> dict["Agent", GreenApple]:
        """
        Draw a green apple from the deck (when the agent is the judge).
        The vectors are set as soon as the new green apple is drawn.
        """
        # Check if the Agent is a judge
        if self._judge_status:
            # Draw a new green apple
            new_green_apple = green_apple_deck.draw_apple()
            if not isinstance(new_green_apple, GreenApple):
                raise TypeError("Expected a GreenApple, but got a different type")

            # Set the green apple adjective vector
            new_green_apple.set_adjective_vector(embedding)

            # Set the green apple synonyms vector, if applicable
            if extra_vectors:
                new_green_apple.set_synonyms_vector(embedding)

            # Assign the green apple to the agent's hand
            self._green_apple = new_green_apple
        else:
            logging.error(f"{self._name} is not the judge.")
            raise ValueError(f"{self._name} is not the judge.")

        # Initialize the green apple dict
        green_apple_dict: dict["Agent", GreenApple] = {self: self._green_apple}

        return green_apple_dict

    def draw_red_apples(self, embedding: Embedding, red_apple_deck: Deck, cards_in_hand: int, extra_vectors: bool) -> int:
        """
        Draw red apples from the deck, ensuring the agent has enough red apples.
        The vectors are set as soon as the new red apples are drawn.

        Returns:
            int: Number of cards drawn (can be used by UI layer to display appropriate message)
        """
        # Calculate the number of red apples to pick up
        diff = cards_in_hand - len(self._red_apples)
        if diff > 0:
            for _ in range(diff):
                # Draw a new red apple
                new_red_apple = red_apple_deck.draw_apple()
                if not isinstance(new_red_apple, RedApple):
                    raise TypeError("Expected a RedApple, but got a different type")

                # Set the red apple noun vector
                new_red_apple.set_noun_vector(embedding)

                # Set the red apple description vector, if applicable
                if extra_vectors:
                    new_red_apple.set_description_vector(embedding)

                # Append the red apple to the agent's hand
                self._red_apples.append(new_red_apple)

            # Log the operation but don't print it directly
            if diff == 1:
                logging.info(f"{self._name} picked up 1 red apple.")
            else:
                logging.info(f"{self._name} picked up {diff} red apples.")

            return diff  # Return the number of cards drawn
        else:
            logging.info(f"{self._name} already has enough red apples.")
            return 0  # No cards drawn

    def choose_red_apple(self, current_judge: "Agent", green_apple: GreenApple) -> dict["Agent", RedApple]:
        """
        Choose a red apple from the agent's hand to play (when the agent is a regular player).
        """
        raise NotImplementedError("Subclass must implement the 'choose_red_apple' method")

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict["Agent", RedApple]:
        """
        Choose the winning red apple from the red apples submitted by the other agents (when the agent is the judge).
        """
        raise NotImplementedError("Subclass must implement the 'choose_winning_red_apple' method")


class HumanAgent(Agent):
    """Human agent for the 'Apples to Apples' game."""
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def choose_red_apple(self, current_judge: Agent, green_apple: GreenApple) -> dict["Agent", RedApple]:
        # Check if the agent is a judge
        if self._judge_status:
            logging.error(f"{self._name} is the judge.")
            raise ValueError(f"{self._name} is the judge.")

        # Check if the input handler is set
        if self._input_handler is None:
            logging.error("Input handler is not set.")
            raise ValueError("Input handler is not set.")

        # Convert the input to an index
        red_apple_index = self._input_handler.prompt_human_agent_choose_red_apple(self, self.get_red_apples(), green_apple)
        chosen_red_apple = self._red_apples.pop(red_apple_index)
        chosen_red_apple_dict: dict["Agent", RedApple] = {self: chosen_red_apple}

        return chosen_red_apple_dict

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict[Agent, RedApple]:
        """
        Prompt the human judge to choose the winning red apple using the assigned input handler.
        """
        # Check if the agent is a judge
        if not self._judge_status:
            logging.error(f"{self._name} is not the judge.")
            raise ValueError(f"{self._name} is not the judge.")

        # Check if the input handler is set
        if self._input_handler is None:
            logging.error("Input handler is not set for HumanAgent.")
            raise ValueError("Input handler is not set.")

        # Get the green apple from the apples_in_play object
        green_apple = apples_in_play.get_green_apple()
        if not green_apple:
             logging.error("Green apple not found in apples_in_play.")
             # Handle error appropriately, maybe raise or return a default
             raise ValueError("Green apple missing for judge selection.")

        # Use the input handler to prompt the judge
        # The prompt_judge_select_winner method returns the winning Agent
        winning_agent = self._input_handler.prompt_judge_select_winner(
            self,
            apples_in_play.get_red_apples_dicts(),
            green_apple
        )

        # Get the corresponding winning red apple from the submissions
        winning_red_apple = apples_in_play.get_red_apple_by_agent(winning_agent)
        if winning_red_apple is None:
            logging.error(f"Could not find submitted red apple for winning agent {winning_agent.get_name()}.")
            # Handle error appropriately
            raise ValueError("Winning red apple not found in submissions.")

        # Return the dictionary {winning_agent: winning_red_apple}
        return {winning_agent: winning_red_apple}


class RandomAgent(Agent):
    """Random agent for the 'Apples to Apples' game."""
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def choose_red_apple(self, current_judge: Agent, green_apple: GreenApple) -> dict["Agent", RedApple]:
        # Check if the agent is a judge
        if self._judge_status:
            logging.error(f"{self._name} is the judge.")
            raise ValueError(f"{self._name} is the judge.")

        # Choose a random red apple
        index = random.choice(range(len(self._red_apples)))
        chosen_red_apple = self._red_apples.pop(index)
        chosen_red_apple_dict: dict["Agent", RedApple] = {self: chosen_red_apple}

        return chosen_red_apple_dict

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict[Agent, RedApple]:
        # Check if the agent is a judge
        if not self._judge_status:
            logging.error(f"{self._name} is not the judge.")
            raise ValueError(f"{self._name} is not the judge.")

        # Choose a random winning red apple
        winning_red_apple = random.choice(apples_in_play.red_apples)

        return winning_red_apple


class AIAgent(Agent):
    """
    AI agent for the 'Apples to Apples' game using Word2Vec and Linear Regression.
    """
    def __init__(self,
                 name: str,
                 ml_model_type: LRModel | NNModel,
                 paths_config: PathsConfig,
                 pretrained_archetype: str,
                 use_extra_vectors: bool = False,
                 training_mode: bool = False
                ) -> None:
        super().__init__(name)
        self.__ml_model_type: LRModel | NNModel = ml_model_type
        self.__pretrained_archetype: str = pretrained_archetype
        self.__use_extra_vectors: bool = use_extra_vectors
        self.__training_mode: bool = training_mode
        self._paths_config = paths_config

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, points={self._points}, judge_status={self._judge_status}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, points={self._points}, judge_status={self._judge_status}, " \
            f"ml_model_type={self.__ml_model_type}, pretrained_archetype={self.__pretrained_archetype}, " \
            f"use_extra_vectors={self.__use_extra_vectors}, training_mode={self.__training_mode}, " \
            f"green_apple={self._green_apple}, red_apples={self._red_apples})"

    def get_opponent_model(self, agent_as_key: Agent) -> Model | None:
        if self.__opponent_ml_models is None:
            logging.error("Opponent ML Models have not been initialized.")
            raise ValueError("Opponent ML Models have not been initialized.")
        else:
            return self.__opponent_ml_models.get(agent_as_key)

    def initialize_models(self, embedding: Embedding, paths_config: PathsConfig, all_players: list[Agent]) -> None:
        """
        Initialize the Linear Regression and/or Neural Network models for the AI agent.
        """
        # Initialize the keyed vectors
        self.__embedding: Embedding = embedding
        # self.__vectors = None # Vectors loaded via custom loader # TODO - Implement custom loader

        # Determine and initialize the opponents
        self.__opponents: list[Agent] = [agent for agent in all_players if agent != self]
        logging.debug(f"Opponents: {[agent.get_name() for agent in self.__opponents]}")

        # Initialize the self and opponent ml models
        if self.__ml_model_type is LRModel:
            self.__self_ml_model: Model = LRModel(self, self, self.__embedding.vector_size, paths_config, self.__pretrained_archetype, self.__use_extra_vectors, self.__training_mode)
            self.__opponent_ml_models: dict[Agent, Model] = {agent: LRModel(self, agent, self.__embedding.vector_size, paths_config, self.__pretrained_archetype, self.__use_extra_vectors, self.__training_mode) for agent in self.__opponents}
        elif self.__ml_model_type is NNModel:
            self.__self_ml_model: Model = NNModel(self, self, self.__embedding.vector_size, paths_config, self.__pretrained_archetype, self.__use_extra_vectors, self.__training_mode)
            self.__opponent_ml_models: dict[Agent, Model] = {agent: NNModel(self, agent, self.__embedding.vector_size, paths_config, self.__pretrained_archetype, self.__use_extra_vectors, self.__training_mode) for agent in self.__opponents}
        logging.debug(f"Self Model initialized - self_ml_model: {self.__self_ml_model}")
        logging.debug(f"Opponent Models initialized - opponent_ml_models: {self.__opponent_ml_models}")

    def get_self_slope_and_bias_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the slope and bias vectors lists for the AI self model.
        """
        return self.__self_ml_model.get_slope_and_bias_vectors()

    def get_opponent_slope_and_bias_vectors(self, opponent: Agent) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the slope and bias vectors lists for the AI opponent model.
        """
        return self.__opponent_ml_models[opponent].get_slope_and_bias_vectors()

    def reset_opponent_models(self) -> None:
        """
        Reset the opponent models to the default archetype.
        """
        # Reset the opponent models
        for opponent in self.__opponents:
            agent_model: Model = self.__opponent_ml_models[opponent]
            agent_model.reset_model()
            message = f"Reset {opponent.get_name()}'s model."
            logging.info(message)

    def choose_red_apple(self, current_judge: Agent, green_apple: GreenApple) -> dict["Agent", RedApple]:
        # Check if the agent is a judge
        if self._judge_status:
            logging.error(f"{self._name} is the judge.")
            raise ValueError(f"{self._name} is the judge.")

        # Run the AI model to choose a red apple based on current judge
        chosen_red_apple: RedApple = self.__opponent_ml_models[current_judge].choose_red_apple(green_apple, self._red_apples)
        self._red_apples.remove(chosen_red_apple)
        chosen_red_apple_dict: dict["Agent", RedApple] = {self: chosen_red_apple}

        return chosen_red_apple_dict

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict[Agent, RedApple]:
        # Choose a winning red apple
        winning_red_apple_dict: dict[Agent, RedApple] = self.__self_ml_model.choose_winning_red_apple(apples_in_play)

        return winning_red_apple_dict

    def train_self_judge_model(self, chosen_apples: ChosenApples) -> None:
        """
        Train the AI self model for the current judge, given the new green and red apples.
        """
        # Train the AI models with the new green apple, red apple, and judge
        self.__self_ml_model.train_model(chosen_apples)

        # Extract the apples for logging
        green_apple: GreenApple = chosen_apples.get_green_apple()
        winning_red_apple: RedApple = chosen_apples.get_winning_red_apple()
        losing_red_apples: list[RedApple] = chosen_apples.get_losing_red_apples()

        # Configure the logging message
        message = f"Trained {self.get_name()}'s self model. Green apple '{green_apple}'. Winning red apple '{winning_red_apple}'."
        if losing_red_apples:
            message += f" Losing red apples: {losing_red_apples}."
        logging.debug(message)

    def train_opponent_judge_model(self, current_judge: Agent, chosen_apples: ChosenApples) -> None:
        """
        Train the AI opponent model for the current judge, given the new green and red apples.
        """
        # Check if the agent is a judge
        for agent in self.__opponents:
            if agent is current_judge:
                # Train the AI models with the new green apple, red apple, and judge
                self.__opponent_ml_models[agent].train_model(chosen_apples)

                # Extract the apples for logging
                green_apple: GreenApple = chosen_apples.get_green_apple()
                winning_red_apple: RedApple = chosen_apples.get_winning_red_apple()
                losing_red_apples: list[RedApple] = chosen_apples.get_losing_red_apples()

                # Configure the logging message
                message = f"Trained {self.get_name()}'s opponent model '{agent.get_name()}'. Green apple '{green_apple}'. Winning red apple '{winning_red_apple}'."
                if losing_red_apples:
                    message += f" Losing red apples: {losing_red_apples}."
                logging.debug(message)

    # def initialize_models(self, embedding: Embedding, players: list[Agent], paths_config: PathsConfig) -> None:
    #     """Initialize opponent models for all other players."""
    #     self._opponent_models = {} # Clear existing models if any
    #     for opponent in players:
    #         if opponent != self: # Don't create a model for self here
    #             # Determine archetype/type for opponent model (you might need more sophisticated logic here)
    #             opponent_archetype = "Literalist" # Example: Default or determine based on opponent type/name
    #             opponent_model_type = "1" # Example: Default to LR
    #             self._add_opponent_model(opponent, opponent_archetype, opponent_model_type, paths_config) # Pass paths_config

    # def _create_model(self, judge_agent: "Agent", archetype: str, model_type_key: str) -> Model:
    #     """Creates a model instance based on the type key."""
    #     model_class = model_type_mapping.get(model_type_key)
    #     if not model_class:
    #         raise ValueError(f"Invalid model type key: {model_type_key}")

    #     return model_class(
    #         self_agent=self,
    #         judge_to_model=judge_agent,
    #         vector_size=self._vector_size,
    #         pretrained_archetype=archetype,
    #         use_extra_vectors=self._use_extra_vectors,
    #         training_mode=self._training_mode,
    #         # --- Pass PathsConfig ---
    #         paths_config=self._paths_config
    #     )

    # def _add_opponent_model(self, opponent_agent: "Agent", model_archetype: str, model_type: str, paths_config: PathsConfig) -> None:
    #      """Adds or updates a model for a specific opponent."""
    #      if opponent_agent.get_name() not in self._opponent_models:
    #          # _create_model will use self._paths_config stored during AIAgent init
    #          self._opponent_models[opponent_agent.get_name()] = self._create_model(
    #              judge_agent=opponent_agent,
    #              archetype=model_archetype,
    #              model_type_key=model_type
    #          )
    #          logging.info(f"Agent '{self.get_name()}' created model for opponent '{opponent_agent.get_name()}'.")


if __name__ == "__main__":
    pass
