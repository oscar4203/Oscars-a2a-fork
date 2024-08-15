# Description: AI agent logic for the 'Apples to Apples' game.

# Standard Libraries
import logging
import random
import numpy as np

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules
from source.apples import GreenApple, RedApple, Deck
from source.data_classes import ApplesInPlay, ChosenApples

class Agent:
    """
    Base class for the agents in the 'Apples to Apples' game
    """
    def __init__(self, name: str) -> None:
        self._name: str = name
        self._points: int = 0
        self._judge_status: bool = False
        self._green_apple: GreenApple | None = None
        self._red_apples: list[RedApple] = []

    def __str__(self) -> str:
        # Retrieve the green apple
        if self._green_apple is not None:
            green_apple = self._green_apple.get_adjective()
        else:
            green_apple = None

        # Retrieve the red apples
        red_apples = [red_apple.get_noun() for red_apple in self._red_apples]

        return f"Agent(name={self._name}, points={self._points}, judge_status={self._judge_status}, " \
            f"green_apple={green_apple}, red_apples={red_apples})"

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

    def draw_green_apple(self, keyed_vectors: KeyedVectors, green_apple_deck: Deck, extra_vectors: bool) -> GreenApple:
        """
        Draw a green card from the deck (when the agent is the judge).
        The vectors are set as soon as the new green apple is drawn.
        """
        # Check if the Agent is a judge
        if self._judge_status:
            # Draw a new green card
            new_green_apple = green_apple_deck.draw_apple()
            if not isinstance(new_green_apple, GreenApple):
                raise TypeError("Expected a GreenApple, but got a different type")

            # Set the green apple adjective vector
            new_green_apple.set_adjective_vector(keyed_vectors)

            # Set the green apple synonyms vector, if applicable
            if extra_vectors:
                new_green_apple.set_synonyms_vector(keyed_vectors)

            # Assign the green apple to the agent's hand
            self._green_apple = new_green_apple
        else:
            logging.error(f"{self._name} is the judge.")
            raise ValueError(f"{self._name} is the judge.")

        # Display the green card drawn
        message = f"{self._name} drew the green card '{self._green_apple}'."
        print(message)
        logging.info(message)

        return self._green_apple

    def draw_red_apples(self, keyed_vectors: KeyedVectors, red_apple_deck: Deck, cards_in_hand: int, extra_vectors: bool) -> Deck | None:
        """
        Draw red apples from the deck, ensuring the agent has enough red apples.
        The vectors are set as soon as the new red apples are drawn.
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
                new_red_apple.set_noun_vector(keyed_vectors)

                # Set the red apple description vector, if applicable
                if extra_vectors:
                    new_red_apple.set_description_vector(keyed_vectors)

                # Append the red apple to the agent's hand
                self._red_apples.append(new_red_apple)
            if diff == 1:
                message = f"{self._name} picked up 1 red apple."
                print(message)
                logging.info(message)
            else:
                message = f"{self._name} picked up {diff} red apples."
                print(message)
                logging.info(message)
        else:
            message = f"{self._name} cannot pick up any more red apples. Agent already has enough red apples."
            print(message)
            logging.info(message)

    def choose_red_apple(self, current_judge: "Agent", green_apple: GreenApple) -> RedApple:
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
    """
    Human agent for the 'Apples to Apples' game.
    """
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def choose_red_apple(self, current_judge: Agent, green_apple: GreenApple) -> RedApple:
        # Check if the agent is a judge
        if self._judge_status:
            logging.error(f"{self._name} is the judge.")
            raise ValueError(f"{self._name} is the judge.")

        # Choose a red apple
        red_apple: RedApple | None = None

        # Display the red apples in the agent's hand
        print(f"{self._name}'s red apples:")
        for i, red_apple in enumerate(self._red_apples):
            print(f"{i + 1}. {red_apple}")

        # Prompt the agent to choose a red apple
        red_apple_len = len(self._red_apples)
        red_apple_index = input(f"Choose a red apple (1 - {red_apple_len}): ")

        # Validate the input
        while not red_apple_index.isdigit() or int(red_apple_index) not in range(1, red_apple_len + 1):
            print(f"Invalid input. Please choose a valid red apple (1 - {red_apple_len}).")
            red_apple_index = input("Choose a red apple: ")

        # Convert the input to an index
        red_apple_index = int(red_apple_index) - 1

        # Remove the red apple from the agent's hand
        red_apple = self._red_apples.pop(red_apple_index)

        # Display the red apple chosen
        print(f"{self._name} chose a red apple.")
        logging.info(f"{self._name} chose the red apple '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict[Agent, RedApple]:
        # Check if the agent is a judge
        if not self._judge_status:
            logging.error(f"{self._name} is not the judge.")
            raise ValueError(f"{self._name} is not the judge.")

        # Display the red apples submitted by the other agents
        print("Red cards submitted by the other agents:")
        for i, red_apple in enumerate(apples_in_play.red_apples):
            print(f"{i + 1}. {red_apple[list(red_apple.keys())[0]]}")

        # Prompt the agent to choose a red apple
        red_apple_len = len(apples_in_play.red_apples)
        red_apple_index = input(f"Choose a winning red apple (1 - {red_apple_len}): ")

        # Validate the input
        while not red_apple_index.isdigit() or int(red_apple_index) not in range(1, red_apple_len + 1):
            print(f"Invalid input. Please choose a valid red apple (1 - {red_apple_len}).")
            red_apple_index = input("Choose a winning red apple: ")

        # Convert the input to an index
        red_apple_index = int(red_apple_index) - 1

        # Remove the red apple from the agent's hand
        # TODO - verify that the pop is doing what it is supposed to do
        winning_red_apple = apples_in_play.red_apples.pop(red_apple_index)

        return winning_red_apple


class RandomAgent(Agent):
    """
    Random agent for the 'Apples to Apples' game.
    """
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def choose_red_apple(self, current_judge: Agent, green_apple: GreenApple) -> RedApple:
        # Check if the agent is a judge
        if self._judge_status:
            logging.error(f"{self._name} is the judge.")
            raise ValueError(f"{self._name} is the judge.")

        # Choose a random red apple
        red_apple = self._red_apples.pop(random.choice(range(len(self._red_apples))))

        # Display the red apple chosen
        print(f"{self._name} chose a red apple.")
        logging.info(f"{self._name} chose the red apple '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict[Agent, RedApple]:
        # Check if the agent is a judge
        if not self._judge_status:
            logging.error(f"{self._name} is not the judge.")
            raise ValueError(f"{self._name} is not the judge.")

        # Choose a random winning red apple
        winning_red_apple = random.choice(apples_in_play.red_apples)

        return winning_red_apple

# Import the "Model" class from local library here to avoid circular importing
from source.model import Model, LRModel, NNModel

class AIAgent(Agent):
    """
    AI agent for the 'Apples to Apples' game using Word2Vec and Linear Regression.
    """
    def __init__(self, name: str, ml_model_type: LRModel | NNModel, pretrained_archetype: str, use_extra_vectors: bool = False, use_losing_red_apples : bool = False, training_mode: bool = False) -> None:
        super().__init__(name)
        self.__ml_model_type: LRModel | NNModel = ml_model_type
        self.__pretrained_archetype: str = pretrained_archetype
        self.__use_extra_vectors: bool = use_extra_vectors
        self.__use_losing_red_apples: bool = use_losing_red_apples
        self.__training_mode: bool = training_mode

    def get_opponent_model(self, agent_as_key: Agent) -> Model | None:
        if self.__opponent_ml_models is None:
            logging.error("Opponent ML Models have not been initialized.")
            raise ValueError("Opponent ML Models have not been initialized.")
        else:
            return self.__opponent_ml_models.get(agent_as_key)

    def initialize_models(self, keyed_vectors: KeyedVectors, all_players: list[Agent]) -> None:
        """
        Initialize the Linear Regression and/or Neural Network models for the AI agent.
        """
        # Initialize the keyed vectors
        self.__keyed_vectors: KeyedVectors = keyed_vectors
        # self.__vectors = None # Vectors loaded via custom loader # TODO - Implement custom loader

        # Determine and initialize the opponents
        self.__opponents: list[Agent] = [agent for agent in all_players if agent != self]
        logging.debug(f"Opponents: {[agent.get_name() for agent in self.__opponents]}")

        # Initialize the self and opponent ml models
        if self.__ml_model_type is LRModel:
            self.__self_ml_model: Model = LRModel(self, self.__keyed_vectors.vector_size, self.__pretrained_archetype, self.__use_extra_vectors, self.__use_losing_red_apples, self.__training_mode)
            self.__opponent_ml_models: dict[Agent, Model] = {agent: LRModel(agent, self.__keyed_vectors.vector_size, self.__pretrained_archetype, self.__use_extra_vectors, self.__use_losing_red_apples, self.__training_mode) for agent in self.__opponents}
        elif self.__ml_model_type is NNModel:
            self.__self_ml_model: Model = NNModel(self, self.__keyed_vectors.vector_size, self.__pretrained_archetype, self.__use_extra_vectors, self.__use_losing_red_apples, self.__training_mode)
            self.__opponent_ml_models: dict[Agent, Model] = {agent: NNModel(agent, self.__keyed_vectors.vector_size, self.__pretrained_archetype, self.__use_extra_vectors, self.__use_losing_red_apples, self.__training_mode) for agent in self.__opponents}
        logging.debug(f"Self Model initialized - self_ml_model: {self.__self_ml_model}")
        logging.debug(f"Opponent Models initialized - opponent_ml_models: {self.__opponent_ml_models}")

    def get_current_slope_and_bias_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current slope and bias vectors for the AI self model.
        """
        return self.__self_ml_model.get_current_slope_and_bias_vectors()

    def reset_opponent_models(self) -> None:
        """
        Reset the opponent models to the default archetype.
        """
        # Reset the opponent models
        for opponent in self.__opponents:
            agent_model: Model = self.__opponent_ml_models[opponent]
            agent_model.reset_model()
            message = f"Reset {opponent.get_name()}'s model."
            print(message)
            logging.info(message)

    def choose_red_apple(self, current_judge: Agent, green_apple: GreenApple) -> RedApple:
        # Check if the agent is a judge
        if self._judge_status:
            logging.error(f"{self._name} is the judge.")
            raise ValueError(f"{self._name} is the judge.")

        # Run the AI model to choose a red apple based on current judge
        red_apple: RedApple = self.__opponent_ml_models[current_judge].choose_red_apple(green_apple, self._red_apples)
        self._red_apples.remove(red_apple)

        # Display the red apple chosen
        print(f"{self._name} chose a red apple.")
        logging.info(f"{self._name} chose the red apple '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict[Agent, RedApple]:
        # Choose a winning red apple
        winning_red_apple_dict: dict[Agent, RedApple] = self.__self_ml_model.choose_winning_red_apple(apples_in_play)

        return winning_red_apple_dict

    def train_self_judge_model(self, chosen_apples: ChosenApples) -> None:
        """
        Train the AI self model for the current judge, given the new green and red apples.
        """
        # Train the AI models with the new green card, red apple, and judge
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
                # Train the AI models with the new green card, red apple, and judge
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


if __name__ == "__main__":
    pass
