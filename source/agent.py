# Description: AI agent logic for the 'Apples to Apples' game.

# Standard Libraries
import logging
import random

# Third-party Libraries
from gensim.models import KeyedVectors

# Local Modules
from source.apples import GreenApple, RedApple, Deck


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

    def __repr__(self) -> str:
        return self.__str__()

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

    def draw_red_apples(self, red_apple_deck: Deck, cards_in_hand: int) -> Deck | None:
        """
        Draw red cards from the deck, ensuring the agent has enough red cards.
        """
        # Calculate the number of red cards to pick up
        diff = cards_in_hand - len(self._red_apples)
        if diff > 0:
            for _ in range(diff):
                # Draw a red card
                new_red_apple = red_apple_deck.draw_apple()
                if not isinstance(new_red_apple, RedApple):
                    raise TypeError("Expected a RedApple, but got a different type")
                self._red_apples.append(new_red_apple)
            if diff == 1:
                print(f"{self._name} picked up 1 red card.")
                logging.info(f"{self._name} picked up 1 red card.")
            else:
                print(f"{self._name} picked up {diff} red cards.")
                logging.info(f"{self._name} picked up {diff} red cards.")
        else:
            print(f"{self._name} cannot pick up any more red cards. Agent already has enough red cards")
            logging.info(f"{self._name} cannot pick up the red card. Agent already has enough red cards")

    def draw_green_apple(self, green_apple_deck: Deck) -> GreenApple:
        """
        Draw a green card from the deck (when the agent is the judge).
        """
        # Check if the Agent is a judge
        if self._judge_status:
            # Draw a green card
            new_green_apple = green_apple_deck.draw_apple()
            if not isinstance(new_green_apple, GreenApple):
                raise TypeError("Expected a GreenApple, but got a different type")
            self._green_apple = new_green_apple
        else:
            logging.error(f"{self._name} is the judge.")
            raise ValueError(f"{self._name} is the judge.")

        # Display the green card drawn
        print(f"{self._name} drew the green card '{self._green_apple}'.")
        logging.info(f"{self._name} drew the green card '{self._green_apple}'.")

        return self._green_apple

    def choose_red_apple(self, current_judge: "Agent", green_apple: GreenApple) -> RedApple: # Define the type of current_judge as a string
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        """
        raise NotImplementedError("Subclass must implement the 'choose_red_apple' method")

    def choose_winning_red_apple(self, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
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

        # Choose a red card
        red_apple: RedApple | None = None

        # Display the red cards in the agent's hand
        print(f"{self._name}'s red cards:")
        for i, red_apple in enumerate(self._red_apples):
            print(f"{i + 1}. {red_apple}")

        # Prompt the agent to choose a red card
        red_apple_len = len(self._red_apples)
        red_apple_index = input(f"Choose a red card (1 - {red_apple_len}): ")

        # Validate the input
        while not red_apple_index.isdigit() or int(red_apple_index) not in range(1, red_apple_len + 1):
            print(f"Invalid input. Please choose a valid red card (1 - {red_apple_len}).")
            red_apple_index = input("Choose a red card: ")

        # Convert the input to an index
        red_apple_index = int(red_apple_index) - 1

        # Remove the red card from the agent's hand
        red_apple = self._red_apples.pop(red_apple_index)

        # Display the red card chosen
        print(f"{self._name} chose a red card.")
        logging.info(f"{self._name} chose the red card '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        # Check if the agent is a judge
        if not self._judge_status:
            logging.error(f"{self._name} is not the judge.")
            raise ValueError(f"{self._name} is not the judge.")

        # Display the red cards submitted by the other agents
        print("Red cards submitted by the other agents:")
        for i, red_apple in enumerate(red_apples):
            print(f"{i + 1}. {red_apple[list(red_apple.keys())[0]]}")

        # Prompt the agent to choose a red card
        red_apple_len = len(red_apples)
        red_apple_index = input(f"Choose a winning red card (1 - {red_apple_len}): ")

        # Validate the input
        while not red_apple_index.isdigit() or int(red_apple_index) not in range(1, red_apple_len + 1):
            print(f"Invalid input. Please choose a valid red card (1 - {red_apple_len}).")
            red_apple_index = input("Choose a winning red card: ")

        # Convert the input to an index
        red_apple_index = int(red_apple_index) - 1

        # Remove the red card from the agent's hand
        winning_red_apple = red_apples.pop(red_apple_index)

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

        # Choose a random red card
        red_apple = self._red_apples.pop(random.choice(range(len(self._red_apples))))

        # Display the red card chosen
        print(f"{self._name} chose a red card.")
        logging.info(f"{self._name} chose the red card '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        # Check if the agent is a judge
        if not self._judge_status:
            logging.error(f"{self._name} is not the judge.")
            raise ValueError(f"{self._name} is not the judge.")

        # Choose a random winning red card
        winning_red_apple = random.choice(red_apples)

        return winning_red_apple

# Import the "Model" class from local library here to avoid circular importing
from source.model import Model, LRModel, NNModel

class AIAgent(Agent):
    """
    AI agent for the 'Apples to Apples' game using Word2Vec and Linear Regression.
    """
    def __init__(self, name: str, model_type: LRModel | NNModel, pretrained_model: str, pretrain: bool) -> None:
        super().__init__(name)
        self.__nlp_model: KeyedVectors | None = None
        self.__vectors = None
        self.__model_type: LRModel | NNModel = model_type
        self.__pretrained_model: str = pretrained_model
        self.__pretrain: bool = pretrain
        self.__self_model: Model | None = None
        self.__opponents: list[Agent] = []
        self.__opponent_models: dict[Agent, LRModel | NNModel] | None = None

    def get_opponent_models(self, key: Agent) -> LRModel | NNModel | None:
        if self.__opponent_models is None:
            logging.error("Opponent Models have not been initialized.")
            raise ValueError("Opponent Models have not been initialized.")
        else:
            return self.__opponent_models.get(key)

    def initialize_models(self, nlp_model: KeyedVectors, all_players: list[Agent]) -> None:
    # def initialize_models(self, vectors, all_players: list[Agent]) -> None:
        """
        Initialize the Linear Regression and/or Neural Network models for the AI agent.
        """
        # Initialize the nlp model
        self.__nlp_model = nlp_model

        # # Initialize the vectors
        # self.__vectors = vectors

        # # Initialize the self_model
        # if self.model_type is LRModel:
        #     self.self_model = LRModel(self, self.nlp_model.vector_size, self.pretrained_model, self.pretrain)
        # elif self.model_type is NNModel:
        #     self.self_model = NNModel(self, self.nlp_model.vector_size, self.pretrained_model, self.pretrain)

        # Determine the opponents
        self.__opponents = [agent for agent in all_players if agent != self]
        logging.debug(f"opponents: {[agent.get_name() for agent in self.__opponents]}")

        # Initialize the self and opponent models
        if self.__model_type is LRModel:
            self.__self_model = LRModel(self, self.__nlp_model.vector_size, self.__pretrained_model, self.__pretrain)
            self.__opponent_models = {agent: LRModel(agent, self.__nlp_model.vector_size, self.__pretrained_model, self.__pretrain) for agent in self.__opponents}
            logging.debug(f"LRModel - opponent_models: {self.__opponent_models}")
        elif self.__model_type is NNModel:
            self.__self_model = NNModel(self, self.__nlp_model.vector_size, self.__pretrained_model, self.__pretrain)
            self.__opponent_models = {agent: NNModel(agent, self.__nlp_model.vector_size, self.__pretrained_model, self.__pretrain) for agent in self.__opponents}
            logging.debug(f"NNModel - opponent_models: {self.__opponent_models}")
        # if self.model_type is LRModel:
        #     self.self_model = LRModel(self, self.vectors.vector_size)
        #     self.opponent_models = {agent: LRModel(agent, self.vectors.vector_size) for agent in self.opponents}
        # elif self.model_type is NNModel:
        #     self.self_model = NNModel(self, self.vectors.vector_size)
        #     self.opponent_models = {agent: NNModel(agent, self.vectors.vector_size) for agent in self.opponents}

    def train_models(self, nlp_model: KeyedVectors, green_apple: GreenApple, winning_red_apple: RedApple, loosing_red_apples: list[RedApple], judge: Agent) -> None:
        """
        Train the AI model with the new green card, red card, and judge.
        """
        # Check if the agent self_model has been initialized
        if self.__opponent_models is None:
            logging.error("Opponent Models have not been initialized.")
            raise ValueError("Opponent Models have not been initialized.")

        # Train the AI models with the new green card, red card, and judge
        for agent in self.__opponents:
            if judge == agent:
                agent_model: LRModel | NNModel = self.__opponent_models[agent]
                if self.__model_type in [LRModel, NNModel]:
                    agent_model.train_model(nlp_model, green_apple, winning_red_apple, loosing_red_apples)
                    logging.debug(f"Trained {agent.get_name()}'s model with the new green card, red card, and judge.")

    def log_models(self):
        print(self.__opponent_models)

    def choose_red_apple(self, current_judge: Agent, green_apple: GreenApple) -> RedApple:
        # Check if the agent is a judge
        if self._judge_status:
            logging.error(f"{self._name} is the judge.")
            raise ValueError(f"{self._name} is the judge.")

        # Choose a red card
        red_apple: RedApple | None = None

        # Check that the models were initialized
        if self.__opponent_models is None:
            logging.error("Models have not been initialized.")
            raise ValueError("Models have not been initialized.")

        # Check that the nlp model was initialized
        if self.__nlp_model is None:
            logging.error("NLP model has not been initialized.")
            raise ValueError("NLP model has not been initialized.")

        # Run the AI model to choose a red card based on current judge
        red_apple = self.__opponent_models[current_judge].choose_red_apple(self.__nlp_model, green_apple, self._red_apples)
        self._red_apples.remove(red_apple)

        # Display the red card chosen
        print(f"{self._name} chose a red card.")
        logging.info(f"{self._name} chose the red card '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        # Check if the agent is a judge
        if not self._judge_status:
            logging.error(f"{self._name} is not the judge.")
            raise ValueError(f"{self._name} is not the judge.")

        # Check if the agent self_model has been initialized
        if self.__self_model is None:
            logging.error("Model has not been initialized.")
            raise ValueError("Model has not been initialized.")

        # Check that the nlp model was initialized
        if self.__nlp_model is None:
            logging.error("NLP model has not been initialized.")
            raise ValueError("NLP model has not been initialized.")

        # Choose a winning red card
        winning_red_apple_dict: dict[str, RedApple] = self.__self_model.choose_winning_red_apple(self.__nlp_model, green_apple, red_apples)

        return winning_red_apple_dict


#Define the mapping from user input to model type
model_type_mapping = {
    '1': LRModel,
    '2': NNModel
}


if __name__ == "__main__":
    pass
