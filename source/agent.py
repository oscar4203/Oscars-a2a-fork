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
        self.name: str = name
        self.points: int = 0
        self.judge: bool = False
        self.green_apple: GreenApple | None = None
        self.red_apples: list[RedApple] = []

    def __str__(self) -> str:
        return f"Agent(name={self.name}, points={self.points}, judge={self.judge}, green_apple={self.green_apple}, red_apples={self.red_apples})"

    def __repr__(self) -> str:
        return self.__str__()

    def draw_red_apples(self, red_apple_deck: Deck) -> Deck | None:
        """
        Draw red cards from the deck, ensuring the agent has 7 red cards.
        """
        # Calculate the number of red cards to pick up
        diff = 7 - len(self.red_apples)
        if diff > 0:
            for _ in range(diff):
                # Draw a red card
                new_red_apple = red_apple_deck.draw_apple()
                if not isinstance(new_red_apple, RedApple):
                    raise TypeError("Expected a RedApple, but got a different type")
                self.red_apples.append(new_red_apple)
            if diff == 1:
                print(f"{self.name} picked up 1 red card.")
                logging.info(f"{self.name} picked up 1 red card.")
            else:
                print(f"{self.name} picked up {diff} red cards.")
                logging.info(f"{self.name} picked up {diff} red cards.")
        else:
            print(f"{self.name} cannot pick up any more red cards. Agent already has 7 red cards")
            logging.info(f"{self.name} cannot pick up the red card. Agent already has 7 red cards")

    def draw_green_apple(self, green_apple_deck: Deck) -> GreenApple:
        """
        Draw a green card from the deck (when the agent is the judge).
        """
        # Check if the Agent is a judge
        if self.judge:
            # Draw a green card
            new_green_apple = green_apple_deck.draw_apple()
            if not isinstance(new_green_apple, GreenApple):
                raise TypeError("Expected a GreenApple, but got a different type")
            self.green_apple = new_green_apple
        else:
            logging.error(f"{self.name} is the judge.")
            raise ValueError(f"{self.name} is the judge.")

        # Display the green card drawn
        print(f"{self.name} drew the green card '{self.green_apple.adjective}'.")
        logging.info(f"{self.name} drew the green card '{self.green_apple}'.")

        return self.green_apple

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
        if self.judge:
            logging.error(f"{self.name} is the judge.")
            raise ValueError(f"{self.name} is the judge.")

        # Choose a red card
        red_apple: RedApple | None = None

        # Display the red cards in the agent's hand
        print(f"{self.name}'s red cards:")
        for i, red_apple in enumerate(self.red_apples):
            print(f"{i + 1}. {red_apple.noun}")

        # Prompt the agent to choose a red card
        red_apple_len = len(self.red_apples)
        red_apple_index = input(f"Choose a red card (1 - {red_apple_len}): ")

        # Validate the input
        while not red_apple_index.isdigit() or int(red_apple_index) not in range(1, red_apple_len + 1):
            print(f"Invalid input. Please choose a valid red card (1 - {red_apple_len}).")
            red_apple_index = input("Choose a red card: ")

        # Convert the input to an index
        red_apple_index = int(red_apple_index) - 1

        # Remove the red card from the agent's hand
        red_apple = self.red_apples.pop(red_apple_index)

        # Display the red card chosen
        print(f"{self.name} chose a red card.")
        logging.info(f"{self.name} chose the red card '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        # Check if the agent is a judge
        if not self.judge:
            logging.error(f"{self.name} is not the judge.")
            raise ValueError(f"{self.name} is not the judge.")

        # Display the red cards submitted by the other agents
        print("Red cards submitted by the other agents:")
        for i, red_apple in enumerate(red_apples):
            print(f"{i + 1}. {red_apple[list(red_apple.keys())[0]].noun}")

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

        # Display the red card chosen
        logging.debug(f"winning_red_apple: {winning_red_apple}")
        round_winner = list(winning_red_apple.keys())[0]
        winning_red_apple_noun = winning_red_apple[round_winner].noun
        print(f"{self.name} chose the winning red card '{winning_red_apple_noun}'.")
        logging.info(f"{self.name} chose the winning red card '{winning_red_apple}'.")

        return winning_red_apple


class RandomAgent(Agent):
    """
    Random agent for the 'Apples to Apples' game.
    """
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def choose_red_apple(self, current_judge: Agent, green_apple: GreenApple) -> RedApple:
        # Check if the agent is a judge
        if self.judge:
            logging.error(f"{self.name} is the judge.")
            raise ValueError(f"{self.name} is the judge.")

        # Choose a random red card
        red_apple = self.red_apples.pop(random.choice(range(len(self.red_apples))))

        # Display the red card chosen
        print(f"{self.name} chose a red card.")
        logging.info(f"{self.name} chose the red card '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        # Check if the agent is a judge
        if not self.judge:
            logging.error(f"{self.name} is not the judge.")
            raise ValueError(f"{self.name} is not the judge.")

        # Choose a random winning red card
        winning_red_apple = random.choice(red_apples)

        # Display the red card chosen
        logging.debug(f"winning_red_apple: {winning_red_apple}")
        round_winner = list(winning_red_apple.keys())[0]
        winning_red_apple_noun = winning_red_apple[round_winner].noun
        print(f"{self.name} chose the winning red card '{winning_red_apple_noun}'.")
        logging.info(f"{self.name} chose the winning red card '{winning_red_apple}'.")

        return winning_red_apple

# Import the "Model" class from local library here to avoid circular importing
from source.model import Model, LRModel, NNModel

class AIAgent(Agent):
    """
    AI agent for the 'Apples to Apples' game using Word2Vec and Linear Regression.
    """
    def __init__(self, name: str, type: LRModel | NNModel) -> None:
        super().__init__(name)
        # self.nlp_model: KeyedVectors | None = None
        self.vectors = None
        self.model_type: LRModel | NNModel = type
        self.self_model: Model | None = None
        self.opponents: list[Agent] = []
        self.opponent_models: dict[Agent, LRModel | NNModel] | None = None

    def initialize_models(self, nlp_model: KeyedVectors, all_players: list[Agent]) -> None:
    # def initialize_models(self, vectors, all_players: list[Agent]) -> None:
        """
        Initialize the Linear Regression and/or Neural Network models for the AI agent.
        """
        # Initialize the nlp model
        self.nlp_model = nlp_model

        # # Initialize the vectors
        # self.vectors = vectors

        # Initialize the self_model
        if self.model_type is LRModel:
            self.self_model = LRModel(self, self.nlp_model.vector_size)
        elif self.model_type is NNModel:
            self.self_model = NNModel(self, self.nlp_model.vector_size)

        # Determine the opponents
        self.opponents = [agent for agent in all_players if agent != self]
        logging.debug(f"opponents: {[agent.name for agent in self.opponents]}")

        # Initialize the models
        if self.model_type is LRModel:
            self.self_model = LRModel(self, self.nlp_model.vector_size)
            self.opponent_models = {agent: LRModel(agent, self.nlp_model.vector_size) for agent in self.opponents}
            logging.debug(f"LRModel - opponent_models: {self.opponent_models}")
        elif self.model_type is NNModel:
            self.self_model = NNModel(self, self.nlp_model.vector_size)
            self.opponent_models = {agent: NNModel(agent, self.nlp_model.vector_size) for agent in self.opponents}
            logging.debug(f"NNModel - opponent_models: {self.opponent_models}")
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
        if self.opponent_models is None:
            logging.error("Opponent Models have not been initialized.")
            raise ValueError("Opponent Models have not been initialized.")

        # Train the AI models with the new green card, red card, and judge
        for agent in self.opponents:
            if judge == agent:
                agent_model: LRModel | NNModel = self.opponent_models[agent]
                if self.model_type in [LRModel, NNModel]:
                    agent_model.train_model(nlp_model, green_apple, winning_red_apple, loosing_red_apples)

    def log_models(self):
        print(self.opponent_models)

    def choose_red_apple(self, current_judge: Agent, green_apple: GreenApple) -> RedApple:
        # Check if the agent is a judge
        if self.judge:
            logging.error(f"{self.name} is the judge.")
            raise ValueError(f"{self.name} is the judge.")

        # Choose a red card
        red_apple: RedApple | None = None

        # Check that the models were initialized
        if self.opponent_models is None:
            logging.error("Models have not been initialized.")
            raise ValueError("Models have not been initialized.")

        # Run the AI model to choose a red card based on current judge
        red_apple = self.opponent_models[current_judge].choose_red_apple(self.nlp_model, green_apple, self.red_apples)
        self.red_apples.remove(red_apple)

        # Display the red card chosen
        print(f"{self.name} chose a red card.")
        logging.info(f"{self.name} chose the red card '{red_apple}'.")

        return red_apple

    def choose_winning_red_apple(self, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        # Check if the agent is a judge
        if not self.judge:
            logging.error(f"{self.name} is not the judge.")
            raise ValueError(f"{self.name} is not the judge.")

        # Check if the agent self_model has been initialized
        if self.self_model is None:
            logging.error("Model has not been initialized.")
            raise ValueError("Model has not been initialized.")

        # Choose a winning red card
        winning_red_apple: dict[str, RedApple] = self.self_model.choose_winning_red_apple(self.nlp_model, green_apple, red_apples)

        # Display the red card chosen
        logging.debug(f"winning_red_apple: {winning_red_apple}")
        round_winner = list(winning_red_apple.keys())[0]
        winning_red_apple_noun = winning_red_apple[round_winner].noun
        print(f"{self.name} chose the winning red card '{winning_red_apple_noun}'.")
        logging.info(f"{self.name} chose the winning red card '{winning_red_apple}'.")

        return winning_red_apple
    

#Define the mapping from user input to model type
model_type_mapping = {
    '1': LRModel,
    '2': NNModel
}


if __name__ == "__main__":
    pass
