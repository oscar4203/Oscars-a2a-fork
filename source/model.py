# Description: AI model logic for use in the AI agents in the 'Apples to Apples' game.

# Standard Libraries
import logging
import numpy as np
from dataclasses import dataclass

# Third-party Libraries

# Local Modules
from source.apples import GreenApple, RedApple
from source.agent import Agent


@dataclass
class ModelData:
    green_apples: list[GreenApple]
    red_apples: list[RedApple]
    winning_red_apples: list[RedApple]

    def __post_init__(self) -> None:
        logging.debug(f"Created ModelData object: {self}")

    def __str__(self) -> str:
        return f"ModelData(green_apples={[apple.adjective for apple in self.green_apples]}, red_apples={[apple.noun for apple in self.red_apples]}, " \
               f"winning_red_apples={[apple.noun for apple in self.red_apples]})"

    def __repr__(self) -> str:
        return f"ModelData(green_apples={[apple.adjective for apple in self.green_apples]}, red_apples={[apple.noun for apple in self.red_apples]}, " \
               f"winning_red_apples={[apple.noun for apple in self.red_apples]})"

    def to_dict(self) -> dict:
        return {
            "green_apples": [apple.adjective for apple in self.green_apples],
            "red_apples": [apple.noun for apple in self.red_apples],
            "winning_red_apples": [apple.noun for apple in self.winning_red_apples]
        }


class Model():
    """
    Base class for the AI models.
    """
    def __init__(self, judge: Agent, vector_size: int) -> None:
        self.judge: Agent = judge
        self.model_data: ModelData = ModelData([], [], [])
        # Initialize slope and bias vectors
        self.slope_vector = np.random.randn(vector_size)
        self.bias_vector = np.random.randn(vector_size)
        self.learning_rate = 0.01  # Learning rate for updates

    def choose_red_apple(self, green_apple: GreenApple, red_apples: list[RedApple]) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        """
        raise NotImplementedError("Subclass must implement the 'choose_red_apple' method")

    def choose_winning_red_apple(self, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        """
        raise NotImplementedError("Subclass must implement the 'choose_winning_red_apple' method")


class LRModel(Model):
    """
    Linear Regression model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int) -> None:
        super().__init__(judge, vector_size)

    def __linear_regression(self, green_apple_vector, red_apple_vector) -> np.ndarray:
        """
        Linear regression algorithm for the AI agent.
        """
        # y = mx + b, where x is the product of green and red apple vectors
        x = np.multiply(green_apple_vector, red_apple_vector)
        y_pred = np.multiply(self.slope_vector, x) + self.bias_vector
        return y_pred

    def __update_parameters(self, green_apple_vector, red_apple_vector, y_target):
        """
        Update the slope and bias vectors based on the error.
        """
        y_pred = self.__linear_regression(green_apple_vector, red_apple_vector)
        error = y_target - y_pred
        # Update rule for gradient descent
        x = np.multiply(green_apple_vector, red_apple_vector)
        self.slope_vector += self.learning_rate * np.dot(error, x)
        self.bias_vector += self.learning_rate * error

    def __train(self, green_apple_vectors, red_apple_vectors, y_target):
        """
        Train the model using pairs of green and red apple vectors.
        """
        for green_apple_vector, red_apple_vector in zip(green_apple_vectors, red_apple_vectors):
            self.__update_parameters(green_apple_vector, red_apple_vector, y_target)


class NNModel(Model):
    """
    Neural Network model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int) -> None:
        super().__init__(judge, vector_size)

    def __forward_propagation(self, green_apple_vector, red_apple_vector) -> np.ndarray:
        """
        Forward propagation algorithm for the AI agent.
        """
        # y = mx + b, where x is the product of green and red apple vectors
        x = np.multiply(green_apple_vector, red_apple_vector)
        y_pred = np.multiply(self.slope_vector, x) + self.bias_vector
        return y_pred

    def __back_propagation(self, green_apple_vector, red_apple_vector, y_target):
        """
        Back propagation algorithm for the AI agent.
        """
        y_pred = self.__forward_propagation(green_apple_vector, red_apple_vector)
        error = y_target - y_pred
        # Update rule for gradient descent
        x = np.multiply(green_apple_vector, red_apple_vector)
        self.slope_vector += self.learning_rate * np.dot(error, x)
        self.bias_vector += self.learning_rate * error

    def __train(self, green_apple_vectors, red_apple_vectors, y_target):
        """
        Train the model using pairs of green and red apple vectors.
        """
        for green_apple_vector, red_apple_vector in zip(green_apple_vectors, red_apple_vectors):
            self.__back_propagation(green_apple_vector, red_apple_vector, y_target)


if __name__ == "__main__":
    pass
