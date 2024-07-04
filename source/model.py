# Description: AI model logic for use in the AI agents in the 'Apples to Apples' game.

# Standard Libraries
import logging
import numpy as np
from dataclasses import dataclass

# Third-party Libraries
from gensim.models import KeyedVectors

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
        self.target_score = 0  # Target score for the model
        self.learning_rate = 0.01  # Learning rate for updates

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(judge={self.judge}, model_data={self.model_data}, "\
               f"slope_vector={self.slope_vector}, bias_vector={self.bias_vector}, "\
               f"learning_rate={self.learning_rate})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(judge={self.judge}, model_data={self.model_data}, "\
               f"slope_vector={self.slope_vector}, bias_vector={self.bias_vector}, "\
               f"learning_rate={self.learning_rate})"

    def choose_red_apple(self, nlp_model: KeyedVectors, green_apple: GreenApple, red_apples: list[RedApple]) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        """
        raise NotImplementedError("Subclass must implement the 'choose_red_apple' method")

    def choose_winning_red_apple(self, nlp_model: KeyedVectors, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
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

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()

    def __linear_regression(self, green_apple_vector, red_apple_vector) -> np.ndarray:
        """
        Linear regression algorithm for the AI agent.
        """
        # y = mx + b, where x is the product of green and red apple vectors
        x = np.multiply(green_apple_vector, red_apple_vector)
        # y_pred = np.multiply(self.slope_vector, x) + self.bias_vector
        y_pred = np.dot(self.slope_vector, x) + self.bias_vector
        return y_pred

    def __update_parameters(self, green_apple_vector, red_apple_vector, y_target):
        """
        Update the slope and bias vectors based on the error.
        """
        y_pred = self.__linear_regression(green_apple_vector, red_apple_vector)
        error = y_target - y_pred
        # Update rule for gradient descent
        x = np.multiply(green_apple_vector, red_apple_vector)
        self.slope_vector += self.learning_rate * np.dot(error, x) # TODO - Change self.slope_vector to a vector, right now it's a scalar
        self.bias_vector += self.learning_rate * error

    def __train(self, green_apple_vectors, red_apple_vectors, y_target):
        """
        Train the model using pairs of green and red apple vectors.
        """
        for green_apple_vector, red_apple_vector in zip(green_apple_vectors, red_apple_vectors):
            self.__update_parameters(green_apple_vector, red_apple_vector, y_target)

    def choose_red_apple(self, nlp_model: KeyedVectors, green_apple: GreenApple, red_apples: list[RedApple]) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        This method applies the private linear regression methods to predict the best red apple.
        """
        # Set the green and red apple vectors
        green_apple.set_adjective_vector(nlp_model)
        green_apple_vector = green_apple.get_adjective_vector()

        # Initialize variables to track the best choice
        closest_score = np.inf
        best_red_apple: RedApple | None = None

        # Iterate through the red apples to find the best one
        for red_apple in red_apples:
            red_apple.set_noun_vector(nlp_model)
            red_apple_vector = red_apple.get_noun_vector()

            # Calculate the predicted score
            predicted_score = self.__linear_regression(green_apple_vector, red_apple_vector)

            # Evaluate the score difference using Euclidean distances
            score_difference = np.linalg.norm(predicted_score - self.target_score)

            if score_difference < closest_score:
                closest_score = score_difference
                best_red_apple = red_apple

        # Check if the best red apple was chosen
        if best_red_apple is None:
            raise ValueError("No red apple was chosen.")

        # Assuming y_target is the score of the best red apple, update the model
        self.__train([green_apple_vector], [best_red_apple.get_noun_vector()], self.target_score)

        return best_red_apple

    def choose_winning_red_apple(self, nlp_model: KeyedVectors, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        This method applies the private linear regression methods to predict the winning red apple.
        """
        # Set the green and red apple vectors
        green_apple.set_adjective_vector(nlp_model)
        green_apple_vector = green_apple.get_adjective_vector()

        # Initialize variables to track the best choice
        closest_score = np.inf
        winning_red_apple: dict[str, RedApple] | None = None

         # Iterate through the red apples to find the best one
        for red_apple_dict in red_apples:
            for _, red_apple in red_apple_dict.items():
                red_apple.set_noun_vector(nlp_model)
                red_apple_vector = red_apple.get_noun_vector()

                # Calculate the predicted score
                predicted_score = self.__linear_regression(green_apple_vector, red_apple_vector)

                # Evaluate the score difference using Euclidean distances
                score_difference = np.linalg.norm(predicted_score - self.target_score)

                if score_difference < closest_score:
                    closest_score = score_difference
                    winning_red_apple = red_apple_dict

        # Check if the winning red apple is None
        if winning_red_apple is None:
            raise ValueError("No winning red apple was chosen.")

        # Assuming y_target is the score of the winning red apple, update the model
        self.__train([green_apple_vector], [winning_red_apple[next(iter(winning_red_apple))].get_noun_vector()], self.target_score)

        return winning_red_apple


class NNModel(Model):
    """
    Neural Network model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int) -> None:
        super().__init__(judge, vector_size)

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()

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

    def choose_red_apple(self, nlp_model: KeyedVectors, green_apple: GreenApple, red_apples: list[RedApple]) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        This method applies the private neural network methods to predict the best red apple.
        """
        # Set the green and red apple vectors
        green_apple.set_adjective_vector(nlp_model)
        for red_apple in red_apples:
            red_apple.set_noun_vector(nlp_model)

        # Initialize the best score and best red apple
        best_score = -np.inf
        best_red_apple: RedApple | None = None
        green_apple_vector = green_apple.get_adjective_vector()

        # Iterate through the red apples to find the best one
        for red_apple in red_apples:
            red_apple_vector = red_apple.get_noun_vector()
            score = self.__forward_propagation(green_apple_vector, red_apple_vector)
            if score > best_score:
                best_score = score
                best_red_apple = red_apple

        # Check if the best red apple is None
        if best_red_apple is None:
            raise ValueError("No red apple was chosen.")

        # Assuming y_target is the score of the best red apple, update the model
        self.__train([green_apple_vector], [best_red_apple.get_noun_vector()], best_score)

        return best_red_apple

    def choose_winning_red_apple(self, nlp_model: KeyedVectors, green_apple: GreenApple, red_apples: list[dict[str, RedApple]]) -> dict[str, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        This method applies the private neural network methods to predict the winning red apple.
        """
        best_score = -np.inf
        winning_red_apple: dict[str, RedApple] | None = None
        green_apple_vector = green_apple.get_adjective_vector()

        for red_apple_dict in red_apples:
            for _, red_apple in red_apple_dict.items():
                red_apple_vector = red_apple.get_noun_vector()
                score = self.__forward_propagation(green_apple_vector, red_apple_vector)
                if score > best_score:
                    best_score = score
                    winning_red_apple = red_apple_dict

        # Check if the winning red apple is None
        if winning_red_apple is None:
            raise ValueError("No winning red apple was chosen.")

        # Assuming y_target is the score of the winning red apple, update the model
        self.__train([green_apple_vector], [winning_red_apple[next(iter(winning_red_apple))].get_noun_vector()], best_score)

        return winning_red_apple


if __name__ == "__main__":
    pass
