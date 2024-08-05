# Description: AI model logic for use in the AI agents in the 'Apples to Apples' game.

# Standard Libraries
import logging
import numpy as np
from dataclasses import dataclass
import os

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
        return f"ModelData(green_apples={[apple.__adjective for apple in self.green_apples]}, red_apples={[apple.get_noun() for apple in self.red_apples]}, " \
               f"winning_red_apples={[apple.get_noun() for apple in self.red_apples]})"

    def __repr__(self) -> str:
        return f"ModelData(green_apples={[apple.__adjective for apple in self.green_apples]}, red_apples={[apple.get_noun() for apple in self.red_apples]}, " \
               f"winning_red_apples={[apple.get_noun() for apple in self.red_apples]})"

    def to_dict(self) -> dict:
        return {
            "green_apples": [apple.__adjective for apple in self.green_apples],
            "red_apples": [apple.get_noun() for apple in self.red_apples],
            "winning_red_apples": [apple.get_noun() for apple in self.winning_red_apples]
        }


class Model():
    """
    Base class for the AI models.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_model: str, pretrain: bool) -> None:
        # Initialize the model attributes
        self._vector_size = vector_size
        self._judge: Agent = judge
        self._model_data: ModelData = ModelData([], [], [])
        self._judge_pairs = [] # Hopefully a better way to store the data.
        self._pretrained_model: str = pretrained_model
        self._pretrain: bool = pretrain

        # Initialize slope and bias vectors
        self._slope_vector, self._bias_vector = self.__load_vectors(vector_size)

        # Learning attributes
        self._y_target: np.ndarray = np.zeros(shape=vector_size)  # Target score for the model
        self._learning_rate = 0.01  # Learning rate for updates

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(judge={self._judge}, model_data={self._model_data}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(judge={self._judge}, model_data={self._model_data}, "\
               f"slope_vector={self._slope_vector}, bias_vector={self._bias_vector}, "\
               f"learning_rate={self._learning_rate})"

    def get_slope_vector(self) -> np.ndarray:
        return self._slope_vector

    def get_bias_vector(self) -> np.ndarray:
        return self._bias_vector

    def __load_vectors(self, vector_size: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the slope and bias vectors from the pretrained model .npy files if they exist, otherwise initialize random values.
        """
        # Ensure the directory exists
        directory = "./agents/"
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logging.info(f"Created directory: {directory}")
            else:
                logging.info(f"Directory already exists: {directory}")
        except OSError as e:
            logging.error(f"Error creating directory: {e}")

        # Define the file paths for the vectors
        slope_vector_file: str = f"{directory}{self._pretrained_model}_slope.npy"
        bias_vector_file: str = f"{directory}{self._pretrained_model}_bias.npy"

        # Load the vectors from the pretrained model
        try:
            # Check if the files exist
            if os.path.exists(slope_vector_file) and os.path.exists(bias_vector_file):
                slope_vector = np.load(slope_vector_file)
                bias_vector = np.load(bias_vector_file)
                logging.info(f"Loaded vectors from {slope_vector_file} and {bias_vector_file}")
            else: # If not, initialize random vectors
                slope_vector = np.random.rand(vector_size)
                bias_vector = np.random.rand(vector_size)
                logging.info("Initialized random vectors")
        # Handle any errors that occur
        except OSError as e:
            logging.error(f"Error loading vectors: {e}")
            slope_vector = np.random.rand(vector_size)
            bias_vector = np.random.rand(vector_size)

        return slope_vector, bias_vector

    def _save_vectors(self) -> None:
        """
        Save the slope and bias vectors to .npy files.
        """
        # Define the directory to save the vectors
        directory = "./agents/"

        try:
            # If pretrain is True, save the vectors to the pretrained model files
            if self._pretrain:
                slope_file: str = f"{directory}{self._pretrained_model}_slope.npy"
                bias_file: str = f"{directory}{self._pretrained_model}_bias.npy"
                np.save(slope_file, self._slope_vector)
                np.save(bias_file, self._bias_vector)
                logging.info(f"Saved vectors to {slope_file} and {bias_file}")
            else: # Otherwise, save the vectors to the temporary model files
                temp_slope_file = f"{directory}{self._pretrained_model}_slope_{self._judge}-temp.npy"
                temp_bias_file = f"{directory}{self._pretrained_model}_bias_{self._judge}-temp.npy"
                np.save(temp_slope_file, self._slope_vector)
                np.save(temp_bias_file, self._bias_vector)
                logging.info(f"Saved vectors to {temp_slope_file} and {temp_bias_file}")
        except OSError as e:
            logging.error(f"Error saving vectors: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

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
    def __init__(self, judge: Agent, vector_size: int, pretrained_model: str, pretrain: bool) -> None:
        super().__init__(judge, vector_size, pretrained_model, pretrain)

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()

    def __result_vector(self, green_apple_vector: np.ndarray, red_apple_vector: np.ndarray) -> np.ndarray:
        """
        Produces the resultant vector when you run throught the algorithm
        """
        x = np.multiply(green_apple_vector, red_apple_vector)
        return np.multiply(self._slope_vector, x) + self._bias_vector

    def result(self, green_apple_vector, red_apple_vector) -> float:
        """
        Produces the final score of the model for a combination of red and green cards.
        """
        return np.sum(self.__result_vector(green_apple_vector, red_apple_vector))

    def __linear_regression(self, x_vectors: np.ndarray, y_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Linear regression algorithm for the AI agent.
        """
        assert(len(x_vectors) == len(y_vectors))

        n = float(len(x_vectors))
        # print("n =", n)
        sumx = np.zeros(self._vector_size)
        sumx2 = np.zeros(self._vector_size)
        sumxy = np.zeros(self._vector_size)
        sumy = np.zeros(self._vector_size)
        sumy2 = np.zeros(self._vector_size)

        for x, y in zip(x_vectors, y_vectors):
            sumx = np.add(sumx, x)
            sumx2 = np.add(sumx2, np.multiply(x, x))
            sumxy = np.add(sumxy, np.multiply(x, y))
            sumy = np.add(sumy, y)
            sumy2 = np.add(sumy2, np.multiply(y, y))

        denoms: np.ndarray = np.full(self._vector_size, n) * sumx2 - np.multiply(sumx, sumx)

        ms = np.zeros(self._vector_size)
        bs = np.zeros(self._vector_size)

        for i, denom in enumerate(denoms):
            if denom == 0.0:
                continue
            ms[i] = (n * sumxy[i] - sumx[i] * sumy[i]) / denom
            bs[i] = (sumy[i] * sumx2[i] - sumx[i] * sumxy[i]) / denom

        return ms, bs

    def __update_parameters(self, green_apple_vectors, red_apple_vectors):
        """
        Update the slope and bias vectors based on the error.
        """
        print(self) #for testing purposes, of

        # Calculate the error
        y_pred = self.__linear_regression(green_apple_vectors, red_apple_vectors)
        error = self._y_target - y_pred

        # Update slope and bias vectors
        x = np.multiply(green_apple_vectors, red_apple_vectors)
        self._slope_vector += self._learning_rate * np.dot(error, x) # TODO - Change self._slope_vector to a vector, right now it's a scalar
        self._bias_vector += self._learning_rate * error

        # Update the target score based on the error
        self._y_target = self._y_target - error

        print(self)

    def train_model(self, nlp_model: KeyedVectors, green_apple: GreenApple, winning_red_apple: RedApple, loosing_red_apples: list[RedApple]) -> None:
        """
        Train the model using pairs of green and red apple vectors.
        """
        # Set the green and red apple vectors
        green_apple.set_adjective_vector(nlp_model)
        winning_red_apple.set_noun_vector(nlp_model)

        for i, red in enumerate(loosing_red_apples):
            loosing_red_apples[i].set_noun_vector(nlp_model)

        # Add the new green and red apples to the model data
        # self.model_data.green_apples.append(new_green_apple)
        # self.model_data.red_apples.append(new_red_apple)

        self._judge_pairs.append((green_apple, winning_red_apple, 1.0))
        for red in loosing_red_apples:
            self._judge_pairs.append((green_apple, red, -1.0))

        # # Get the green and red apple vectors
        # green_apple_vectors = [apple.get_adjective_vector() for apple in self.model_data.green_apples]
        # red_apple_vectors = [apple.get_noun_vector() for apple in self.model_data.red_apples]

        # Calculate the target score
        # for green_apple_vector, red_apple_vector in zip(green_apple_vectors, red_apple_vectors):
        #     self.y_target = self.__linear_regression(green_apple_vector, red_apple_vector)
        #     self.__update_parameters(green_apple_vector, red_apple_vector)

        xs= []
        ys = []

        # an array of vectors of x and y data
        for pair in self._judge_pairs:
            g_vec = pair[0].get_adjective_vector()
            r_vec = pair[1].get_noun_vector()
            x_vec = np.multiply(g_vec, r_vec)
            y_vec = np.full(self._vector_size, pair[2])
            xs.append(x_vec)
            ys.append(y_vec)

        nxs = np.array(xs)
        nys = np.array(ys)

        self._slope_vector, self._bias_vector = self.__linear_regression(nxs, nys)

        # Save the updated slope and bias vectors
        # logging.debug(f"Updated slope vector: {self._slope_vector}")
        # logging.debug(f"Updated bias vector: {self._bias_vector}")
        self._save_vectors()
        logging.debug(f"Saved updated vectors")

        # Save the updated slope and bias vectors
        # logging.debug(f"Updated slope vector: {self._slope_vector}")
        # logging.debug(f"Updated bias vector: {self._bias_vector}")
        self._save_vectors()
        logging.debug(f"Saved updated vectors")

    def choose_red_apple(self, nlp_model: KeyedVectors, green_apple: GreenApple, red_apples: list[RedApple]) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        This method applies the private linear regression methods to predict the best red apple.
        """
        # Set the green and red apple vectors
        green_apple.set_adjective_vector(nlp_model)
        green_apple_vector = green_apple.get_adjective_vector()

        best_red_apple: RedApple | None = None
        best_score: float = -np.inf

        for red in red_apples:
            red.set_noun_vector(nlp_model)
            r_vec = red.get_noun_vector()

            score = self.result(green_apple_vector, r_vec)

            if score > best_score:
                best_red_apple = red
                best_score = score

        # Check if the best red apple was chosen
        if best_red_apple is None:
            raise ValueError("No red apple was chosen.")

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

                # Check that the green and red vectors are not None
                if green_apple_vector is None:
                    raise ValueError("Green apple vector is None.")
                if red_apple_vector is None:
                    raise ValueError("Red apple vector is None.")

                # Calculate the predicted score
                predicted_score = self.__linear_regression(green_apple_vector, red_apple_vector)

                # Evaluate the score difference using Euclidean distances
                score_difference = np.linalg.norm(predicted_score - self._y_target)

                if score_difference < closest_score:
                    closest_score = score_difference
                    winning_red_apple = red_apple_dict

        # Check if the winning red apple is None
        if winning_red_apple is None:
            raise ValueError("No winning red apple was chosen.")

        return winning_red_apple


class NNModel(Model):
    """
    Neural Network model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_model: str, pretrain: bool) -> None:
        super().__init__(judge, vector_size, pretrained_model, pretrain)

    def __forward_propagation(self, green_apple_vector, red_apple_vector) -> np.ndarray:
        """
        Forward propagation algorithm for the AI agent.
        """
        # y = mx + b, where x is the product of green and red apple vectors
        x = np.multiply(green_apple_vector, red_apple_vector)
        y_pred = np.multiply(self._slope_vector, x) + self._bias_vector
        return y_pred

    def __back_propagation(self, green_apple_vector, red_apple_vector):
        """
        Back propagation algorithm for the AI agent.
        """
        # Calculate the error
        y_pred = self.__forward_propagation(green_apple_vector, red_apple_vector)
        error = self._y_target - y_pred

        # Update rule for gradient descent
        x = np.multiply(green_apple_vector, red_apple_vector)
        self._slope_vector += self._learning_rate * np.dot(error, x)
        self._bias_vector += self._learning_rate * error

        # Update the target score based on the error
        self._y_target = self._y_target - error

    def train_model(self, nlp_model: KeyedVectors, green_apple: GreenApple, winning_red_apple: RedApple, loosing_red_apples: list[RedApple]) -> None:
        """
        Train the model using pairs of green and red apple vectors.
        """
        # Set the green and red apple vectors
        green_apple.set_adjective_vector(nlp_model)
        winning_red_apple.set_noun_vector(nlp_model)

        # Add the new green and red apples to the model data
        self._model_data.green_apples.append(green_apple)
        self._model_data.red_apples.append(winning_red_apple)

        # Get the green and red apple vectors

        green_apple_vectors = [apple.get_adjective_vector() for apple in self._model_data.green_apples]
        red_apple_vectors = [apple.get_noun_vector() for apple in self._model_data.red_apples]

        # Calculate the target score
        for green_apple_vector, red_apple_vector in zip(green_apple_vectors, red_apple_vectors):
            self._y_target = self.__forward_propagation(green_apple_vector, red_apple_vector)
            self.__back_propagation(green_apple_vector, red_apple_vector)

        # Save the updated slope and bias vectors
        super()._save_vectors()

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

        return winning_red_apple


if __name__ == "__main__":
    pass
