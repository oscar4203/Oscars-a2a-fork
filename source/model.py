# Description: AI model logic for use in the AI agents in the 'Apples to Apples' game.

# Standard Libraries
import os
import logging
import numpy as np

# Third-party Libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # Suppress TensorFlow logging
import keras.api._v2.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, ELU
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Local Modules
from source.apples import GreenApple, RedApple
from source.agent import Agent
from source.data_classes import ChosenAppleVectors


class Model():
    """
    Base class for the AI models.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_archetype: str, training_mode: bool) -> None:
        # Initialize the model attributes
        self._vector_base_directory = "./agents/"
        self._judge: Agent = judge # The judge to be modeled
        self._vector_size = vector_size
        self._pretrained_archetype: str = pretrained_archetype # The name of the pretrained model archetype (e.g., Literalist, Contrarian, Comedian)
        self._training_mode: bool = training_mode
        self._winning_apples: list[dict[GreenApple, RedApple]] = []
        self._losing_apples: list[dict[GreenApple, RedApple]] = []
        self._winning_apple_vectors: np.ndarray = np.empty((0, self._vector_size))
        self._losing_apple_vectors: np.ndarray = np.empty((0, self._vector_size))

        # Initialize pretrained vectors
        # NOTE: slope and bias vectors will remained unchanged for a self model
        # self._y_target: np.ndarray,
        # self._slope_vectors: np.ndarray,
        # self._green_apple_vectors: np.ndarray,
        # self._red_apple_vectors: np.ndarray,
        # self._bias_vectors: np.ndarray = self._get_pretrained_vectors()

        self._pretrained_vectors: list[ChosenAppleVectors] = self._load_pretrained_vectors()

        # Learning attributes
        self._y_target: np.ndarray = np.zeros(shape=vector_size) # Target score for the model
        self._learning_rate = 0.01  # Learning rate for updates

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(judge={self._judge.get_name()}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(judge={self._judge}, "\
               f"slope_vector={self._slope_vectors}, bias_vector={self._bias_vectors}, "\
               f"y_target={self._y_target}, learning_rate={self._learning_rate})"

    # def get_current_slope_and_bias_vectors(self) -> tuple[np.ndarray, np.ndarray]:
    #     return self._slope_vectors, self._bias_vectors

    def _save_pretrained_vectors(self, chosen_apple_vectors: list[ChosenAppleVectors]) -> None:
        """
        Load the pretrained vectors from the .npz file.
        The vectors include: green apple vectors, winning red apple vectors, and losing red apple vectors.
        """
        # Ensure the base directory exists
        try:
            if not os.path.exists(self._vector_base_directory):
                os.makedirs(self._vector_base_directory, exist_ok=True)
                logging.info(f"Created vector directory: {self._vector_base_directory}")
            else:
                logging.info(f"Directory already exists: {self._vector_base_directory}")
        except OSError as e:
            logging.error(f"Error creating vector directory: {e}")

        # Ensure the tmp directory exists, if not in training mode
        if not self._training_mode:
            tmp_directory = self._vector_base_directory + "tmp/"
            try:
                if not os.path.exists(tmp_directory):
                    os.makedirs(tmp_directory, exist_ok=True)
                    logging.info(f"Created tmp directory: {tmp_directory}")
                else:
                    logging.info(f"Tmp directory already exists: {tmp_directory}")
            except OSError as e:
                logging.error(f"Error creating tmp directory: {e}")

        # Define the file path for the pretrained vectors
        pretrained_vectors_filepath = f"{self._vector_base_directory}{self._pretrained_archetype}_vectors.npz"

        # If in training_mode, save the vectors to the tmp directory
        if self._training_mode:
            pretrained_vectors_filepath = f"{tmp_directory}{self._pretrained_archetype}_vectors-tmp.npz"

        # Load the vectors from the pretrained model
        try:
            # If the file exists, load the previous pretrained vectors
            if os.path.exists(pretrained_vectors_filepath):
                existing_pretrained_vectors = self._load_pretrained_vectors()
                existing_pretrained_vectors.extend(chosen_apple_vectors) # Keep the order between the existing and new vectors
                chosen_apple_vectors = existing_pretrained_vectors

            # Create a dictionary to store each ChosenAppleVectors object
            data_dict: dict[str, np.ndarray] = {}
            for i, item in enumerate(chosen_apple_vectors):
                data_dict[f'green_apple_vector_{i}'] = item.green_apple_vector
                data_dict[f'winning_red_apple_vector_{i}'] = item.winning_red_apple_vector
                data_dict[f'losing_red_apple_vectors_{i}'] = item.losing_red_apple_vectors

            # Save to .npz file
            np.savez(pretrained_vectors_filepath, **data_dict)
            logging.info(f"Saved vectors to {pretrained_vectors_filepath}")
        # Handle any errors that occur
        except OSError as e:
            logging.error(f"Error saving vectors: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    def _load_pretrained_vectors(self) -> list[ChosenAppleVectors]:
        # Ensure the base directory exists
        try:
            if not os.path.exists(self._vector_base_directory):
                os.makedirs(self._vector_base_directory, exist_ok=True)
                logging.info(f"Created vector directory: {self._vector_base_directory}")
            else:
                logging.info(f"Directory already exists: {self._vector_base_directory}")
        except OSError as e:
            logging.error(f"Error creating vector directory: {e}")

        # Define the file path for the pretrained vectors
        pretrained_vectors_filepath = f"{self._vector_base_directory}{self._pretrained_archetype}_vectors.npz"

        try:
            loaded_data = np.load(pretrained_vectors_filepath)
            data = []

            # Determine the number of ChosenAppleVectors objects
            num_objects = len([key for key in loaded_data.keys() if key.startswith('green_apple_vector_')])

            for i in range(num_objects):
                green_apple_vector = loaded_data[f'green_apple_vector_{i}']
                winning_red_apple_vector = loaded_data[f'winning_red_apple_vector_{i}']
                losing_red_apple_vectors = loaded_data[f'losing_red_apple_vectors_{i}']
                data.append(ChosenAppleVectors(
                    green_apple_vector=green_apple_vector,
                    winning_red_apple_vector=winning_red_apple_vector,
                    losing_red_apple_vectors=losing_red_apple_vectors
                ))
            logging.info(f"Loaded vectors from {pretrained_vectors_filepath}")
        except OSError as e:
            logging.error(f"Error loading vectors: {e}")
            data = []

        return data

    def reset_model(self) -> None:
        """
        Reset the model data and vectors.
        """
        # TODO - revisit this, might have to rewrite, perhaps reset with random values?
        self._pretrained_vectors = self._load_pretrained_vectors()
        logging.debug(f"Reset the model data and vectors.")

    def _append_new_winning_losing_apples(self, green_apple: GreenApple, winning_red_apple: RedApple, losing_red_apples: list[RedApple]) -> None:
        """
        Append the new winning and losing green and red apple pairs.
        """
        # Append the winning apple pair
        winning_apple_pair: dict[GreenApple, RedApple] = {green_apple: winning_red_apple}
        self._winning_apples.append(winning_apple_pair)

        # Append the losing apple pairs
        for red_apple in losing_red_apples:
            losing_apple_pair: dict[GreenApple, RedApple] = {green_apple: red_apple}
            self._losing_apples.append(losing_apple_pair)

    def _calculate_x_vector(self, green_apple: GreenApple, red_apple: RedApple, train_on_extra_vectors: bool) -> np.ndarray:
        """
        Calculate and return the new winning x vector, which is the product of the green and red apple vectors.
        """
        # Get the green and red apple vectors
        green_vector: np.ndarray | None = green_apple.get_adjective_vector()
        red_vector: np.ndarray | None = red_apple.get_noun_vector()

        # Check that the green vector is not None
        if green_vector is None:
            logging.error(f"Green apple vector is None.")
            raise ValueError("Green apple vector is None.")

        # Check that the red vector is not None
        if red_vector is None:
            logging.error(f"Red apple vector is None.")
            raise ValueError("Red apple vector is None.")

        # Calculate the x vector (product of green and red vectors)
        x_vector: np.ndarray = np.multiply(green_vector, red_vector)
        logging.debug(f"New winning apple pair vector: {x_vector}")

        # Train on the extra vectors, if applicable
        if train_on_extra_vectors:
            # Get the extra vectors
            green_vector_extra: np.ndarray | None = green_apple.get_synonyms_vector()
            winning_red_vector_extra: np.ndarray | None = red_apple.get_description_vector()

            # Check that the green extra vector is not None
            if green_vector_extra is None:
                logging.error(f"Green apple vector is None.")
                raise ValueError("Green apple vector is None.")

            # Check that the red extra vector is not None
            if winning_red_vector_extra is None:
                logging.error(f"Red apple vector is None.")
                raise ValueError("Red apple vector is None.")

            # Calculate the extra x vector (product of green and red extra vectors)
            x_vector_extra: np.ndarray = np.multiply(green_vector_extra, winning_red_vector_extra)
            logging.debug(f"New winning apple pair extra vector: {x_vector_extra}")

            # Calculate the average of the x and extra x vectors
            x_vector = np.add(x_vector, x_vector_extra) / 2

        return x_vector

    def _initialize_y_vectors(self, x_vectors: np.ndarray, winning_apple: bool = True) -> np.ndarray:
        """
        Linear regression algorithm for the AI agent.
        \nEquation: y = mx + b ===>>> where y is the predicted preference output, m is the slope vector, x is the product of green and red apple vectors, and b is the bias vector.
        """
        # Set the y value
        if winning_apple:
            y_value = 1
        else:
            y_value = -1

        # Get the number of vectors
        num_of_vectors = len(x_vectors)

        # Create an ndarray filled with y values
        y_vectors = np.full((num_of_vectors, self._vector_size), y_value)

        # Ensure the x and y target arrays have the same number of vectors with the same dimensions
        assert num_of_vectors == len(y_vectors), "Number of vectors do not match"
        assert all(x.shape == y.shape for x, y in zip(x_vectors, y_vectors)), "Vector dimensions do not match"

        return y_vectors

    # TODO - REMOVE THIS METHOD
    def _calculate_and_append_losing_x_vectors(self, green_apple: GreenApple, losing_red_apples: list[RedApple], train_on_extra_vectors: bool) -> None:
        """
        Calculate and append the new losing x vectors, which are a multiplication of green and red apple vectors.
        """
        # Get the green apple vector
        green_vector: np.ndarray | None = green_apple.get_adjective_vector()

        # Check that the green and red vectors are not None
        if green_vector is None:
            logging.error(f"Green apple vector is None.")
            raise ValueError("Green apple vector is None.")

        # Get the green apple extra vector, if applicable
        if train_on_extra_vectors:
            green_vector_extra: np.ndarray | None = green_apple.get_synonyms_vector()

            # Check that the green extra vector is not None
            if green_vector_extra is None:
                logging.error(f"Green apple vector is None.")
                raise ValueError("Green apple vector is None.")

        # Calculate the losing apple pair vectors
        for red_apple in losing_red_apples:
            # Get the red apple vector
            red_vector: np.ndarray | None = red_apple.get_noun_vector()

            # Check that the red vector is not None
            if red_vector is None:
                logging.error(f"Red apple vector is None.")
                raise ValueError("Red apple vector is None.")

            # Calculate the x vector (product of green and red vectors)
            x_vector: np.ndarray = np.multiply(green_vector, red_vector)
            logging.debug(f"New losing apple pair vector: {x_vector}")

            # Append the x vector to the list of losing apple pair vectors
            self._losing_apple_vectors = np.vstack([self._losing_apple_vectors, x_vector])

            # Get the red apple extra vector, if applicable
            if train_on_extra_vectors:
                # Get the red apple extra vector
                red_vector_extra: np.ndarray | None = red_apple.get_description_vector()

                # Check that the red extra vector is not None
                if red_vector_extra is None:
                    logging.error(f"Red apple vector is None.")
                    raise ValueError("Red apple vector is None.")

                # Calculate the extra x vector (product of green and red extra vectors)
                x_vector_extra: np.ndarray = np.multiply(green_vector_extra, red_vector_extra)
                logging.debug(f"New losing apple pair extra vector: {x_vector_extra}")

                # Append the extra x vector to the list of losing apple pair vectors
                self._losing_apple_vectors = np.vstack([self._losing_apple_vectors, x_vector_extra])

    # # TODO - REMOVE THIS METHOD
    # def _calculate_and_append_all_x_vectors(self, green_apple: GreenApple, winning_red_apple: RedApple, losing_red_apples: list[RedApple], train_on_extra_vectors: bool, train_on_losing_red_apples: bool) -> None:
    #     """
    #     Helper function for train_model(). Calculate and append methods for the x vectors for all winning and losing green and red apple pairs.
    #     """
    #     # Append the new winning and losing apple pairs
    #     self._append_new_winning_losing_apples(green_apple, winning_red_apple, losing_red_apples)

    #     # Calculate and append the new winning x vectors
    #     winning_x_vector: np.ndarray = self._calculate_x_vector(green_apple, winning_red_apple, train_on_extra_vectors)

    #     # Calculate and append the new losing x vectors, if applicable
    #     if train_on_losing_red_apples:
    #         self._calculate_and_append_losing_x_vectors(green_apple, losing_red_apples, train_on_extra_vectors)

    def train_model(self, green_apple: GreenApple, winning_red_apple: RedApple, losing_red_apples: list[RedApple], train_on_extra_vectors: bool, train_on_losing_red_apples: bool) -> None:
        """
        Train the model using pairs of green and red apple vectors.
        """
        raise NotImplementedError("Subclass must implement the 'train_model' method")


    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple], train_on_extra_vectors: bool = False, train_on_losing_red_apples: bool = False) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        """
        raise NotImplementedError("Subclass must implement the 'choose_red_apple' method")

    def _calculate_y_output(self, slope_vector: np.ndarray, x_vector: np.ndarray, bias_vector: np.ndarray) -> np.ndarray:
        """
        Caculates the y_vector preference output given a slope vector, x vector, and bias vector.
        """
        return np.multiply(slope_vector, x_vector) + bias_vector

    def _calculate_score(self, slope_predict: np.ndarray, bias_predict: np.ndarray, slope_target: np.ndarray, bias_target: np.ndarray, use_euclidean: bool = False) -> float:
        """
        Calculates the score using either Mean Squared Error (MSE) or Euclidean distance given the predicted slope and bias vectors.
        The output is always non-negative.

        Parameters:
        slope_predict (np.ndarray): Predicted slope vectors.
        bias_predict (np.ndarray): Predicted bias vectors.
        use_euclidean (bool): If True, use Euclidean distance; otherwise, use MSE. Default is False.

        Returns:
        float: The calculated score.
        """
        if use_euclidean:
            # Calculate the Euclidean distance for slope and bias
            euclidean_slope = np.linalg.norm(slope_predict - slope_target)
            euclidean_bias = np.linalg.norm(bias_predict - bias_target)

            # Combine the Euclidean distances for slope and bias
            total_distance = euclidean_slope + euclidean_bias
            return float(total_distance)
        else:
            # Calculate the MSE for slope and bias
            mse_slope = np.mean((slope_predict - slope_target) ** 2)
            mse_bias = np.mean((bias_predict - bias_target) ** 2)

            # Combine the MSE for slope and bias
            mse_total = mse_slope + mse_bias
            return float(mse_total)

    def choose_winning_red_apple(self, green_apple: GreenApple, opponent_red_apples: list[dict[Agent, RedApple]], train_on_extra_vectors: bool = False) -> dict[Agent, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        """
        raise NotImplementedError("Subclass must implement the 'choose_winning_red_apple' method")


class LRModel(Model):
    """
    Linear Regression model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_archetype: str, training_mode: bool) -> None:
        super().__init__(judge, vector_size, pretrained_archetype, training_mode)

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()

    def __linear_regression(self, x_vector_array: np.ndarray, y_vector_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Linear regression algorithm for the AI agent.
        \nEquation: y = mx + b ===>>> where y is the predicted preference output, m is the slope vector, x is the product of green and red apple vectors, and b is the bias vector.
        """
        # Initalize the sum variables
        sumx: np.ndarray = np.zeros(self._vector_size)
        sumx2: np.ndarray = np.zeros(self._vector_size)
        sumxy: np.ndarray = np.zeros(self._vector_size)
        sumy: np.ndarray = np.zeros(self._vector_size)
        sumy2: np.ndarray = np.zeros(self._vector_size)

        # Calculate the sums for the x and y vectors
        for x, y in zip(x_vector_array, y_vector_array):
            sumx = np.add(sumx, x)
            sumx2 = np.add(sumx2, np.multiply(x, x))
            sumxy = np.add(sumxy, np.multiply(x, y))
            sumy = np.add(sumy, y)
            sumy2 = np.add(sumy2, np.multiply(y, y))

        logging.debug(f"Final sums - sumx:{sumx}, sumx2:{sumx2}, sumxy:{sumxy}, sumy:{sumy}, sumy2:{sumy2}")

        # Determine the number of vectors
        n: int = len(x_vector_array)
        ny: int = len(y_vector_array)
        assert n == ny, "Number of x and y vectors do not match"

        # Calculate the denominators
        denoms: np.ndarray = np.full(self._vector_size, n) * sumx2 - np.multiply(sumx, sumx)

        logging.debug(f"Denominators: {denoms}")

        # Initialize the slope and intercept elements to zero
        m: np.ndarray = np.zeros(self._vector_size)
        b: np.ndarray = np.zeros(self._vector_size)

        logging.debug(f"Initial slope: {m}")
        logging.debug(f"Initial intercept: {b}")

        # Calculate the slopes and intercepts
        for i, denom in enumerate(denoms):
            # Avoid division by zero
            if denom == 0.0:
                continue
            m[i] = (n * sumxy[i] - sumx[i] * sumy[i]) / denom
            b[i] = (sumy[i] * sumx2[i] - sumx[i] * sumxy[i]) / denom

        logging.debug(f"Final slope: {m}")
        logging.debug(f"Final intercept: {b}")

        return m, b

    def train_model(self, green_apple: GreenApple, winning_red_apple: RedApple, losing_red_apples: list[RedApple], train_on_extra_vectors: bool, train_on_losing_red_apples: bool) -> None:
        """
        Train the model using winning green and red apple pairs and losing green and red apple pairs if applicable.
        """
        # Append the new winning and losing apple pairs
        self._append_new_winning_losing_apples(green_apple, winning_red_apple, losing_red_apples)

        # Calculate the x vectors for the winning and losing apple pairs
        x_vector_array = np.array([self._calculate_x_vector(green_apple, winning_red_apple, train_on_extra_vectors)])

        # Initialize the y vectors for the winning apple pairs
        y_vector_array = self._initialize_y_vectors(x_vector_array, winning_apple=True)

        # Calculate the x vectors for the losing apple pairs, if applicable
        if train_on_losing_red_apples:
            for red_apple in losing_red_apples:
                x_vector_array = np.vstack([x_vector_array, self._calculate_x_vector(green_apple, red_apple, train_on_extra_vectors)])

            # Initialize the y vectors for the losing apple pairs
            y_vector_array = np.vstack([y_vector_array, self._initialize_y_vectors(x_vector_array[1:], winning_apple=False)])

        # Run the linear regression algorithm
        logging.debug(f"Old slope vector: {self._slope_vectors}")
        logging.debug(f"Old bias vector: {self._bias_vectors}")
        slope_vector, bias_vector = self.__linear_regression(x_vector_array, y_vector_array)

        # Save the updated slope and bias vectors to .npy files
        self._save_vectors_to_npy(slope_vector, bias_vector)

        # Update the slope and bias vectors locally
        self._slope_vectors = slope_vector
        self._bias_vectors = bias_vector

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple], train_on_extra_vectors: bool = False, train_on_losing_red_apples: bool = False) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        This method applies the private linear regression methods to predict the best red apple.
        """
        # Get the green apple vector
        green_apple_vector = green_apple.get_adjective_vector()

        # Initialize the best score and best red apple
        best_red_apple: RedApple | None = None
        best_score: float = -np.inf

        # Get the red apple vectors and calculate the score
        for red_apple in red_apples_in_hand:
            red_apple_vector = red_apple.get_noun_vector()

            # Check that the green and red vectors are not None
            if green_apple_vector is None:
                raise ValueError("Green apple vector is None.")
            if red_apple_vector is None:
                raise ValueError("Red apple vector is None.")

            score = self._calculate_score(green_apple_vector, red_apple_vector)

            if score > best_score:
                best_red_apple = red_apple
                best_score = score

        # Check if the best red apple was chosen
        if best_red_apple is None:
            raise ValueError("No red apple was chosen.")

        return best_red_apple

    def choose_winning_red_apple(self, green_apple: GreenApple, opponent_red_apples: list[dict[Agent, RedApple]], train_on_extra_vectors: bool = False) -> dict[Agent, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        This method is only used by the self model and applies the private linear regression methods to predict the winning red apple.
        """
        # Initialize variables to track the best choice
        winning_red_apple: dict[Agent, RedApple] | None = None
        best_score = np.inf

        # Iterate through the red apples to find the best one
        for red_apple_dict in opponent_red_apples:
            for _, red_apple in red_apple_dict.items():
                # Calculate the x vectors for the given green and red apple pair
                x_vector = self._calculate_x_vector(green_apple, red_apple, train_on_extra_vectors)
                logging.debug(f"x_vector: {x_vector}")

                # Initialize the y_predict vector (the submitted red apple is assumed to be a winning apple)
                y_predict = self._initialize_y_vectors(np.array([x_vector]), winning_apple=True)

                # Use linear regression to predict the preference output
                slope_predict, bias_predict = self.__linear_regression(x_vector, y_predict)

                # Evaluate the score using RMSE
                score_mse = self._calculate_score(slope_predict, bias_predict, slope_target, bias_target)
                logging.debug(f"score_mse: {score_mse}")

                # Evaluate the score using Euclidean distance
                score_euclid = self._calculate_score(slope_predict, bias_predict, slope_target, bias_target, True)
                logging.debug(f"score_euclid: {score_euclid}")

                # Choose which score to use
                score = score_mse
                # score = score_euclid

                # Update the best score and accompanying red apple
                if score < best_score:
                    best_score = score
                    winning_red_apple = red_apple_dict
                    logging.debug(f"New best score: {best_score}")
                    logging.debug(f"New best red apple: {winning_red_apple}")

        # Check if the winning red apple is None
        if winning_red_apple is None:
            raise ValueError("No winning red apple was chosen.")
        logging.debug(f"Winning red apple: {winning_red_apple}")

        return winning_red_apple


class NNModel(Model):
    """
    Neural Network model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_archetype: str, training_mode: bool) -> None:
        super().__init__(judge, vector_size, pretrained_archetype, training_mode)

        # Define the neural network model architecture with two hidden layers
        self.model = Sequential([
            Dense(vector_size, input_dim=vector_size, activation="relu"), # Input layer
            # BatchNormalization(),
            # Dropout(0.5),
            Dense(vector_size, activation="relu"), # Hidden layer 1
            # BatchNormalization(),
            # Dropout(0.5),
            Dense(vector_size, activation="relu"), # Hidden layer 2
            # BatchNormalization(),
            # Dropout(0.5),
            Dense(vector_size)  # Output layer
        ])

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=self._learning_rate), loss="mean_squared_error")

    def __forward_propagation(self, green_apple_vector, red_apple_vector) -> np.ndarray:
        """
        Forward propagation algorithm for the AI agent.
        """
        # y = mx + b, where x is the product of green and red apple vectors
        x = np.multiply(green_apple_vector, red_apple_vector)
        # y_pred = np.multiply(self._slope_vector, x) + self._bias_vector
        # return y_pred
        return self.model.predict(np.array([x]))[0]

    def __back_propagation(self, green_apple_vector, red_apple_vector):
        """
        Back propagation algorithm for the AI agent.
        """
        # # Calculate the error
        # y_pred = self.__forward_propagation(green_apple_vector, red_apple_vector)
        # error = self._y_target - y_pred

        # # Update rule for gradient descent
        # x = np.multiply(green_apple_vector, red_apple_vector)
        # self._slope_vector += self._learning_rate * np.dot(error, x)
        # self._bias_vector += self._learning_rate * error

        # # Update the target score based on the error
        # self._y_target = self._y_target - error
        x = np.multiply(green_apple_vector, red_apple_vector)
        self.model.train_on_batch(np.array([x]), np.array([self._y_target]))

    def train_model(self, green_apple: GreenApple, winning_red_apple: RedApple, losing_red_apples: list[RedApple], train_on_extra_vectors: bool, train_on_losing_red_apples: bool) -> None:
        """
        Train the model using pairs of green and red apple vectors.
        """
        # Append the new winning and losing apple pairs
        self._append_new_winning_losing_apples(green_apple, winning_red_apple, losing_red_apples)

        # Calculate the x vectors for the winning and losing apple pairs
        x_vector_array = np.array([self._calculate_x_vector(green_apple, winning_red_apple, train_on_extra_vectors)])

        # Initialize the y vectors for the winning apple pairs
        y_vector_array = self._initialize_y_vectors(x_vector_array, winning_apple=True)

        # Calculate the x vectors for the losing apple pairs, if applicable
        if train_on_losing_red_apples:
            for red_apple in losing_red_apples:
                x_vector_array = np.vstack([x_vector_array, self._calculate_x_vector(green_apple, red_apple, train_on_extra_vectors)])

            # Initialize the y vectors for the losing apple pairs
            y_vector_array = np.vstack([y_vector_array, self._initialize_y_vectors(x_vector_array[1:], winning_apple=False)])

        # Log the old slope and bias vectors
        logging.debug(f"Old slope vector: {self._slope_vectors}")
        logging.debug(f"Old bias vector: {self._bias_vectors}")

        # TODO - Finish implementing the neural network training
        # # Calculate the target score
        # for pair in winning_apple_pairs_vectors:
        #     self._y_target = self.__forward_propagation(pair["green_apple_vector"], pair["red_apple_vector"])
        #     self.__back_propagation(pair["green_apple_vector"], pair["red_apple_vector"])

        # # Save the updated slope and bias vectors
        # logging.debug(f"Updated slope vector: {self._slope_vector}")
        # logging.debug(f"Updated bias vector: {self._bias_vector}")
        # self._save_vectors()

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple], train_on_extra_vectors: bool = False, train_on_losing_red_apples: bool = False) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        This method applies the private neural network methods to predict the best red apple.
        """
        # Get the green vector
        green_apple_vector = green_apple.get_adjective_vector()

        # Initialize the best score and best red apple
        best_red_apple: RedApple | None = None
        best_score: float = -np.inf

        # Iterate through the red apples to find the best one
        for red_apple in red_apples_in_hand:
            red_apple_vector = red_apple.get_noun_vector()

            # Check that the green and red vectors are not None
            if green_apple_vector is None:
                raise ValueError("Green apple vector is None.")
            if red_apple_vector is None:
                raise ValueError("Red apple vector is None.")

            score = self._calculate_score(green_apple_vector, red_apple_vector)

            if score > best_score:
                best_red_apple = red_apple
                best_score = score

        # Check if the best red apple is None
        if best_red_apple is None:
            raise ValueError("No red apple was chosen.")

        return best_red_apple

    def choose_winning_red_apple(self, green_apple: GreenApple, opponent_red_apples: list[dict[Agent, RedApple]], train_on_extra_vectors: bool = False) -> dict[Agent, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        This method applies the private neural network methods to predict the winning red apple.
        """
        # Initialize variables to track the best choice
        winning_red_apple: dict[Agent, RedApple] | None = None
        best_score = np.inf

        # Iterate through the red apples to find the best one
        for red_apple_dict in opponent_red_apples:
            for _, red_apple in red_apple_dict.items():
                # Calculate the x vectors for the given green and red apple pair
                x_vector = self._calculate_x_vector(green_apple, red_apple, train_on_extra_vectors)
                logging.debug(f"x_vector: {x_vector}")

                # Initialize the y_predict vector (the submitted red apple is assumed to be a winning apple)
                y_predict = self._initialize_y_vectors(np.array([x_vector]), winning_apple=True)

                # Calculate the predicted score
                slope_vector, bias_vector = self.__forward_propagation(x_vector, y_predict)

                # Evaluate the score
                score = self._calculate_score(slope_vector, bias_vector)
                logging.debug(f"Score: {score}")

                # Update the best score and accompanying red apple
                if score < best_score:
                    best_score = score
                    winning_red_apple = red_apple_dict
                    logging.debug(f"New best score: {best_score}")
                    logging.debug(f"New best red apple: {winning_red_apple}")

        # Check if the winning red apple is None
        if winning_red_apple is None:
            raise ValueError("No winning red apple was chosen.")
        logging.debug(f"Winning red apple: {winning_red_apple}")

        return winning_red_apple


# Define the mapping from user input to model type
model_type_mapping = {
    '1': LRModel,
    '2': NNModel
}


if __name__ == "__main__":
    pass
