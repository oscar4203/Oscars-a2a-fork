# Description: AI model logic for use in the AI agents in the 'Apples to Apples' game.

# Standard Libraries
import os
import logging
from dataclasses import dataclass
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
from source.data_classes import GameState


class Model():
    """
    Base class for the AI models.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_archetype: str, training_mode: bool) -> None:
        # Initialize the model attributes
        self._vector_base_directory = "./agents/"
        self._vector_size = vector_size
        self._judge: Agent = judge # The judge to be modeled
        self._winning_apples: list[dict[GreenApple, RedApple]] = []
        self._losing_apples: list[dict[GreenApple, RedApple]] = []
        self._winning_apple_vectors: np.ndarray = np.empty((0, self._vector_size))
        self._losing_apple_vectors: np.ndarray = np.empty((0, self._vector_size))
        self._pretrained_archetype: str = pretrained_archetype # The name of the pretrained model archetype (e.g., Literalist, Contrarian, Comedian)
        self._training_mode: bool = training_mode

        # Initialize slope and bias vectors
        self._slope_vector, self._bias_vector = self._get_pretrained_slope_and_bias_vectors()

        # Learning attributes
        self._y_target: np.ndarray = np.zeros(shape=vector_size) # Target score for the model
        self._learning_rate = 0.01  # Learning rate for updates

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(judge={self._judge}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(judge={self._judge}, "\
               f"slope_vector={self._slope_vector}, bias_vector={self._bias_vector}, "\
               f"y_target={self._y_target}, learning_rate={self._learning_rate})"

    def get_current_slope_and_bias_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        return self._slope_vector, self._bias_vector

    def _get_pretrained_slope_and_bias_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the slope and bias vectors from the pretrained model .npy files if they exist, otherwise initialize random values.
        """
        # Ensure the directory exists
        try:
            if not os.path.exists(self._vector_base_directory):
                os.makedirs(self._vector_base_directory, exist_ok=True)
                logging.info(f"Created vector directory: {self._vector_base_directory}")
            else:
                logging.info(f"Directory already exists: {self._vector_base_directory}")
        except OSError as e:
            logging.error(f"Error creating vector directory: {e}")

        # Define the file paths for the vectors
        slope_vector_file: str = f"{self._vector_base_directory}{self._pretrained_archetype}_slope.npy"
        bias_vector_file: str = f"{self._vector_base_directory}{self._pretrained_archetype}_bias.npy"

        # Load the vectors from the pretrained model
        try:
            # Check if the files exist
            if os.path.exists(slope_vector_file) and os.path.exists(bias_vector_file):
                slope_vector = np.load(slope_vector_file)
                bias_vector = np.load(bias_vector_file)
                logging.info(f"Loaded vectors from {slope_vector_file} and {bias_vector_file}")
            else: # If not, initialize random vectors
                slope_vector = np.random.rand(self._vector_size)
                bias_vector = np.random.rand(self._vector_size)
                logging.info("Initialized random vectors")
        # Handle any errors that occur
        except OSError as e:
            logging.error(f"Error loading vectors: {e}")
            slope_vector = np.random.rand(self._vector_size)
            bias_vector = np.random.rand(self._vector_size)
            logging.info("Initialized random vectors")

        return slope_vector, bias_vector

    def _save_vectors_to_npy(self, slope_vector: np.ndarray, bias_vector: np.ndarray) -> None:
        """
        Save the slope and bias vectors to .npy files.
        """
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

        try:
            # If training_mode is True, save the vectors to the pretrained model files
            logging.debug(f"=============SUPER TESTING - SAVE VECTORS===============")
            if self._training_mode:
                slope_file: str = f"{self._vector_base_directory}{self._pretrained_archetype}_slope.npy"
                bias_file: str = f"{self._vector_base_directory}{self._pretrained_archetype}_bias.npy"
                np.save(slope_file, slope_vector)
                np.save(bias_file, bias_vector)
                logging.info(f"Saved vectors to {slope_file} and {bias_file}")
                logging.debug(f"slope_vector saved: {slope_vector}")
                logging.debug(f"bias_vector saved: {bias_vector}")
            else: # Otherwise, save the vectors to the temporary model files
                tmp_slope_file = f"{tmp_directory}{self._pretrained_archetype}_slope_{self._judge.get_name()}-tmp.npy"
                tmp_bias_file = f"{tmp_directory}{self._pretrained_archetype}_bias_{self._judge.get_name()}-tmp.npy"
                np.save(tmp_slope_file, slope_vector)
                np.save(tmp_bias_file, bias_vector)
                logging.info(f"Saved vectors to {tmp_slope_file} and {tmp_bias_file}")
                logging.debug(f"slope_vector saved to tmp: {slope_vector}")
                logging.debug(f"bias_vector saved to tmp: {bias_vector}")
            logging.debug(f"=============SUPER TESTING - SAVE VECTORS===============")
        except OSError as e:
            logging.error(f"Error saving vectors: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    def reset_model(self) -> None:
        """
        Reset the model data and vectors.
        """
        logging.debug(f"Resetting winning and losing apple pairs")
        logging.debug(f"Old winning apple pairs: {self._winning_apples}")
        logging.debug(f"Old losing apple pairs: {self._losing_apples}")
        self._winning_apples = []
        self._losing_apples = []
        logging.debug(f"New winning apple pairs: {self._winning_apples}")
        logging.debug(f"New losing apple pairs: {self._losing_apples}")

        # Reset the slope and bias vectors
        logging.debug(f"Resetting current slope and bias vectors")
        logging.debug(f"Old slope vector: {self._slope_vector}")
        logging.debug(f"Old bias vector: {self._bias_vector}")
        self._slope_vector, self._bias_vector = self._get_pretrained_slope_and_bias_vectors()
        logging.debug(f"New slope vector: {self._slope_vector}")
        logging.debug(f"New bias vector: {self._bias_vector}")

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


    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple], train_on_losing_red_apples: bool) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        """
        raise NotImplementedError("Subclass must implement the 'choose_red_apple' method")

    def _calculate_preference_output(self, slope_vector: np.ndarray, x_vector: np.ndarray, bias_vector: np.ndarray) -> np.ndarray:
        """
        Caculates the y_vector preference output given a slope vector, x vector, and bias vector.
        """
        return np.multiply(slope_vector, x_vector) + bias_vector

    def _calculate_score(self, y_predict: np.ndarray, y_target: np.ndarray) -> float:
        """
        Calculates the Euclidean distance given the predicted preference output and the target preference output.
        """
        euclidean_distance = np.linalg.norm(y_predict - y_target)
        return float(euclidean_distance)

    def choose_winning_red_apple(self, green_apple: GreenApple, opponent_red_apples: list[dict[Agent, RedApple]]) -> dict[Agent, RedApple]:
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
        logging.debug(f"Old slope vector: {self._slope_vector}")
        logging.debug(f"Old bias vector: {self._bias_vector}")
        slope_vector, bias_vector = self.__linear_regression(x_vector_array, y_vector_array)

        # Save the updated slope and bias vectors to .npy files
        self._save_vectors_to_npy(slope_vector, bias_vector)

        # Update the slope and bias vectors locally
        self._slope_vector = slope_vector
        self._bias_vector = bias_vector

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple], train_on_losing_red_apples: bool) -> RedApple:
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
        This method applies the private linear regression methods to predict the winning red apple.
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

                # Load the slope and bias vectors from the trained archetype
                slope_pretrain, bias_pretrain = self._get_pretrained_slope_and_bias_vectors()
                logging.debug(f"slope_pretrain: {slope_pretrain}, bias_pretrain: {bias_pretrain}")

                # Calculate the target preference output
                y_target = self._calculate_preference_output(slope_pretrain, x_vector, bias_pretrain)
                logging.debug(f"y_target: {y_target}")

                # Evaluate the score using Euclidean distances
                score = self._calculate_score(y_target, y_target)
                logging.debug(f"Score: {score}")

                # Update the best score and red apple
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
        logging.debug(f"Old slope vector: {self._slope_vector}")
        logging.debug(f"Old bias vector: {self._bias_vector}")

        # TODO - Finish implementing the neural network training
        # # Calculate the target score
        # for pair in winning_apple_pairs_vectors:
        #     self._y_target = self.__forward_propagation(pair["green_apple_vector"], pair["red_apple_vector"])
        #     self.__back_propagation(pair["green_apple_vector"], pair["red_apple_vector"])

        # # Save the updated slope and bias vectors
        # logging.debug(f"Updated slope vector: {self._slope_vector}")
        # logging.debug(f"Updated bias vector: {self._bias_vector}")
        # self._save_vectors()

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple], train_on_losing_red_apples: bool) -> RedApple:
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

    def choose_winning_red_apple(self, green_apple: GreenApple, opponent_red_apples: list[dict[Agent, RedApple]]) -> dict[Agent, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        This method applies the private neural network methods to predict the winning red apple.
        """
        # Get the green and red apple vectors
        green_apple_vector = green_apple.get_adjective_vector()

        # Initialize variables to track the best choice
        winning_red_apple: dict[Agent, RedApple] | None = None
        best_score = -np.inf

        for red_apple_dict in opponent_red_apples:
            for _, red_apple in red_apple_dict.items():
                red_apple_vector = red_apple.get_noun_vector()

                # Check that the green and red vectors are not None
                if green_apple_vector is None:
                    raise ValueError("Green apple vector is None.")
                if red_apple_vector is None:
                    raise ValueError("Red apple vector is None.")

                # Calculate the predicted score
                predicted_score = self.__forward_propagation(green_apple_vector, red_apple_vector)

                # Evaluate the score difference using Euclidean distances
                score_difference = np.linalg.norm(predicted_score - self._y_target)

                if score_difference < best_score:
                    best_score = score_difference
                    winning_red_apple = red_apple_dict

                # # Calculate the score
                # score = self._calculate_score(green_apple_vector, red_apple_vector)

                # # Update the best score and red apple
                # if score > best_score:
                #     winning_red_apple = red_apple_dict
                #     best_score = score

        # Check if the winning red apple is None
        if winning_red_apple is None:
            raise ValueError("No winning red apple was chosen.")

        return winning_red_apple


# Define the mapping from user input to model type
model_type_mapping = {
    '1': LRModel,
    '2': NNModel
}


if __name__ == "__main__":
    pass
