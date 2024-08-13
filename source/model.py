# Description: AI model logic for use in the AI agents in the 'Apples to Apples' game.

# Standard Libraries
import os
import logging
import numpy as np
from typing import Callable

# Third-party Libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # Suppress TensorFlow logging
import keras.api._v2.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, ELU
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Local Modules
# if TYPE_CHECKING:
from source.apples import GreenApple, RedApple
from source.agent import Agent
from source.data_classes import ApplesInPlay, ChosenApples, ChosenAppleVectors, ChosenAppleVectorsExtra


class Model():
    """
    Base class for the AI models.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_archetype: str, training_mode: bool = False) -> None:
        # Initialize the model attributes
        self._vector_base_directory = "./agent_archetypes/"
        self._judge: Agent = judge # The judge to be modeled
        self._vector_size = vector_size
        self._pretrained_archetype: str = pretrained_archetype # The name of the pretrained model archetype (e.g., Literalist, Contrarian, Comedian)
        self._training_mode: bool = training_mode
        self._chosen_apples: list[ChosenApples] = []
        self._pretrained_vectors: list[ChosenAppleVectors | ChosenAppleVectorsExtra] = self._load_pretrained_vectors()
        logging.debug(f"self._pretrained_vectors: {self._pretrained_vectors}")

         # Initialize predicted slope vector and bias vectors
        self._slope_predict: np.ndarray = np.empty(self._vector_size)
        self._bias_predict: np.ndarray = np.empty(self._vector_size)

        # Learning attributes
        self._y_target: np.ndarray = np.empty(self._vector_size) # Target score for the model
        self._learning_rate = 0.01  # Learning rate for updates

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(judge={self._judge.get_name()}"

    def _format_vector_filepath(self, use_extra_vectors: bool = False) -> str:
        """
        Format the vector file path.
        """
        # Define the file path for the pretrained vectors
        filepath = f"{self._vector_base_directory}{self._pretrained_archetype}_vectors"

        # Add the extra vectors to the file path, if applicable
        if use_extra_vectors:
            filepath += "-extra"

        # If no in training_mode, save the vectors to the tmp directory
        if not self._training_mode:
            filepath += "-tmp"

        # Add the file extension
        filepath += ".npz"

        return filepath

    def _ensure_directory_exists(self, directory: str) -> None:
        """
        Ensure the directory exists, create it if it doesn't.
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logging.info(f"Created directory: {directory}")
            else:
                logging.info(f"Directory already exists: {directory}")
        except OSError as e:
            logging.error(f"Error creating directory: {e}")

    def _file_exists(self, filepath: str) -> bool:
        """
        Check if a file exists.
        """
        if os.path.isfile(filepath):
            logging.info(f"File exists: {filepath}")
            return True
        else:
            logging.warning(f"File does not exist: {filepath}")
            return False

    def _load_pretrained_vectors(self, use_extra_vectors: bool = False) -> list[ChosenAppleVectors | ChosenAppleVectorsExtra]:
        # Ensure the base directory exists
        self._ensure_directory_exists(self._vector_base_directory)

        # Define the file path for the pretrained vectors
        filepath = self._format_vector_filepath(use_extra_vectors)

        # Load the pretrained vectors from the .npz file
        try:
            loaded_data = np.load(filepath)
            data = []

            # Determine the number of ChosenAppleVectors objects
            num_objects = len([key for key in loaded_data.keys() if key.startswith('green_apple_vector_')])

            if use_extra_vectors:
                for i in range(num_objects):
                    green_apple_vector = loaded_data[f'green_apple_vector_{i}']
                    winning_red_apple_vector = loaded_data[f'winning_red_apple_vector_{i}']
                    losing_red_apple_vectors = loaded_data[f'losing_red_apple_vectors_{i}']
                    green_apple_vector_extra = loaded_data[f'green_apple_vector_extra_{i}']
                    winning_red_apple_vector_extra = loaded_data[f'winning_red_apple_vector_extra_{i}']
                    losing_red_apple_vectors_extra = loaded_data[f'losing_red_apple_vectors_extra_{i}']
                    data.append(ChosenAppleVectorsExtra(
                        green_apple_vector=green_apple_vector,
                        winning_red_apple_vector=winning_red_apple_vector,
                        losing_red_apple_vectors=losing_red_apple_vectors,
                        green_apple_vector_extra=green_apple_vector_extra,
                        winning_red_apple_vector_extra=winning_red_apple_vector_extra,
                        losing_red_apple_vectors_extra=losing_red_apple_vectors_extra
                    ))
            else:
                for i in range(num_objects):
                    green_apple_vector = loaded_data[f'green_apple_vector_{i}']
                    winning_red_apple_vector = loaded_data[f'winning_red_apple_vector_{i}']
                    losing_red_apple_vectors = loaded_data[f'losing_red_apple_vectors_{i}']
                    data.append(ChosenAppleVectors(
                        green_apple_vector=green_apple_vector,
                        winning_red_apple_vector=winning_red_apple_vector,
                        losing_red_apple_vectors=losing_red_apple_vectors
                    ))
            logging.info(f"Loaded vectors from {filepath}")
        except OSError as e:
            logging.error(f"Error loading vectors: {e}")
            data = []

        return data

    def _save_chosen_apple_vectors(self, chosen_apple_vectors: list[ChosenAppleVectors | ChosenAppleVectorsExtra], use_extra_vectors: bool = False) -> None:
        """
        Load the pretrained vectors from the .npz file.
        The vectors include: green apple vectors, winning red apple vectors, and losing red apple vectors.
        """
        # Ensure the base directory exists
        self._ensure_directory_exists(self._vector_base_directory)

        # Ensure the tmp directory exists, if not in training mode
        if not self._training_mode:
            self._ensure_directory_exists(self._vector_base_directory + "tmp/")

        # Define the file path for the pretrained vectors
        filepath = self._format_vector_filepath(use_extra_vectors)

        # Load the vectors from the pretrained model
        try:
            # If the file exists, load the previous pretrained vectors
            if self._file_exists(filepath):
                existing_pretrained_vectors = self._load_pretrained_vectors()
                existing_pretrained_vectors.extend(chosen_apple_vectors) # Keep the order between the existing and new vectors
                chosen_apple_vectors = existing_pretrained_vectors

            # Create a dictionary to store each ChosenAppleVectors object
            data_dict: dict[str, np.ndarray] = {}
            if use_extra_vectors:
                for i, item in enumerate(chosen_apple_vectors):
                    # Verify the item is a ChosenAppleVectorsExtra object
                    if not isinstance(item, ChosenAppleVectorsExtra):
                        logging.error(f"Item is not a ChosenAppleVectorsExtra object.")
                        raise ValueError("Item is not a ChosenAppleVectorsExtra object.")
                    data_dict[f'green_apple_vector_{i}'] = item.green_apple_vector
                    data_dict[f'winning_red_apple_vector_{i}'] = item.winning_red_apple_vector
                    data_dict[f'losing_red_apple_vectors_{i}'] = item.losing_red_apple_vectors
                    data_dict[f'green_apple_vector_extra_{i}'] = item.green_apple_vector_extra
                    data_dict[f'winning_red_apple_vector_extra_{i}'] = item.winning_red_apple_vector_extra
                    data_dict[f'losing_red_apple_vectors_extra_{i}'] = item.losing_red_apple_vectors_extra
            else:
                for i, item in enumerate(chosen_apple_vectors):
                    # Verify the item is a ChosenAppleVectors object
                    if not isinstance(item, ChosenAppleVectors):
                        logging.error(f"Item is not a ChosenAppleVectors object.")
                        raise ValueError("Item is not a ChosenAppleVectors object.")
                    data_dict[f'green_apple_vector_{i}'] = item.green_apple_vector
                    data_dict[f'winning_red_apple_vector_{i}'] = item.winning_red_apple_vector
                    data_dict[f'losing_red_apple_vectors_{i}'] = item.losing_red_apple_vectors

            # Save to .npz file
            np.savez(filepath, **data_dict)
            logging.info(f"Saved vectors to {filepath}")
        # Handle any errors that occur
        except OSError as e:
            logging.error(f"Error saving vectors: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    def reset_model(self) -> None:
        """
        Reset the model data and vectors.
        """
        # TODO - revisit this, might have to rewrite, perhaps reset with random values?
        self._pretrained_vectors = self._load_pretrained_vectors()
        logging.debug(f"Reset the model data and vectors.")

    # def _append_new_winning_losing_apples(self, green_apple: GreenApple, winning_red_apple: RedApple, losing_red_apples: list[RedApple]) -> None:
    #     """
    #     Append the new winning and losing green and red apple pairs.
    #     """
    #     # Append the winning apple pair
    #     winning_apple_pair: dict[GreenApple, RedApple] = {green_apple: winning_red_apple}
    #     self._winning_apples.append(winning_apple_pair)

    #     # Append the losing apple pairs
    #     for red_apple in losing_red_apples:
    #         losing_apple_pair: dict[GreenApple, RedApple] = {green_apple: red_apple}
    #         self._losing_apples.append(losing_apple_pair)

    def _calculate_x_vector(self, green_apple_vector: np.ndarray, red_apple_vector: np.ndarray) -> np.ndarray:
        """
        Calculate and return x vector, which is the product of the green and red apple vectors.
        """
        # Calculate the x vector (product of green and red vectors)
        x_vector: np.ndarray = np.multiply(green_apple_vector, red_apple_vector)
        logging.debug(f"x_vector: {x_vector}")
        return x_vector

    def _calculate_x_vector_from_apples(self, green_apple: GreenApple, red_apple: RedApple, use_extra_vectors: bool) -> np.ndarray:
        """
        Calculate and return the new x vector, which is the product of the green and red apple vectors.
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
        logging.debug(f"x_vector: {x_vector}")

        # Include the extra vectors, if applicable
        if use_extra_vectors:
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
            logging.debug(f"x_vector_extra: {x_vector_extra}")

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

        # Create an ndarray filled with y values
        y_vectors = np.full(x_vectors.shape, y_value)

        logging.debug(f"x_vectors shape: {x_vectors.shape}")
        logging.debug(f"y_vectors shape: {y_vectors.shape}")
        logging.debug(f"x_vectors: {x_vectors}")
        logging.debug(f"y_vectors: {y_vectors}")

        # Ensure the x and y target arrays have the same dimensions
        assert x_vectors.shape == y_vectors.shape, "Vector dimensions do not match"

        return y_vectors

    def _extract_pretrained_slope_bias(self, model_function: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]], use_losing_red_apples: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper function to extract the x, y, slope, and bias vectors from the pretrained data.
        """
        # Extract the target winning apple vectors
        green_apple_target: np.ndarray = np.empty(self._vector_size)
        red_apple_target: np.ndarray = np.empty(self._vector_size)
        logging.debug(f"self._pretrained_vectors: {self._pretrained_vectors}")
        for pretrained_apples in self._pretrained_vectors:
            green_apple_target = np.vstack([green_apple_target, pretrained_apples.green_apple_vector])
            red_apple_target = np.vstack([red_apple_target, pretrained_apples.winning_red_apple_vector])

        # Calculate the winning x_target vectors
        x_target = self._calculate_x_vector(green_apple_target, red_apple_target)

        # Initialize the winning y_target vectors
        y_target = self._initialize_y_vectors(x_target, winning_apple=True)

        # Process the losing apple pairs, if applicable
        if use_losing_red_apples:
            for pretrained_apples in self._pretrained_vectors:
                for losing_red_apple in pretrained_apples.losing_red_apple_vectors:
                    # Calculate the x vectors for the losing apple pairs
                    x_target = np.vstack([x_target, self._calculate_x_vector(pretrained_apples.green_apple_vector, losing_red_apple)])
            # Initialize the losing y_target vectors
            y_target = np.vstack([y_target, self._initialize_y_vectors(x_target, winning_apple=False)])

        # Use linear regression or neural network function to calculate the target slope and bias vectors
        slope_target, bias_target = model_function(x_target, y_target)

        # Check if all elements in the slope_target are NaN
        all_nan = np.all(np.isnan(slope_target))

        # If all elements in the array are NaN, initialize the target slope to zero
        if all_nan:
            logging.debug("All elements in the slope_target are NaN.")
            slope_target = np.zeros(self._vector_size)
            logging.debug("Initialized the target slope and bias vectors to zero.")

        # Check if all elements in the bias_target are NaN
        all_nan = np.all(np.isnan(bias_target))

        # If all elements in the array are NaN, initialize the target bias to zero
        if all_nan:
            logging.debug("All elements in the bias_target are NaN.")
            bias_target = np.zeros(self._vector_size)
            logging.debug("Initialized the target slope and bias vectors to zero.")

        return slope_target, bias_target

    def train_model(self, chosen_apples: ChosenApples, use_extra_vectors: bool, use_losing_red_apples: bool) -> None:
        """
        Train the model using pairs of green and red apple vectors.
        """
        raise NotImplementedError("Subclass must implement the 'train_model' method")

    def get_current_slope_and_bias_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current slope and bias vectors.
        """
        return self._slope_predict, self._bias_predict

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple], use_extra_vectors: bool = False, use_losing_red_apples: bool = False) -> RedApple:
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
        slope_target (np.ndarray): Target slope vectors.
        bias_target (np.ndarray): Target bias vectors.
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

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay, use_extra_vectors: bool = False, use_losing_red_apples: bool = False) -> dict[Agent, RedApple]:
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
        logging.debug(f"x_vector_array shape: {x_vector_array.shape}")
        logging.debug(f"y_vector_array shape: {y_vector_array.shape}")
        # Ensure the x and y target arrays have the same dimensions
        assert x_vector_array.shape == y_vector_array.shape, "Vector dimensions do not match"

        # Determine the number of vectors
        n: int = x_vector_array.shape[0]

        # Initalize the sum variables
        sumx: np.ndarray = np.empty(self._vector_size)
        sumx2: np.ndarray = np.empty(self._vector_size)
        sumxy: np.ndarray = np.empty(self._vector_size)
        sumy: np.ndarray = np.empty(self._vector_size)
        sumy2: np.ndarray = np.empty(self._vector_size)

        # Calculate the sums for the x and y vectors
        for x, y in zip(x_vector_array, y_vector_array):
            sumx = np.add(sumx, x)
            sumx2 = np.add(sumx2, np.multiply(x, x))
            sumxy = np.add(sumxy, np.multiply(x, y))
            sumy = np.add(sumy, y)
            sumy2 = np.add(sumy2, np.multiply(y, y))

        logging.debug(f"Final sums - sumx:{sumx}, sumx2:{sumx2}, sumxy:{sumxy}, sumy:{sumy}, sumy2:{sumy2}")

        # Calculate the denominators
        denoms: np.ndarray = np.full(self._vector_size, n) * sumx2 - np.multiply(sumx, sumx)

        logging.debug(f"denoms: {denoms}")

        # Initialize the slope and intercept elements to zero
        m: np.ndarray = np.empty(self._vector_size)
        b: np.ndarray = np.empty(self._vector_size)

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

    def train_model(self, chosen_apples: ChosenApples, use_extra_vectors: bool, use_losing_red_apples: bool) -> None:
        """
        Train the model using winning green and red apple pairs and losing green and red apple pairs if applicable.
        """
        # Append the new chosen apples
        self._chosen_apples.append(chosen_apples)

        # Extract the chosen green apple vector
        green_apple_vector: np.ndarray | None = chosen_apples.get_green_apple().get_adjective_vector()
        # Check that the green apple vector is not None
        if green_apple_vector is None:
            logging.error(f"Green apple vector is None.")
            raise ValueError("Green apple vector is None.")

        # Extract the winning red apple vector
        winning_red_apple_vector: np.ndarray | None = chosen_apples.get_winning_red_apple().get_noun_vector()
        # Check that the winning red apple vector is not None
        if winning_red_apple_vector is None:
            logging.error(f"Winning red apple vector is None.")
            raise ValueError("Winning red apple vector is None.")

        # Initialize the losing red apple vectors
        losing_red_apple_vectors: np.ndarray = np.empty(self._vector_size)

        # Initialize the extra vectors if applicable
        if use_extra_vectors:
            # Extract the extra green apple vector
            green_apple_vector_extra: np.ndarray | None = chosen_apples.get_green_apple().get_synonyms_vector()
            # Check that the green apple vector is not None
            if green_apple_vector_extra is None:
                logging.error(f"Green apple vector is None.")
                raise ValueError("Green apple vector is None.")

            # Extract the extra winning red apple vector
            winning_red_apple_vector_extra: np.ndarray | None = chosen_apples.get_winning_red_apple().get_description_vector()
            # Check that the winning red apple vector is not None
            if winning_red_apple_vector_extra is None:
                logging.error(f"Winning red apple vector is None.")
                raise ValueError("Winning red apple vector is None.")

            # Initialize the extra losing red apple vector
            losing_red_apple_vectors_extra: np.ndarray = np.empty(self._vector_size)

        # Get the losing red apple vectors and extra vectors if applicable
        for losing_red_apple in chosen_apples.get_losing_red_apples():
            noun: np.ndarray | None = losing_red_apple.get_noun_vector()
            # Check that the noun vector is not None
            if noun is None:
                logging.error(f"Noun vector is None.")
                raise ValueError("Noun vector is None.")
            losing_red_apple_vectors = np.vstack([losing_red_apple_vectors, noun])
            if use_extra_vectors:
                description: np.ndarray | None = losing_red_apple.get_description_vector()
                # Check that the description vector is not None
                if description is None:
                    logging.error(f"Description vector is None.")
                    raise ValueError("Description vector is None.")
                losing_red_apple_vectors_extra = np.vstack([losing_red_apple_vectors_extra, description])

        # Create the chosen apple vectors
        if use_extra_vectors:
            chosen_apple_vectors_extra: ChosenAppleVectorsExtra = ChosenAppleVectorsExtra(
                green_apple_vector=green_apple_vector,
                winning_red_apple_vector=winning_red_apple_vector,
                losing_red_apple_vectors=losing_red_apple_vectors,
                green_apple_vector_extra=green_apple_vector_extra,
                winning_red_apple_vector_extra=winning_red_apple_vector_extra,
                losing_red_apple_vectors_extra=losing_red_apple_vectors_extra
            )
        else:
            chosen_apple_vectors: ChosenAppleVectors = ChosenAppleVectors(
                green_apple_vector=green_apple_vector,
                winning_red_apple_vector=winning_red_apple_vector,
                losing_red_apple_vectors=losing_red_apple_vectors
            )

        # Save the chosen apple vectors to .npz file
        if use_extra_vectors:
            self._save_chosen_apple_vectors([chosen_apple_vectors_extra], use_extra_vectors)
        else:
            self._save_chosen_apple_vectors([chosen_apple_vectors], use_extra_vectors)
        logging.info(f"Trained the model using the chosen apple vectors.")

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple], use_extra_vectors: bool = False, use_losing_red_apples: bool = False) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        This method applies the private linear regression methods to predict the best red apple.
        """
        # Initialize the best score and best red apple
        best_red_apple: RedApple | None = None
        best_score: float = -np.inf

        # Extract the target x, y, slope, and bias vectors
        slope_target, bias_target = self._extract_pretrained_slope_bias(self.__linear_regression, use_losing_red_apples)
        logging.debug(f"slope_target: {slope_target}")
        logging.debug(f"bias_target: {bias_target}")

        # Iterate through the red apples to find the best one
        for red_apple in red_apples_in_hand:
            # Calculate the winning x_predict vector
            x_predict: np.ndarray = self._calculate_x_vector_from_apples(green_apple, red_apple, use_extra_vectors)
            logging.debug(f"x_vector: {x_predict}")

            # Initialize the winning y_predict vector
            y_predict = self._initialize_y_vectors(x_predict, winning_apple=True)
            logging.debug(f"y_vector: {y_predict}")

            # Use linear regression to predict the preference output
            self._slope_predict, self._bias_predict = self.__linear_regression(x_predict, y_predict)
            logging.debug(f"self._slope_predict: {self._slope_predict}")
            logging.debug(f"self._bias_predict: {self._bias_predict}")

            # Evaluate the score using RMSE
            score_mse = self._calculate_score(self._slope_predict, self._bias_predict, slope_target, bias_target)
            logging.debug(f"score_mse: {score_mse}")

            # Evaluate the score using Euclidean distance
            score_euclid = self._calculate_score(self._slope_predict, self._bias_predict, slope_target, bias_target, True)
            logging.debug(f"score_euclid: {score_euclid}")

            # Choose which score to use
            score = score_mse
            # score = score_euclid

            # Update the best score and accompanying red apple
            if score < best_score:
                best_score = score
                best_red_apple = red_apple
                logging.debug(f"New best score: {best_score}")
                logging.debug(f"New best red apple: {best_red_apple}")

        # Check if the best red apple was chosen
        if best_red_apple is None:
            raise ValueError("No red apple was chosen.")

        return best_red_apple

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay, use_extra_vectors: bool = False, use_losing_red_apples: bool = False) -> dict[Agent, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        This method is only used by the self model and applies the private linear regression methods to predict the winning red apple.
        """
        # Initialize variables to track the best choice
        winning_red_apple: dict[Agent, RedApple] | None = None
        best_score = np.inf

        # Extract the target x, y, slope, and bias vectors
        slope_target, bias_target = self._extract_pretrained_slope_bias(self.__linear_regression, use_losing_red_apples)
        logging.debug(f"slope_target: {slope_target}")
        logging.debug(f"bias_target: {bias_target}")

        # Get the green apple vector from apples in play, if applicable
        if use_losing_red_apples:
            green_apple_vectors: np.ndarray | None = apples_in_play.get_green_apple().get_adjective_vector()

            # Check that the green apple vectors is not None
            if green_apple_vectors is None:
                logging.error("Green apple vector is None.")
                raise ValueError("Green apple vector is None.")

            if use_extra_vectors:
                green_apple_vector_extra: np.ndarray | None = apples_in_play.get_green_apple().get_synonyms_vector()

                # Check that the green apple vector is not None
                if green_apple_vector_extra is None:
                    logging.error("Green apple vector is None.")
                    raise ValueError("Green apple vector is None.")

                # Append the green apple vector to the array
                green_apple_vectors = np.vstack([green_apple_vectors, green_apple_vector_extra])

        # Iterate through the red apples to find the best one
        for red_apple in apples_in_play.red_apples:
            # Calculate the winning x_predict vector
            x_predict: np.ndarray = self._calculate_x_vector_from_apples(apples_in_play.get_green_apple(), list(red_apple.values())[0], use_extra_vectors)
            logging.debug(f"x_vector: {x_predict}")

            # Initialize the winning y_predict vector
            y_predict = self._initialize_y_vectors(x_predict, winning_apple=True)
            logging.debug(f"y_vector: {y_predict}")

            # Process the losing apple pairs, if applicable
            if use_losing_red_apples:
                for vector in self._pretrained_vectors:
                    for losing_red_apple_vector in vector.losing_red_apple_vectors:
                        for green_apple_vector in green_apple_vectors:
                            # Calculate the x vectors for the losing apple pairs
                            x_predict = np.vstack([x_predict, self._calculate_x_vector(green_apple_vector, losing_red_apple_vector)])

                # Initialize the losing y_predict vectors
                y_predict = np.vstack([y_predict, self._initialize_y_vectors(x_predict, winning_apple=False)])

            # Use linear regression to predict the preference output
            self._slope_predict, self._bias_predict = self.__linear_regression(x_predict, y_predict)
            logging.debug(f"self._slope_predict: {self._slope_predict}")
            logging.debug(f"self._bias_predict: {self._bias_predict}")

            # Evaluate the score using RMSE
            score_mse = self._calculate_score(self._slope_predict, self._bias_predict, slope_target, bias_target)
            logging.debug(f"score_mse: {score_mse}")

            # Evaluate the score using Euclidean distance
            score_euclid = self._calculate_score(self._slope_predict, self._bias_predict, slope_target, bias_target, True)
            logging.debug(f"score_euclid: {score_euclid}")

            # Choose which score to use
            score = score_mse
            # score = score_euclid

            # Update the best score and accompanying red apple
            if score < best_score:
                best_score = score
                winning_red_apple = red_apple
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

    def __forward_propagation(self, x_vector_array: np.ndarray, y_vector_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward propagation algorithm for the AI agent.
        """
        # TODO - FIX THIS METHOD
        # y = mx + b, where x is the product of green and red apple vectors
        x: np.ndarray = np.multiply(x_vector_array, y_vector_array)
        # y_pred = np.multiply(self._slope_vector, x) + self._bias_vector
        # return y_pred
        prediction = self.model.predict(x)

        slope = prediction[:self._vector_size]
        bias = prediction[self._vector_size:]

        return slope, bias

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
        x: np.ndarray = np.multiply(green_apple_vector, red_apple_vector)
        self.model.train_on_batch(x, self._y_target)

    def train_model(self, chosen_apples: ChosenApples, use_extra_vectors: bool, use_losing_red_apples: bool) -> None:
        """
        Train the model using winning green and red apple pairs and losing green and red apple pairs if applicable.
        """
        # Append the new chosen apples
        self._chosen_apples.append(chosen_apples)

        # Extract the chosen green apple vector
        green_apple_vector: np.ndarray | None = chosen_apples.get_green_apple().get_adjective_vector()
        # Check that the green apple vector is not None
        if green_apple_vector is None:
            logging.error(f"Green apple vector is None.")
            raise ValueError("Green apple vector is None.")

        # Extract the winning red apple vector
        winning_red_apple_vector: np.ndarray | None = chosen_apples.get_winning_red_apple().get_noun_vector()
        # Check that the winning red apple vector is not None
        if winning_red_apple_vector is None:
            logging.error(f"Winning red apple vector is None.")
            raise ValueError("Winning red apple vector is None.")

        # Initialize the losing red apple vectors
        losing_red_apple_vectors: np.ndarray = np.empty(self._vector_size)

        # Initialize the extra vectors if applicable
        if use_extra_vectors:
            # Extract the extra green apple vector
            green_apple_vector_extra: np.ndarray | None = chosen_apples.get_green_apple().get_synonyms_vector()
            # Check that the green apple vector is not None
            if green_apple_vector_extra is None:
                logging.error(f"Green apple vector is None.")
                raise ValueError("Green apple vector is None.")

            # Extract the extra winning red apple vector
            winning_red_apple_vector_extra: np.ndarray | None = chosen_apples.get_winning_red_apple().get_description_vector()
            # Check that the winning red apple vector is not None
            if winning_red_apple_vector_extra is None:
                logging.error(f"Winning red apple vector is None.")
                raise ValueError("Winning red apple vector is None.")

            # Initialize the extra losing red apple vector
            losing_red_apple_vectors_extra: np.ndarray = np.empty(self._vector_size)

        # Get the losing red apple vectors and extra vectors if applicable
        for losing_red_apple in chosen_apples.get_losing_red_apples():
            noun: np.ndarray | None = losing_red_apple.get_noun_vector()
            # Check that the noun vector is not None
            if noun is None:
                logging.error(f"Noun vector is None.")
                raise ValueError("Noun vector is None.")
            losing_red_apple_vectors = np.vstack([losing_red_apple_vectors, noun])
            if use_extra_vectors:
                description: np.ndarray | None = losing_red_apple.get_description_vector()
                # Check that the description vector is not None
                if description is None:
                    logging.error(f"Description vector is None.")
                    raise ValueError("Description vector is None.")
                losing_red_apple_vectors_extra = np.vstack([losing_red_apple_vectors_extra, description])

        # Create the chosen apple vectors
        if use_extra_vectors:
            chosen_apple_vectors_extra: ChosenAppleVectorsExtra = ChosenAppleVectorsExtra(
                green_apple_vector=green_apple_vector,
                winning_red_apple_vector=winning_red_apple_vector,
                losing_red_apple_vectors=losing_red_apple_vectors,
                green_apple_vector_extra=green_apple_vector_extra,
                winning_red_apple_vector_extra=winning_red_apple_vector_extra,
                losing_red_apple_vectors_extra=losing_red_apple_vectors_extra
            )
        else:
            chosen_apple_vectors: ChosenAppleVectors = ChosenAppleVectors(
                green_apple_vector=green_apple_vector,
                winning_red_apple_vector=winning_red_apple_vector,
                losing_red_apple_vectors=losing_red_apple_vectors
            )

        # Save the chosen apple vectors to .npz file
        if use_extra_vectors:
            self._save_chosen_apple_vectors([chosen_apple_vectors_extra], use_extra_vectors)
        else:
            self._save_chosen_apple_vectors([chosen_apple_vectors], use_extra_vectors)
        logging.info(f"Trained the model using the chosen apple vectors.")

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple], use_extra_vectors: bool = False, use_losing_red_apples: bool = False) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        This method applies the private neural network methods to predict the best red apple.
        """
        # Initialize the best score and best red apple
        best_red_apple: RedApple | None = None
        best_score: float = -np.inf

        # Extract the target x, y, slope, and bias vectors
        slope_target, bias_target = self._extract_pretrained_slope_bias(self.__forward_propagation, use_losing_red_apples)
        logging.debug(f"slope_target: {slope_target}")
        logging.debug(f"bias_target: {bias_target}")

        # Iterate through the red apples to find the best one
        for red_apple in red_apples_in_hand:
            # Calculate the winning x_predict vector
            x_predict: np.ndarray = self._calculate_x_vector_from_apples(green_apple, red_apple, use_extra_vectors)
            logging.debug(f"x_vector: {x_predict}")

            # Initialize the winning y_predict vector
            y_predict = self._initialize_y_vectors(x_predict, winning_apple=True)
            logging.debug(f"y_vector: {y_predict}")

            # Use forward propogation to predict the preference output
            self._slope_predict, self._bias_predict = self.__forward_propagation(x_predict, y_predict)
            logging.debug(f"self._slope_predict: {self._slope_predict}")
            logging.debug(f"self._bias_predict: {self._bias_predict}")

            # Evaluate the score using RMSE
            score_mse = self._calculate_score(self._slope_predict, self._bias_predict, slope_target, bias_target)
            logging.debug(f"score_mse: {score_mse}")

            # Evaluate the score using Euclidean distance
            score_euclid = self._calculate_score(self._slope_predict, self._bias_predict, slope_target, bias_target, True)
            logging.debug(f"score_euclid: {score_euclid}")

            # Choose which score to use
            score = score_mse
            # score = score_euclid

            # Update the best score and accompanying red apple
            if score < best_score:
                best_score = score
                best_red_apple = red_apple
                logging.debug(f"New best score: {best_score}")
                logging.debug(f"New best red apple: {best_red_apple}")

        # Check if the best red apple was chosen
        if best_red_apple is None:
            raise ValueError("No red apple was chosen.")

        return best_red_apple

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay, use_extra_vectors: bool = False, use_losing_red_apples: bool = False) -> dict[Agent, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        This method applies the private neural network methods to predict the winning red apple.
        """
        # Initialize variables to track the best choice
        winning_red_apple: dict[Agent, RedApple] | None = None
        best_score = np.inf

        # Extract the target x, y, slope, and bias vectors
        slope_target, bias_target = self._extract_pretrained_slope_bias(self.__forward_propagation, use_losing_red_apples)
        logging.debug(f"slope_target: {slope_target}")
        logging.debug(f"bias_target: {bias_target}")

        # Get the green apple vector from apples in play, if applicable
        if use_losing_red_apples:
            green_apple_vector = apples_in_play.get_green_apple().get_adjective_vector()

            # Check that the green apple vector is not None
            if green_apple_vector is None:
                logging.error("Green apple vector is None.")
                raise ValueError("Green apple vector is None.")

        # Iterate through the red apples to find the best one
        for red_apple in apples_in_play.red_apples:
            # Calculate the winning x_predict vector
            x_predict: np.ndarray = self._calculate_x_vector_from_apples(apples_in_play.get_green_apple(), list(red_apple.values())[0], use_extra_vectors)
            logging.debug(f"x_vector: {x_predict}")

            # Initialize the winning y_predict vector
            y_predict = self._initialize_y_vectors(x_predict, winning_apple=True)
            logging.debug(f"y_vector: {y_predict}")

            # Process the losing apple pairs, if applicable
            if use_losing_red_apples:
                for vector in self._pretrained_vectors:
                    for losing_red_apple_vector in vector.losing_red_apple_vectors:
                        # Calculate the x vectors for the losing apple pairs
                        x_predict = np.vstack([x_predict, self._calculate_x_vector(green_apple_vector, losing_red_apple_vector)])

                # Initialize the losing y_predict vectors
                y_predict = np.vstack([y_predict, self._initialize_y_vectors(x_predict, winning_apple=False)])

            # Use linear regression to predict the preference output
            self._slope_predict, self._bias_predict = self.__forward_propagation(x_predict, y_predict)
            logging.debug(f"self._slope_predict: {self._slope_predict}")
            logging.debug(f"self._bias_predict: {self._bias_predict}")

            # Evaluate the score using RMSE
            score_mse = self._calculate_score(self._slope_predict, self._bias_predict, slope_target, bias_target)
            logging.debug(f"score_mse: {score_mse}")

            # Evaluate the score using Euclidean distance
            score_euclid = self._calculate_score(self._slope_predict, self._bias_predict, slope_target, bias_target, True)
            logging.debug(f"score_euclid: {score_euclid}")

            # Choose which score to use
            score = score_mse
            # score = score_euclid

            # Update the best score and accompanying red apple
            if score < best_score:
                best_score = score
                winning_red_apple = red_apple
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
