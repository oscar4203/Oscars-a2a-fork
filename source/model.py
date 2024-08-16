# Description: AI model logic for use in the AI agents in the 'Apples to Apples' game.

# Standard Libraries
import logging
import os
import numpy as np
from typing import Callable
import re

# Third-party Libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # Suppress TensorFlow logging
import keras.api._v2.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, ELU
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from scipy import stats

# Local Modules
# if TYPE_CHECKING:
from source.apples import GreenApple, RedApple
from source.agent import Agent
from source.data_classes import ApplesInPlay, ChosenApples, ChosenAppleVectors, ChosenAppleVectorsExtra


class Model():
    """
    Base class for the AI models.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_archetype: str, use_extra_vectors: bool = False, use_losing_red_apples : bool = False, training_mode: bool = False) -> None:
        # Initialize the model attributes
        self._vector_base_directory = "./agent_archetypes/"
        self._judge: Agent = judge # The judge to be modeled
        self._vector_size = vector_size
        self._pretrained_archetype: str = pretrained_archetype # The name of the pretrained model archetype (e.g., Literalist, Contrarian, Comedian)
        self._use_extra_vectors: bool = use_extra_vectors
        self._use_losing_red_apples: bool = use_losing_red_apples
        self._training_mode: bool = training_mode

        # Load the pretrained vectors
        self._pretrained_vectors: list[ChosenAppleVectors | ChosenAppleVectorsExtra] = \
            self._load_vectors(self._format_vector_filepath(False))

        # Check that the pretrained vectors have at least 2 vectors
        if len(self._pretrained_vectors) < 2:
            message = f"Pretrained vectors must have at least 2 vectors."\
                f"\nPlease train the {self._pretrained_archetype} with 'extra_vectors' set to {self._use_extra_vectors}."
            logging.error(message)
            raise ValueError(message)

        # Initialize the chosen apples and vectors
        self._chosen_apples: list[ChosenApples] = []
        self._chosen_apple_vectors: list[ChosenAppleVectors | ChosenAppleVectorsExtra] = []

         # Initialize target and predict slope and bias vectors
        self._slope_predict: np.ndarray = np.zeros((0, self._vector_size))
        self._bias_predict: np.ndarray = np.zeros((0, self._vector_size))

        # Learning attributes
        self._y_target: np.ndarray = np.zeros((0, self._vector_size)) # Target score for the model
        self._learning_rate = 0.01  # Learning rate for updates

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(judge={self._judge.get_name()}"

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

    def _format_vector_filepath(self, tmp_vectors: bool = False) -> str:
        """
        Format the vector file path.
        """
        # Configure the directory, either for pretrained vectors or tmp vectors
        directory = self._vector_base_directory
        if tmp_vectors:
            tmp_directory = "tmp/"
            directory += tmp_directory

        # Ensure the formatted directory exists
        self._ensure_directory_exists(directory)

        # Define the filename for the vectors
        filename = f"{self._pretrained_archetype}_vectors"

        # Add the extra vectors to the filename, if applicable
        if self._use_extra_vectors:
            filename += "-extra"

        # If loading and saving tmp vectors, add "-tmp" to the filename
        if tmp_vectors:
            filename += "-tmp"

        # Add the file extension
        filename += ".npz"

        # Combine the directory and filename
        filepath = os.path.join(directory, filename)
        logging.debug(f"formatted filepath: {filepath}")

        return filepath

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

    def _load_vectors(self, filepath: str) -> list[ChosenAppleVectors | ChosenAppleVectorsExtra]:
        """
        Load the vectors from the .npz file.
        """
        # Check if the vectors file exists
        if not self._file_exists(filepath):
            logging.info(f"Vector file does not exist: {filepath}")
            return []

        # Load the vectors from the .npz file
        try:
            loaded_data = np.load(filepath)
            logging.debug(f"Loaded data keys: {list(loaded_data.keys())}")
            data = []

            # Compile regex pattern
            pattern = re.compile(r'green_apple_vector_\d+')

            # Determine the number of ChosenAppleVectors objects
            num_objects = len([key for key in loaded_data.keys() if pattern.match(key)])
            logging.debug(f"num_objects: {num_objects}")

            # Load the vectors from the .npz file
            if self._use_extra_vectors:
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
            logging.debug(f"Loaded 'data'. len(data): {len(data)}")
            logging.debug(f"'data': {data}")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            data = [] # Return an empty list if an error occurs
        except KeyError as e:
            logging.error(f"Key not found: {e}")
            data = [] # Return an empty list if an error occurs
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            data = []

        return data

    def _prepare_data_dict(self, chosen_apple_vectors: list[ChosenAppleVectors | ChosenAppleVectorsExtra]) -> dict[str, np.ndarray]:
        """
        Prepare the data dictionary for saving the chosen apple vectors to a .npz file.
        """
        # Create a dictionary to store each ChosenAppleVectors object
        data_dict: dict[str, np.ndarray] = {}
        if self._use_extra_vectors:
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

        return data_dict

    def _save_chosen_apple_vectors(self, chosen_apple_vectors: list[ChosenAppleVectors | ChosenAppleVectorsExtra], training_mode: bool = False) -> None:
        """
        Save the chosen apple vectors to a .npz file.
        The vectors include: green apple vectors, winning red apple vectors, and losing red apple vectors.
        """
        # Define the filepath for the vectors
        filepath = self._format_vector_filepath(False if training_mode else True)

        # Prepare the data dictionary
        data_dict = self._prepare_data_dict(chosen_apple_vectors)

        # Save the chosen apple vectors to a .npz file
        try:
            # Save to .npz file
            np.savez(filepath, **data_dict)
            logging.info(f"Saved vectors to {filepath}")
            logging.debug(f"Saved 'data_dict'. len(data_dict): {len(data_dict)}")
            logging.debug(f"'data_dict.keys()': {data_dict.keys()}")
        # Handle any errors that occur
        except OSError as e:
            logging.error(f"Error saving vectors: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    def get_current_slope_and_bias_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current slope and bias vectors.
        """
        raise NotImplementedError("Subclass must implement the 'get_current_slope_and_bias_vectors' method")

    def reset_model(self) -> None:
        """
        Reload the pretrained vectors and reset the model vectors.
        """
        # Reload the pretrained vectors
        self._pretrained_vectors = self._load_vectors(self._format_vector_filepath(False))

        # Reset the model vectors
        self._chosen_apple_vectors = []
        logging.info(f"Reset the pretrained vectors and model vectors..")

    def _collect_chosen_apple_vectors(self, chosen_apples: ChosenApples) -> ChosenAppleVectors | ChosenAppleVectorsExtra:
        """
        Collect the vectors from the chosen apples object, and store them in a ChosenAppleVectors or ChosenAppleVectorsExtra object.
        """
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

        # Initialize the extra vectors if applicable
        if self._use_extra_vectors:
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

        # Initialize the losing red apple vectors and extra vectors
        losing_red_apple_vectors: np.ndarray = np.zeros((0, self._vector_size))
        losing_red_apple_vectors_extra: np.ndarray = np.zeros((0, self._vector_size))

        # Get the losing red apple vectors and extra vectors if applicable
        for losing_red_apple in chosen_apples.get_losing_red_apples():
            noun_vector: np.ndarray | None = losing_red_apple.get_noun_vector()
            # Check that the noun vector is not None
            if noun_vector is None:
                logging.error(f"Noun vector is None.")
                raise ValueError("Noun vector is None.")

            # Append the noun vector to the losing red apple vectors
            losing_red_apple_vectors: np.ndarray = np.vstack([losing_red_apple_vectors, noun_vector])

            if self._use_extra_vectors:
                description_vector: np.ndarray | None = losing_red_apple.get_description_vector()
                # Check that the description_vector vector is not None
                if description_vector is None:
                    logging.error(f"Description vector is None.")
                    raise ValueError("Description vector is None.")

                # Append the description vector to the losing red apple vectors
                losing_red_apple_vectors_extra: np.ndarray = np.vstack([losing_red_apple_vectors_extra, description_vector])

        # Create the chosen apple vectors
        if self._use_extra_vectors:
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

        return chosen_apple_vectors_extra if self._use_extra_vectors else chosen_apple_vectors

    def _normalize_vectors(self, vector_array: np.ndarray) -> np.ndarray:
        """
        Normalize the input vectors using L2 (Euclidean Norm).
        This function is designed to normalize 2D arrays, where each row is a vector.
        If the input array is 1D, it will be reshaped to 2D prior to normalization, then reshaped back to 1D.
        """
        # Check if the vector array is 1D, if so reshape it to 2D
        if vector_array.ndim == 1:
            two_dim = False
            vector_array = vector_array.reshape(1, -1)
        elif vector_array.ndim > 2:
            logging.error(f"Vector array has more than 2 dimensions.")
            raise ValueError("Vector array has more than 2 dimensions.")
        else:
            two_dim = True

        # Axis=1 normalizes each row individually, keepdims=True keeps the dimensions of the array
        norms = np.linalg.norm(vector_array, axis=1, keepdims=True)

        # Calculate the mean of the norms, excluding zeros
        mean_norm = np.mean(norms[norms != 0])

        # Replace zero norms with the mean norm
        norms[norms == 0] = mean_norm

        # Normalize the vectors
        normalized_array = vector_array / norms

        # Reshape the array back to 1D if it was originally 1D
        if not two_dim:
            normalized_array = normalized_array.reshape(-1)

        return normalized_array

    def _calculate_x_vector(self, green_apple_vector: np.ndarray, red_apple_vector: np.ndarray) -> np.ndarray:
        """
        Calculate the x vector, which is the product of the green and red apple vectors.
        This method normalizes the x vector before returning it.
        """
        logging.debug(f"green_apple_vector: {green_apple_vector}")
        logging.debug(f"red_apple_vector: {red_apple_vector}")
        # Calculate the x vector (product of green and red vectors)
        x_vector: np.ndarray = np.multiply(green_apple_vector, red_apple_vector)
        logging.debug(f"x_vector: {x_vector}")

        return x_vector

    def _calculate_x_vector_from_apples(self, green_apple: GreenApple, red_apple: RedApple) -> np.ndarray:
        """
        Calculate and return the new x vector, which is the product of the green and red apple vectors.
        """
        # Get the green and red apple vectors
        green_apple_vector: np.ndarray | None = green_apple.get_adjective_vector()
        red_apple_vector: np.ndarray | None = red_apple.get_noun_vector()

        # Check that the green vector is not None
        if green_apple_vector is None:
            logging.error(f"Green apple vector is None.")
            raise ValueError("Green apple vector is None.")

        # Check that the red vector is not None
        if red_apple_vector is None:
            logging.error(f"Red apple vector is None.")
            raise ValueError("Red apple vector is None.")

        # Calculate the x vector (product of green and red vectors)
        x_vector: np.ndarray = self._calculate_x_vector(green_apple_vector, red_apple_vector)

        # Include the extra vectors, if applicable
        if self._use_extra_vectors:
            # Get the extra vectors
            green_vector_extra: np.ndarray | None = green_apple.get_synonyms_vector()
            red_vector_extra: np.ndarray | None = red_apple.get_description_vector()

            # Check that the green extra vector is not None
            if green_vector_extra is None:
                logging.error(f"Green apple vector is None.")
                raise ValueError("Green apple vector is None.")

            # Check that the red extra vector is not None
            if red_vector_extra is None:
                logging.error(f"Red apple vector is None.")
                raise ValueError("Red apple vector is None.")

            # Calculate the extra x vector (product of green and red extra vectors)
            x_vector_extra: np.ndarray = self._calculate_x_vector(green_vector_extra, red_vector_extra)

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
            y_value: int = 1
        else:
            y_value: int = -1

        # Create an ndarray filled with y values
        y_vectors = np.full(x_vectors.shape, y_value)

        logging.debug(f"x_vectors shape: {x_vectors.shape}")
        logging.debug(f"y_vectors shape: {y_vectors.shape}")
        logging.debug(f"x_vectors: {x_vectors}")
        logging.debug(f"y_vectors: {y_vectors}")

        # Ensure the x and y target arrays have the same dimensions
        assert x_vectors.shape == y_vectors.shape, "Vector dimensions do not match"

        return y_vectors

    def _calculate_x_and_y_vectors_from_chosen_apple_vectors(self, chosen_apple_vectors: list[ChosenAppleVectors | ChosenAppleVectorsExtra]) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the x and y vectors from the chosen apple vectors.
        """
        # Initialize the winning x_vectors
        x_vectors = np.zeros((0, self._vector_size))

        # Process the chosen apple vectors
        for chosen_apple in chosen_apple_vectors:
            # Calculate the x vectors for the winning apple pairs
            x_vectors = np.vstack([x_vectors, self._calculate_x_vector(chosen_apple.green_apple_vector, chosen_apple.winning_red_apple_vector)])

            # Include the extra vectors, if applicable
            if self._use_extra_vectors and isinstance(chosen_apple, ChosenAppleVectorsExtra):
                # Calculate the x vectors for the winning apple pairs
                x_vectors = np.vstack([x_vectors, self._calculate_x_vector(chosen_apple.green_apple_vector_extra, chosen_apple.winning_red_apple_vector_extra)])

        # Initialize the winning y_vectors
        y_vectors = self._initialize_y_vectors(x_vectors, winning_apple=True)

        # Process the losing apple pairs, if applicable
        if self._use_losing_red_apples:
            for chosen_apple in chosen_apple_vectors:
                for losing_red_apple in chosen_apple.losing_red_apple_vectors:
                    # Calculate the x vectors for the losing apple pairs
                    x_vectors = np.vstack([x_vectors, self._calculate_x_vector(chosen_apple.green_apple_vector, losing_red_apple)])

                    # Include the extra vectors, if applicable
                    if self._use_extra_vectors and isinstance(chosen_apple, ChosenAppleVectorsExtra):
                        # Calculate the x vectors for the losing apple pairs
                        x_vectors = np.vstack([x_vectors, self._calculate_x_vector(chosen_apple.green_apple_vector_extra, losing_red_apple)])

            y_vectors = np.vstack([y_vectors, self._initialize_y_vectors(x_vectors, winning_apple=False)])

        return x_vectors, y_vectors

    def _calculate_losing_x_and_y_vectors_from_green_apple(self, green_apple: GreenApple) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the x and y vectors from the losing red apple vectors.
        """
        # Extract the green apple vector
        green_apple_vector: np.ndarray | None = green_apple.get_adjective_vector()

        # Check that the green apple vector is not None
        if green_apple_vector is None:
            logging.error(f"Green apple vector is None.")
            raise ValueError("Green apple vector is None.")

        # Extract the extra green apple vector, if applicable
        if self._use_extra_vectors:
            green_apple_vector_extra: np.ndarray | None = green_apple.get_synonyms_vector()

            # Check that the green apple vector is not None
            if green_apple_vector_extra is None:
                logging.error(f"Green apple vector is None.")
                raise ValueError("Green apple vector is None.")

        # Initialize the losing x_vectors and y_vectors
        x_vectors = np.zeros((0, self._vector_size))
        y_vectors = np.zeros((0, self._vector_size))

        # Process the losing red apple vectors
        for vector in self._pretrained_vectors:
            for losing_red_apple in vector.losing_red_apple_vectors:
                # Calculate the x vectors for the losing apple pairs
                x_vectors = np.vstack([x_vectors, self._calculate_x_vector(green_apple_vector, losing_red_apple.get_noun_vector())])
                if self._use_extra_vectors:
                    x_vectors = np.vstack([x_vectors, self._calculate_x_vector(green_apple_vector_extra, losing_red_apple.get_description_vector())])

        # Initialize the losing y_vectors
        y_vectors = self._initialize_y_vectors(x_vectors, winning_apple=False)

        return x_vectors, y_vectors

    def _calculate_slope_and_bias_vectors(self, chosen_apple_vectors: list[ChosenAppleVectors | ChosenAppleVectorsExtra], model_function: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the slope and bias vectors from the pretrained data.
        """
        # Extract the winning apple vectors
        green_apple: np.ndarray = np.zeros((0, self._vector_size))
        red_apple_target: np.ndarray = np.zeros((0, self._vector_size))
        for chosen_apple in chosen_apple_vectors:
            green_apple = np.vstack([green_apple, chosen_apple.green_apple_vector])
            red_apple_target = np.vstack([red_apple_target, chosen_apple.winning_red_apple_vector])

        # Calculate the winning x_vectors
        x_vectors = self._calculate_x_vector(green_apple, red_apple_target)
        x = 10*np.random.rand(100, 1)

        # Initialize the winning y_vectors
        y_vectors = self._initialize_y_vectors(x_vectors, winning_apple=True)

        # Process the losing apple pairs, if applicable
        if self._use_losing_red_apples and len(chosen_apple_vectors) > 0:
            for chosen_apple in chosen_apple_vectors:
                for losing_red_apple in chosen_apple.losing_red_apple_vectors:
                    # Calculate the x vectors for the losing apple pairs
                    x_vectors = np.vstack([x_vectors, self._calculate_x_vector(chosen_apple.green_apple_vector, losing_red_apple)])
            y_vectors = np.vstack([y_vectors, self._initialize_y_vectors(x_vectors, winning_apple=False)])

        # Use linear regression or neural network function to calculate the slope and bias vectors
        slope, bias = model_function(x_vectors, y_vectors)

        # Check if all elements in the slope are NaN
        all_nan = np.all(np.isnan(slope))

        # If all elements in the array are NaN, initialize the slope to zero
        if all_nan:
            logging.debug("All elements in the slope are NaN.")
            slope = np.zeros(self._vector_size)
            logging.debug("Initialized the slope and bias vectors to zero.")

        # Check if all elements in the bias are NaN
        all_nan = np.all(np.isnan(bias))

        # If all elements in the array are NaN, initialize the bias to zero
        if all_nan:
            logging.debug("All elements in the bias are NaN.")
            bias = np.zeros(self._vector_size)
            logging.debug("Initialized the slope and bias vectors to zero.")

        return slope, bias

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
        logging.debug(f"slope_predict: {slope_predict}")
        logging.debug(f"bias_predict: {bias_predict}")
        logging.debug(f"slope_target: {slope_target}")
        logging.debug(f"bias_target: {bias_target}")

        # Check for NaN values in the vectors
        if np.isnan(slope_predict).any() or np.isnan(slope_target).any() or \
        np.isnan(bias_predict).any() or np.isnan(bias_target).any():
            logging.error("NaN values found in input vectors.")
            return float('nan')

        if use_euclidean:
            # Calculate the Euclidean distance for slope and bias
            euclidean_slope = np.linalg.norm(slope_predict - slope_target)
            euclidean_bias = np.linalg.norm(bias_predict - bias_target)

            logging.debug(f"euclidean_slope: {euclidean_slope}")
            logging.debug(f"euclidean_bias: {euclidean_bias}")

            # Combine the Euclidean distances for slope and bias
            total_distance = euclidean_slope + euclidean_bias
            logging.debug(f"total_distance: {total_distance}")

            return float(total_distance)
        else:
            # # Calculate the MSE for slope and bias
            # mse_slope = np.mean((slope_predict - slope_target) ** 2)
            # mse_bias = np.mean((bias_predict - bias_target) ** 2)

            # logging.debug(f"mse_slope: {mse_slope}")
            # logging.debug(f"mse_bias: {mse_bias}")

            # # Combine the MSE for slope and bias
            # mse_total = mse_slope + mse_bias
            # logging.debug(f"mse_total: {mse_total}")

            # return float(mse_total)

            # TODO - Decide whether to use MSE or MAE (less sensitive to outliers)
            # Calculate the MAE for slope and bias
            mae_slope = np.mean(np.abs(slope_predict - slope_target))
            mae_bias = np.mean(np.abs(bias_predict - bias_target))

            logging.debug(f"mae_slope: {mae_slope}")
            logging.debug(f"mae_bias: {mae_bias}")

            # Combine the MAE for slope and bias
            mae_total = mae_slope + mae_bias
            logging.debug(f"mae_total: {mae_total}")

            return float(mae_total)

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple]) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        """
        raise NotImplementedError("Subclass must implement the 'choose_red_apple' method")

    # def _calculate_y_output(self, slope_vector: np.ndarray, x_vector: np.ndarray, bias_vector: np.ndarray) -> np.ndarray:
    #     """
    #     Caculates the y_vector preference output given a slope vector, x vector, and bias vector.
    #     """
    #     return np.multiply(slope_vector, x_vector) + bias_vector

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict[Agent, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        """
        raise NotImplementedError("Subclass must implement the 'choose_winning_red_apple' method")

    def train_model(self, chosen_apples: ChosenApples) -> None:
        """
        Train the model using pairs of green and red apple vectors.
        """
        raise NotImplementedError("Subclass must implement the 'train_model' method")

class LRModel(Model):
    """
    Linear Regression model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_archetype: str, use_extra_vectors: bool = False, use_losing_red_apples : bool = False, training_mode: bool = False) -> None:
        super().__init__(judge, vector_size, pretrained_archetype, use_extra_vectors, use_losing_red_apples, training_mode)
        # Initialize the target slope and bias vectors, if not in training mode
        if not self._training_mode:
            self._slope_target, self._bias_target = self._calculate_slope_and_bias_vectors(self._pretrained_vectors, self.__linear_regression)

    def get_current_slope_and_bias_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current slope and bias vectors.
        """
        if self._training_mode:
            return self._slope_target, self._bias_target
        else:
            return self._slope_predict, self._bias_predict

    # def __linear_regression(self, x_vector_array: np.ndarray, y_vector_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Linear regression algorithm for the AI agent, which calculates the slope and bias vectors given an x_vector_array and y_vector_array.
    #     \nEquation: y = mx + b ===>>> where y is the predicted preference output, m is the slope vector, x is the product of green and red apple vectors, and b is the bias vector.
    #     """
    #     # Ensure there are no NaN or infinite values in the arrays
    #     assert not np.any(np.isnan(x_vector_array)), "x_vector_array contains NaNs"
    #     assert not np.any(np.isnan(y_vector_array)), "y_vector_array contains NaNs"
    #     assert not np.any(np.isinf(x_vector_array)), "x_vector_array contains infinite values"
    #     assert not np.any(np.isinf(y_vector_array)), "y_vector_array contains infinite values"

    #     # Ensure the x and y target arrays have the same dimensions
    #     logging.debug(f"x_vector_array shape: {x_vector_array.shape}")
    #     logging.debug(f"y_vector_array shape: {y_vector_array.shape}")
    #     assert x_vector_array.shape == y_vector_array.shape, "Vector dimensions do not match"

    #     # Reshape 1D arrays to 2D arrays
    #     if x_vector_array.ndim == 1 and y_vector_array.ndim == 1:
    #         logging.debug(f"Reshaping 1D arrays to 2D arrays.")
    #         x_vector_array = x_vector_array.reshape(1, -1)
    #         y_vector_array = y_vector_array.reshape(1, -1)
    #         logging.debug(f"x_vector_array shape after: {x_vector_array.shape}")
    #         logging.debug(f"y_vector_array shape after: {y_vector_array.shape}")
    #     elif x_vector_array.ndim == 2 and y_vector_array.ndim == 2:
    #         logging.debug(f"Arrays are already 2D.")
    #     else:
    #         error_message = f"Invalid dimensions for x and y vectors. x_vector_array.ndim: "\
    #             f"{x_vector_array.ndim}, y_vector_array.ndim: {y_vector_array.ndim}. "\
    #             f"Only 1D or 2D arrays are supported."
    #         logging.error(error_message)
    #         raise ValueError(error_message)

    #     # Ensure the x and y vector arrays have more than 1 row
    #     assert x_vector_array.shape[0] > 1, "x_vector_array must have more than 1 row"
    #     assert y_vector_array.shape[0] > 1, "y_vector_array must have more than 1 row"

    #     # Normalize the x and y vectors
    #     x_vector_array = self._normalize_vectors(x_vector_array)
    #     y_vector_array = self._normalize_vectors(y_vector_array)

    #     # Determine the number of columns in the x vector array
    #     num_columns = x_vector_array.shape[1]
    #     logging.debug(f"num_columns: {num_columns}")

    #     # Initialize arrays to store results
    #     slopes: np.ndarray = np.empty(num_columns)
    #     intercepts: np.ndarray = np.empty(num_columns)
    #     r_values: np.ndarray = np.empty(num_columns)
    #     p_values: np.ndarray = np.empty(num_columns)
    #     std_errs: np.ndarray = np.empty(num_columns)

    #     # Perform linear regression for each feature (column)
    #     for i in range(x_vector_array.shape[1]):
    #         # Select all rows for the ith column using [:, i] indexing
    #         slope, intercept, r, p, se = stats.linregress(x_vector_array[:, i], y_vector_array[:, i])
    #         slopes[i] = slope
    #         intercepts[i] = intercept
    #         r_values[i] = r
    #         p_values[i] = p
    #         std_errs[i] = se

    #     # Logging the results
    #     logging.debug(f"slopes: {slopes}")
    #     logging.debug(f"intercepts: {intercepts}")
    #     logging.debug(f"r_values: {r_values}")
    #     logging.debug(f"p_values: {p_values}")
    #     logging.debug(f"std_errs: {std_errs}")

    #     return slopes, intercepts

    def __linear_regression(self, x_vector_array: np.ndarray, y_vector_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Linear regression algorithm for the AI agent, which calculates the slope and bias vectors given an x_vector_array and y_vector_array.
        \nEquation: y = mx + b ===>>> where y is the predicted preference output, m is the slope vector, x is the product of green and red apple vectors, and b is the bias vector.
        """
        # Check for NaN and infinite values in the arrays
        assert not np.any(np.isnan(x_vector_array)), "x_vector_array contains NaNs"
        assert not np.any(np.isnan(y_vector_array)), "y_vector_array contains NaNs"
        assert not np.any(np.isinf(x_vector_array)), "x_vector_array contains infinite values"
        assert not np.any(np.isinf(y_vector_array)), "y_vector_array contains infinite values"

        # Ensure the x and y target arrays have the same dimensions
        logging.debug(f"x_vector_array shape: {x_vector_array.shape}")
        logging.debug(f"y_vector_array shape: {y_vector_array.shape}")
        assert x_vector_array.shape == y_vector_array.shape, "Vector dimensions do not match"

        # Reshape 1D arrays to 2D arrays
        if x_vector_array.ndim == 1 and y_vector_array.ndim == 1:
            logging.debug(f"Reshaping 1D arrays to 2D arrays.")
            x_vector_array = x_vector_array.reshape(1, -1)
            y_vector_array = y_vector_array.reshape(1, -1)
            logging.debug(f"x_vector_array shape: {x_vector_array.shape}")
            logging.debug(f"y_vector_array shape: {y_vector_array.shape}")
        elif x_vector_array.ndim == 2 and y_vector_array.ndim == 2:
            logging.debug(f"Arrays are already 2D.")
            logging.debug(f"x_vector_array shape: {x_vector_array.shape}")
            logging.debug(f"y_vector_array shape: {y_vector_array.shape}")
        else:
            error_message = f"Invalid dimensions for x and y vectors. x_vector_array.ndim: "\
                f"{x_vector_array.ndim}, y_vector_array.ndim: {y_vector_array.ndim}. "\
                f"Only 1D or 2D arrays are supported."
            logging.error(error_message)
            raise ValueError(error_message)

        # Normalize the x and y vectors
        x_vector_array = self._normalize_vectors(x_vector_array)
        y_vector_array = self._normalize_vectors(y_vector_array)

        # Determine the number of vectors
        n: int = x_vector_array.shape[0]
        logging.debug(f"n: {n}")

        # Initalize the sum variables to zero
        sumx: np.ndarray = np.zeros(self._vector_size)
        sumx2: np.ndarray = np.zeros(self._vector_size)
        sumxy: np.ndarray = np.zeros(self._vector_size)
        sumy: np.ndarray = np.zeros(self._vector_size)
        sumy2: np.ndarray = np.zeros(self._vector_size)

        # Iterate over each vector and sum the values
        for x_vector, y_vector in zip(x_vector_array, y_vector_array):
            sumx = np.add(sumx, x_vector)
            sumx2 = np.add(sumx2, np.multiply(x_vector, x_vector))
            sumxy = np.add(sumxy, np.multiply(x_vector, y_vector))
            sumy = np.add(sumy, y_vector)
            sumy2 = np.add(sumy2, np.multiply(y_vector, y_vector))

        logging.debug(f"Final sums - sumx:{sumx}, sumx2:{sumx2}, sumxy:{sumxy}, sumy:{sumy}, sumy2:{sumy2}")

        # Check for NaN and infinite values in the sums
        assert not np.any(np.isnan(sumx)), "sumx contains NaNs"
        assert not np.any(np.isnan(sumx2)), "sumx2 contains NaNs"
        assert not np.any(np.isnan(sumxy)), "sumxy contains NaNs"
        assert not np.any(np.isnan(sumy)), "sumy contains NaNs"
        assert not np.any(np.isnan(sumy2)), "sumy2 contains NaNs"
        assert not np.any(np.isinf(sumx)), "sumx contains infinite values"
        assert not np.any(np.isinf(sumx2)), "sumx2 contains infinite values"
        assert not np.any(np.isinf(sumxy)), "sumxy contains infinite values"
        assert not np.any(np.isinf(sumy)), "sumy contains infinite values"
        assert not np.any(np.isinf(sumy2)), "sumy2 contains infinite values"

        # Calculate the denominators
        denoms: np.ndarray = np.full(self._vector_size, n) * sumx2 - np.multiply(sumx, sumx)

        logging.debug(f"denoms: {denoms}")

        # Check for NaN and infinite values in the demons
        assert not np.any(np.isnan(denoms)), "denoms contains NaNs"
        assert not np.any(np.isinf(denoms)), "denoms contains infinite values"

        # Initialize the slope and bias elements to zero
        slope: np.ndarray = np.zeros(self._vector_size)
        bias: np.ndarray = np.zeros(self._vector_size)

        # Calculate the slopes and biases
        for i, denom in enumerate(denoms):
            # Avoid division by zero
            if denom == 0.0:
                continue
            slope[i] = (n * sumxy[i] - sumx[i] * sumy[i]) / denom
            bias[i] = (sumy[i] * sumx2[i] - sumx[i] * sumxy[i]) / denom

        logging.debug(f"slope: {slope}")
        logging.debug(f"bias: {bias}")

        return slope, bias

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple]) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        This method applies the private linear regression methods to predict the best red apple.
        """
        # Determine the number of arrays in the current_model_vectors
        num_arrays: int = len(self._chosen_apple_vectors)
        logging.debug(f"num_arrays: {num_arrays}")

        # Initialize the x_predict_base and _base arrays
        x_predict_base: np.ndarray = np.zeros((0, self._vector_size))
        y_predict_base: np.ndarray = np.zeros((0, self._vector_size))

        # Iterate through the chosen apple vectors
        for i in range(num_arrays):
            # Extract the green and red apple vectors
            green_apple_vector: np.ndarray = self._chosen_apple_vectors[i].green_apple_vector
            red_apple_vector: np.ndarray = self._chosen_apple_vectors[i].winning_red_apple_vector

            # Calculate the x vector
            x_predict_base = np.vstack([x_predict_base, self._calculate_x_vector(green_apple_vector, red_apple_vector)])

            # Calculate the y vector
            y_predict_base = np.vstack([y_predict_base, self._initialize_y_vectors(x_predict_base, winning_apple=True)])

        # Ensure there are at least 1 x_predict_base arrays to calculate linear regression
        if x_predict_base.shape[0] < 1:
            x_predict_base = np.vstack([x_predict_base, self._calculate_x_vector(self._pretrained_vectors[0].green_apple_vector, self._pretrained_vectors[0].winning_red_apple_vector)])

        # Ensure there are at least 1 y_predict_base arrays to calculate linear regression
        if y_predict_base.shape[0] < 1:
            y_predict_base = np.vstack([y_predict_base, self._initialize_y_vectors(x_predict_base, winning_apple=True)])

        logging.debug(f"x_predict_base: {x_predict_base}")
        logging.debug(f"y_predict_base: {y_predict_base}")

        # Initialize the best score and best red apple
        best_red_apple: RedApple | None = None
        best_score: float = np.inf

        # Iterate through the red apples to find the best one
        for red_apple in red_apples_in_hand:
            # Calculate the winning x_predict vector
            x_predict: np.ndarray = self._calculate_x_vector_from_apples(green_apple, red_apple)
            logging.debug(f"x_predict before: {x_predict}")

            # Initialize the winning y_predict vector
            y_predict = self._initialize_y_vectors(x_predict, winning_apple=True)
            logging.debug(f"y_predict before: {y_predict}")

            # Stack the base vectors with the new vectors
            x_predict = np.vstack([x_predict_base, x_predict])
            y_predict = np.vstack([y_predict_base, y_predict])
            logging.debug(f"x_predict after stacking: {x_predict}")
            logging.debug(f"y_predict after stacking: {y_predict}")

            # Use linear regression to predict the preference output
            self._slope_predict, self._bias_predict = self.__linear_regression(x_predict, y_predict)
            logging.debug(f"self._slope_predict: {self._slope_predict}")
            logging.debug(f"self._bias_predict: {self._bias_predict}")

            # Evaluate the score using RMSE
            score_mse = self._calculate_score(self._slope_predict, self._bias_predict, self._slope_target, self._bias_target)
            logging.debug(f"score_mse: {score_mse}")

            # Evaluate the score using Euclidean distance
            score_euclid = self._calculate_score(self._slope_predict, self._bias_predict, self._slope_target, self._bias_target, True)
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

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict[Agent, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        This method is only used by the self model and applies the private linear regression methods to predict the winning red apple.
        """
        # Initialize variables to track the best choice
        winning_red_apple: dict[Agent, RedApple] | None = None
        best_score = np.inf

        # If in training mode, choose the only red apple and return early
        if self._training_mode:
            winning_red_apple = apples_in_play.red_apples[0]
            return winning_red_apple

        # Initialize the x_predict_base and _base arrays
        x_predict_base: np.ndarray = np.zeros((0, self._vector_size))
        y_predict_base: np.ndarray = np.zeros((0, self._vector_size))

        # Calculate the x and y vectors from the pretrained data
        x_predict_base, y_predict_base = self._calculate_x_and_y_vectors_from_chosen_apple_vectors(self._pretrained_vectors)
        logging.debug(f"x_predict_base: {x_predict_base}")
        logging.debug(f"y_predict_base: {y_predict_base}")

        # Iterate through the red apples to find the best one
        for red_apple_dict in apples_in_play.red_apples:
            # Initialize the x_predict and y_predict arrays
            x_predict: np.ndarray = np.zeros((0, self._vector_size))
            y_predict: np.ndarray = np.zeros((0, self._vector_size))

            # Extract the red apple from the dictionary
            red_apple: RedApple = list(red_apple_dict.values())[0]

            # Calculate the winning x_predict vector
            x_predict = self._calculate_x_vector_from_apples(apples_in_play.get_green_apple(), red_apple)
            logging.debug(f"x_predict: {x_predict}")

            # Initialize the winning y_predict vector
            y_predict = self._initialize_y_vectors(x_predict, winning_apple=True)
            logging.debug(f"y_predict: {y_predict}")

            # Process the losing apple pairs, if applicable
            if self._use_losing_red_apples:
                # Calculate the losing x_predict and y_predict vectors
                losing_x_predict, losing_y_predict = self._calculate_losing_x_and_y_vectors_from_green_apple(apples_in_play.get_green_apple())

                # Stack the losing x_predict and y_predict vectors
                x_predict = np.vstack([x_predict, losing_x_predict])
                y_predict = np.vstack([y_predict, losing_y_predict])

            # Stack the base vectors with the new vectors
            x_predict = np.vstack([x_predict_base, x_predict])
            y_predict = np.vstack([y_predict_base, y_predict])

            logging.debug(f"x_predict after stacking: {x_predict}")
            logging.debug(f"y_predict after stacking: {y_predict}")

            # Use linear regression to predict the preference output
            self._slope_predict, self._bias_predict = self.__linear_regression(x_predict, y_predict)
            logging.debug(f"self._slope_predict: {self._slope_predict}")
            logging.debug(f"self._bias_predict: {self._bias_predict}")

            # Evaluate the score using RMSE
            score_mse = self._calculate_score(self._slope_predict, self._bias_predict, self._slope_target, self._bias_target)
            logging.debug(f"score_mse: {score_mse}")

            # Evaluate the score using Euclidean distance
            score_euclid = self._calculate_score(self._slope_predict, self._bias_predict, self._slope_target, self._bias_target, True)
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

    def train_model(self, chosen_apples: ChosenApples) -> None:
        """
        Train the model using winning green and red apple pairs and losing green and red apple pairs if applicable.
        """
        # Append the new chosen apples
        self._chosen_apples.append(chosen_apples)

        # Collect the new chosen apple vectors
        chosen_apple_vectors: ChosenAppleVectors | ChosenAppleVectorsExtra = self._collect_chosen_apple_vectors(chosen_apples)

        # Append and save the chosen apple vectors, then calculate the slope and bias vectors
        if self._training_mode:
            # Append the chosen apple vectors to the list
            self._pretrained_vectors.append(chosen_apple_vectors)
            # Save the chosen apple vectors to .npz file
            self._save_chosen_apple_vectors(self._pretrained_vectors, self._training_mode)
        else:
            # Append the chosen apple vectors to the list
            self._chosen_apple_vectors.append(chosen_apple_vectors)
            # Save the chosen apple vectors to .npz file
            self._save_chosen_apple_vectors(self._chosen_apple_vectors, self._training_mode)
            # Extract and update the slope and bias vectors, but only if there are at least 2 chosen apple vectors
            if len(self._chosen_apple_vectors) >= 2:
                self._slope_predict, self._bias_predict = self._calculate_slope_and_bias_vectors(self._chosen_apple_vectors, self.__linear_regression)

        logging.info(f"Trained the model using the chosen apple vectors.")


class NNModel(Model):
    """
    Neural Network model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int, pretrained_archetype: str, use_extra_vectors: bool = False, use_losing_red_apples : bool = False, training_mode: bool = False) -> None:
        super().__init__(judge, vector_size, pretrained_archetype, use_extra_vectors, use_losing_red_apples, training_mode)
        # Initialize the target slope and bias vectors, if not in training mode
        if not self._training_mode:
            self._slope_target, self._bias_target = self._calculate_slope_and_bias_vectors(self._pretrained_vectors, self.__forward_propagation)

        # Define the neural network model architecture with two hidden layers
        self.__nn_model = Sequential([
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
        self.__nn_model.compile(optimizer=Adam(learning_rate=self._learning_rate), loss="mean_squared_error")

    def get_current_slope_and_bias_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current slope and bias vectors.
        """
        if self._training_mode:
            return self._slope_target, self._bias_target
        else:
            return self._slope_predict, self._bias_predict

    def __forward_propagation(self, x_vector_array: np.ndarray, y_vector_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward propagation algorithm for the AI agent.
        """
        # TODO - FIX THIS METHOD
        # y = mx + b, where x is the product of green and red apple vectors
        x: np.ndarray = np.multiply(x_vector_array, y_vector_array)
        # y_pred = np.multiply(self._slope_vector, x) + self._bias_vector
        # return y_pred
        prediction = self.__nn_model.predict(x)

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
        self.__nn_model.train_on_batch(x, self._y_target)

    def choose_red_apple(self, green_apple: GreenApple, red_apples_in_hand: list[RedApple]) -> RedApple:
        """
        Choose a red card from the agent's hand to play (when the agent is a regular player).
        This method applies the private neural network methods to predict the best red apple.
        """
        # Initialize the best score and best red apple
        best_red_apple: RedApple | None = None
        best_score: float = np.inf

        # Iterate through the red apples to find the best one
        for red_apple in red_apples_in_hand:
            # Calculate the winning x_predict vector
            x_predict: np.ndarray = self._calculate_x_vector_from_apples(green_apple, red_apple)
            logging.debug(f"x_predict: {x_predict}")

            # Initialize the winning y_predict vector
            y_predict = self._initialize_y_vectors(x_predict, winning_apple=True)
            logging.debug(f"y_predict: {y_predict}")

            # Use forward propogation to predict the preference output
            self._slope_predict, self._bias_predict = self.__forward_propagation(x_predict, y_predict)
            logging.debug(f"self._slope_predict: {self._slope_predict}")
            logging.debug(f"self._bias_predict: {self._bias_predict}")

            # Evaluate the score using RMSE
            score_mse = self._calculate_score(self._slope_predict, self._bias_predict, self._slope_target, self._bias_target)
            logging.debug(f"score_mse: {score_mse}")

            # Evaluate the score using Euclidean distance
            score_euclid = self._calculate_score(self._slope_predict, self._bias_predict, self._slope_target, self._bias_target, True)
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

    def choose_winning_red_apple(self, apples_in_play: ApplesInPlay) -> dict[Agent, RedApple]:
        """
        Choose the winning red card from the red cards submitted by the other agents (when the agent is the judge).
        This method applies the private neural network methods to predict the winning red apple.
        """
        # Initialize variables to track the best choice
        winning_red_apple: dict[Agent, RedApple] | None = None
        best_score = np.inf

        # If in training mode, choose the only red apple and return early
        if self._training_mode:
            winning_red_apple = apples_in_play.red_apples[0]
            return winning_red_apple

        # Initialize the x_predict_base and _base arrays
        x_predict_base: np.ndarray = np.zeros((0, self._vector_size))
        y_predict_base: np.ndarray = np.zeros((0, self._vector_size))

        # Calculate the x and y vectors from the pretrained data
        x_predict_base, y_predict_base = self._calculate_x_and_y_vectors_from_chosen_apple_vectors(self._pretrained_vectors)
        logging.debug(f"x_predict_base: {x_predict_base}")
        logging.debug(f"y_predict_base: {y_predict_base}")

        # Iterate through the red apples to find the best one
        for red_apple_dict in apples_in_play.red_apples:
            # Initialize the x_predict and y_predict arrays
            x_predict: np.ndarray = np.zeros((0, self._vector_size))
            y_predict: np.ndarray = np.zeros((0, self._vector_size))

            # Extract the red apple from the dictionary
            red_apple: RedApple = list(red_apple_dict.values())[0]

            # Calculate the winning x_predict vector
            x_predict = self._calculate_x_vector_from_apples(apples_in_play.get_green_apple(), red_apple)
            logging.debug(f"x_predict: {x_predict}")

            # Initialize the winning y_predict vector
            y_predict = self._initialize_y_vectors(x_predict, winning_apple=True)
            logging.debug(f"y_predict: {y_predict}")

            # Process the losing apple pairs, if applicable
            if self._use_losing_red_apples:
                # Calculate the losing x_predict and y_predict vectors
                losing_x_predict, losing_y_predict = self._calculate_losing_x_and_y_vectors_from_green_apple(apples_in_play.get_green_apple())

                # Stack the losing x_predict and y_predict vectors
                x_predict = np.vstack([x_predict, losing_x_predict])
                y_predict = np.vstack([y_predict, losing_y_predict])

            # Stack the base vectors with the new vectors
            x_predict = np.vstack([x_predict_base, x_predict])
            y_predict = np.vstack([y_predict_base, y_predict])

            logging.debug(f"x_predict after stacking: {x_predict}")
            logging.debug(f"y_predict after stacking: {y_predict}")

            # Use linear regression to predict the preference output
            self._slope_predict, self._bias_predict = self.__forward_propagation(x_predict, y_predict)
            logging.debug(f"self._slope_predict: {self._slope_predict}")
            logging.debug(f"self._bias_predict: {self._bias_predict}")

            # Evaluate the score using RMSE
            score_mse = self._calculate_score(self._slope_predict, self._bias_predict, self._slope_target, self._bias_target)
            logging.debug(f"score_mse: {score_mse}")

            # Evaluate the score using Euclidean distance
            score_euclid = self._calculate_score(self._slope_predict, self._bias_predict, self._slope_target, self._bias_target, True)
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

    def train_model(self, chosen_apples: ChosenApples) -> None:
        """
        Train the model using winning green and red apple pairs and losing green and red apple pairs if applicable.
        """
        # Append the new chosen apples
        self._chosen_apples.append(chosen_apples)

        # Collect the new chosen apple vectors
        chosen_apple_vectors: ChosenAppleVectors | ChosenAppleVectorsExtra = self._collect_chosen_apple_vectors(chosen_apples)

        # Append and save the chosen apple vectors, then calculate the slope and bias vectors
        if self._training_mode:
            # Append the chosen apple vectors to the list
            self._pretrained_vectors.append(chosen_apple_vectors)
            # Save the chosen apple vectors to .npz file
            self._save_chosen_apple_vectors(self._pretrained_vectors, self._training_mode)
        else:
            # Append the chosen apple vectors to the list
            self._chosen_apple_vectors.append(chosen_apple_vectors)
            # Save the chosen apple vectors to .npz file
            self._save_chosen_apple_vectors(self._chosen_apple_vectors, self._training_mode)
            # Extract and update the slope and bias vectors, but only if there are at least 2 chosen apple vectors
            if len(self._chosen_apple_vectors) >= 2:
                self._slope_predict, self._bias_predict = self._calculate_slope_and_bias_vectors(self._chosen_apple_vectors, self.__forward_propagation)

        logging.info(f"Trained the model using the chosen apple vectors.")


# Define the mapping from user input to model type
model_type_mapping = {
    '1': LRModel,
    '2': NNModel
}


if __name__ == "__main__":
    pass
