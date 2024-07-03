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
        self.slope_vector: np.ndarray = np.zeros(vector_size)
        self.bias_vector: np.ndarray = np.zeros(vector_size)


class LRModel(Model):
    """
    Linear Regression model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int) -> None:
        super().__init__(judge, vector_size)


class NNModel(Model):
    """
    Neural Network model for the AI agent.
    """
    def __init__(self, judge: Agent, vector_size: int) -> None:
        super().__init__(judge, vector_size)


if __name__ == "__main__":
    pass
