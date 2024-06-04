# Description: Green and Red 'Apple' cards.

# Standard Libraries
import numpy as np
import pandas as pd

# Third-party Libraries
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("../")

# Local Modules
from source.apples import GreenApple, RedApple, Deck


def create_green_apple_df(green_apple_deck: Deck, model: KeyedVectors) -> pd.DataFrame:
    # Create a list to store the Green Apple vectors
    green_apple_vectors = []

    # Get the Green Apple adjective vectors
    for green_apple in green_apple_deck.apples:
        if isinstance(green_apple, GreenApple):
            # Get the Green Apple data
            adjective = green_apple.get_adjective()
            # synonyms = green_apple.get_synonyms()

            # Get the vectors
            adjective_vector: np.ndarray = model[adjective]
            # synonyms_vector: np.ndarray = model[synonyms]

            # Add the vector to the list
            green_apple_vectors.append(adjective_vector.transpose())
        else:
            raise TypeError("Expected a GreenApple, but got a different type")

    # Create a DataFrame to store the Green Apple cards
    green_apple_df = pd.DataFrame(green_apple_vectors)

    return green_apple_df


def create_red_apple_df(red_apple_deck: Deck, model: KeyedVectors) -> pd.DataFrame:
    # Create a list to store the Red Apple vectors
    red_apple_vectors = []

    # Get the Red Apple noun vectors
    for red_apple in red_apple_deck.apples:
        if isinstance(red_apple, RedApple):
            # Get the Red Apple data
            noun = red_apple.get_noun()
            description = red_apple.get_description()

            # Get the vectors
            noun_vector: np.ndarray = model[noun]
            description_vector: np.ndarray = model[description]

            # Add the vector to the list
            red_apple_vectors.append(noun_vector.transpose())
        else:
            raise TypeError("Expected a RedApple, but got a different type")

    # Create a DataFrame to store the Red Apple cards
    red_apple_df = pd.DataFrame(red_apple_vectors)

    return red_apple_df


# def run_pca(data: pd.DataFrame, n_components: int) -> pd.DataFrame:
#     # Create a PCA object
#     pca = PCA(n_components=n_components)



def main():
    # Load the pre-trained word vectors
    model = KeyedVectors.load_word2vec_format("apples/GoogleNews-vectors-negative300.bin", binary=True)

    # Load the Green Apple cards to a Deck
    green_apple_deck = Deck()
    green_apple_deck.load_deck("Green Apples", "apples/green_apples.csv")

    # Load the Red Apple cards to a Deck
    red_apple_deck = Deck()
    red_apple_deck.load_deck("Red Apples", "apples/red_apples.csv")

    # Get the Green Apple adjective vectors
    green_apple_df = create_green_apple_df(green_apple_deck, model)

    # Get the Red Apple noun vectors
    red_apple_df = create_red_apple_df(red_apple_deck, model)


if __name__ == "__main__":
    main()
