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
            # Set the Green Apple adjective vector
            green_apple.set_adjective_vector(model)
            # green_apple.set_synonyms_vector(model)

            adjective_vector = green_apple.get_adjective_vector()
            if adjective_vector is None:
                raise ValueError("The adjective vector is None")

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
            red_apple.set_noun_vector(model)
            # red_apple.set_description_vector(model)

            noun_vector = red_apple.get_noun_vector()
            if noun_vector is None:
                raise ValueError("The noun vector is None")

            # Add the vector to the list
            red_apple_vectors.append(noun_vector.transpose())
        else:
            raise TypeError("Expected a RedApple, but got a different type")

    # Create a DataFrame to store the Red Apple cards
    red_apple_df = pd.DataFrame(red_apple_vectors)

    return red_apple_df


def run_pca(data: pd.DataFrame, n_components: int) -> None:
    # Create a PCA object
    pca = PCA(n_components=n_components)

    # Standardize the data
    scaler = StandardScaler()

    # Fit the data
    data_scaled = scaler.fit_transform(data)

    # Fit the PCA model
    pca.fit(data_scaled)

    # Transform the data
    data_transformed = pca.transform(data_scaled)

    # # Create a DataFrame to store the transformed data
    # transformed_df = pd.DataFrame(data_transformed)

    # Plot the PCA
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.xlabel("Principle Component")
    plt.ylabel("Percent of Explained Variance")
    plt.title("Scree Plot")
    plt.show()


def main():
    # Load the pre-trained word vectors
    model = KeyedVectors.load_word2vec_format("../apples/GoogleNews-vectors-negative300.bin", binary=True)

    # Load the Green Apple cards to a Deck
    green_apple_deck = Deck()
    green_apple_deck.load_deck("Green Apples", "../apples/green_apples.csv")

    # # Load the Red Apple cards to a Deck
    # red_apple_deck = Deck()
    # red_apple_deck.load_deck("Red Apples", "../apples/red_apples.csv")

    # Get the Green Apple adjective vectors
    green_apple_df = create_green_apple_df(green_apple_deck, model)
    # Run PCA on the Green Apple data
    run_pca(green_apple_df, 50)

    # # Get the Red Apple noun vectors
    # red_apple_df = create_red_apple_df(red_apple_deck, model)
    # # Run PCA on the Red Apple data
    # run_pca(red_apple_df, 10)


if __name__ == "__main__":
    main()
