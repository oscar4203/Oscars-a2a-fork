# Description: Green and Red 'Apple' cards.

# Standard Libraries
import pandas as pd

# Third-party Libraries
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

# Local Modules
from src.apples.apples import Deck
from experiments.pca_exploration import create_green_apple_df, create_red_apple_df


def run_tsne(df: pd.DataFrame, n_dimensions: int, random_state: int, n_features: int, title: str):
    # Apply t-SNE
    tsne = TSNE(n_components=n_dimensions, random_state=random_state)
    vectors_2d = tsne.fit_transform(df)
    tsne_result = tsne.fit_transform(vectors_2d[:, :n_features])

    # Plot the results
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    plt.title(title)
    plt.show()


def main():
    # Load the pre-trained word vectors
    model = KeyedVectors.load_word2vec_format("../data/embeddings/GoogleNews-vectors-negative300.bin", binary=True)

    # Set the number of components and features for PCA
    n_dimensions = 2
    random_state = 42
    n_features = 300

    # Load the Green Apple cards to a Deck
    green_apple_deck = Deck()
    green_apple_deck.load_deck("Green Apples", "../data/apples/green_apples.csv")

    # Load the Red Apple cards to a Deck
    red_apple_deck = Deck()
    red_apple_deck.load_deck("Red Apples", "../data/apples/red_apples.csv")

    # Get the Green Apple adjective vectors
    green_apple_df = create_green_apple_df(green_apple_deck, model)
    # Run TSNE on the Green Apple data
    run_tsne(green_apple_df, n_dimensions, random_state, n_features, "Green Apple Adjective Vectors")

    # Get the Red Apple noun vectors
    red_apple_df = create_red_apple_df(red_apple_deck, model)
    # Run TSNE on the Red Apple data
    run_tsne(red_apple_df, n_dimensions, random_state, n_features, "Red Apple Noun Vectors")


if __name__ == "__main__":
    main()
