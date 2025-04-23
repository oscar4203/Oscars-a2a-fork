# Description: Green and Red 'Apple' cards.

# Standard Libraries
import numpy as np
import pandas as pd

# Third-party Libraries
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

# Local Modules
from src.apples.apples import GreenApple, RedApple, Deck


def create_green_apple_df(green_apple_deck: Deck, model: KeyedVectors) -> pd.DataFrame:
    # Create a list to store the Green Apple vectors
    green_apple_vectors = []

    # Get the Green Apple adjective vectors
    for green_apple in green_apple_deck.__apples:
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
    for red_apple in red_apple_deck.__apples:
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


def run_pca(data: pd.DataFrame, n_components: int, n_features: int, title: str) -> None:
    # Create a PCA object
    pca = PCA(n_components=n_components)

    # Standardize the data
    scaler = StandardScaler()

    # Fit the data
    data_scaled = scaler.fit_transform(data)

    # Fit the PCA model
    pca.fit(data_scaled)

    # # Transform the data
    # data_transformed = pca.transform(data_scaled)

    # # Create a DataFrame to store the transformed data
    # transformed_df = pd.DataFrame(data_transformed)

    # Label the components
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]

    # Create a scree plot
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.xlabel("Principle Component")
    plt.ylabel("Percent of Explained Variance")
    plt.title(f"Scree Plot - {title}")

    # Add annotation for the total percent of variance explained by the top n_components
    total_var = np.sum(per_var[:n_components])
    # round the float to 2 decimal places
    total_var = round(total_var, 2)
    plt.annotate(f'Total percent explained by top {n_components} PCs: {total_var}%',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # Show the plot
    plt.show()

    # Analyze the PCA loadings for the first few components
    loadings = pca.components_
    feature_names = data.columns

    # Sum up the absolute values of each dimension's coefficients across all PCs
    feature_importance = np.sum(np.abs(loadings), axis=0)

    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    top_features = feature_names[sorted_idx][:n_features]
    top_importance = feature_importance[sorted_idx][:n_features]

    # Plot the top 25 features
    plt.figure(figsize=(20, 16))
    plt.barh(range(n_features), top_importance[::-1], align='center')
    plt.yticks(range(n_features), top_features[::-1]) # type: ignore
    plt.xlabel(f"Sum of Absolute Coefficients - Top {n_components} Components")
    plt.title(f"Top {n_features} Features by PCA Coefficient Importance - {title}")
    plt.show()


def main():
    # Load the pre-trained word vectors
    model = KeyedVectors.load_word2vec_format("../data/embeddings/GoogleNews-vectors-negative300.bin", binary=True)

    # Set the number of components and features for PCA
    n_components = 50
    n_features = 100

    # # Load the Green Apple cards to a Deck
    # green_apple_deck = Deck()
    # green_apple_deck.load_deck("Green Apples", "../data/apples/green_apples.csv")

    # Load the Red Apple cards to a Deck
    red_apple_deck = Deck()
    red_apple_deck.load_deck("Red Apples", "../data/apples/red_apples.csv")

    # # Get the Green Apple adjective vectors
    # green_apple_df = create_green_apple_df(green_apple_deck, model)
    # # Run PCA on the Green Apple data
    # run_pca(green_apple_df, n_components, n_features, "Green Apple Adjectives")

    # Get the Red Apple noun vectors
    red_apple_df = create_red_apple_df(red_apple_deck, model)
    # Run PCA on the Red Apple data
    run_pca(red_apple_df, n_components, n_features, "Red Apple Nouns")


if __name__ == "__main__":
    main()
