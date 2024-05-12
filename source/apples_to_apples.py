# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse

# Third-party Libraries

# Local Modules
from config import configure_logging
from results import GameResults, log_results
from agent import Player


class GreenApple:
    def __init__(self, adjective: str, synonyms: list[str] | None = None) -> None:
        self.adjective: str = adjective
        self.synonyms: list[str] | None = synonyms

    def __str__(self) -> str:
        return f"GreenApple(adjective={self.adjective}, synonyms={self.synonyms})"


class RedApple:
    def __init__(self, noun: str, description: str | None = None) -> None:
        self.noun: str = noun
        self.description: str | None = description

    def __str__(self) -> str:
        return f"RedApple(noun={self.noun}, description={self.description})"

class ApplesToApples:
    def __init__(self, number_of_players: int, points_to_win: int, green_expansion: str = '', red_expansion: str = '') -> None:
        self.number_of_players: int = number_of_players
        self.points_to_win: int = points_to_win
        self.green_expansion_filename: str = green_expansion
        self.red_expansion_filename: str = red_expansion
        self.winner: Player | None = None
        self.players: list[Player] = []
        self.round: int = 0
        self.current_judge: Player | None = None
        self.green_apples_in_play: GreenApple | None = None
        self.red_apples_in_play: list[RedApple] = []
        self.discarded_green_apples: list[GreenApple] = []
        self.discarded_red_apples: list[RedApple] = []

    def start(self) -> None:
        print("Starting 'Apples to Apples' game.")
        logging.info("Starting 'Apples to Apples' game.")
        logging.info("Initializing players.")

        # Intro to the game
        print("\nI assume I am player 1!")
        logging.info("\nI assume I am player 1!")

        # Initialize the players
        self.__initialize_players()

        # Choose the starting judge
        self.__choose_judge()

        # Start the game loop
        self.__game_loop()

    def __initialize_players(self) -> None:
        print("The other players are:")
        logging.info("The other players are:")

        # Create the players
        for i in range(self.number_of_players - 1):
            self.players.append(Player(f"Player {i + 1}"))
            print(self.players[i].name + ",", end=' ')
            logging.info(self.players[i])

            # Have each player pick up 7 red cards
            self.players[i].pickup_red_apples()

        # Add the last player
        self.players.append(Player(f"Player {self.number_of_players}"))
        print(self.players[-1].name)
        logging.info(self.players[-1])

        # Have the last player pick up 7 red cards
        self.players[-1].pickup_red_apples()

    def __choose_judge(self) -> None:
        # Choose the starting judge
        choice = input(f"\nPlease choose the starting judge (1-{self.number_of_players}): ")
        logging.info(f"\nPlease choose the starting judge (1-{self.number_of_players}): {choice}")

        # Validate the user input
        while not choice.isdigit() or int(choice) < 1 or int(choice) > self.number_of_players:
            choice = input(f"Invalid input. Please enter a number (1-{self.number_of_players}): ")
            logging.error(f"Invalid input. Please enter a number (1-{self.number_of_players}): {choice}")

        # Set the judge
        self.current_judge = self.players[int(choice) - 1]
        self.players[int(choice) - 1].judge = True
        print(f"{self.players[int(choice) - 1].name} is the starting judge.")

    def __is_game_over(self) -> Player | None:
        for player in self.players:
            if player.points >= self.points_to_win:
                return player
        return None

    def __judge_prompt(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Prompt the judge to select a green card
        print(f"\n{self.current_judge.name}, please select a green card.")
        logging.info(f"\n{self.current_judge.name}, please select a green card.")

        # Set the green card in play
        self.green_apples_in_play = GreenApple("Green card")
        print(f"Green card: {self.green_apples_in_play}")
        logging.info(f"Green card: {self.green_apples_in_play}")

    def __player_prompt(self) -> None:
        # Prompt the players to select a red card
        for player in self.players:
            if player.judge:
                continue

            print(f"\n{player.name}, please select a red card.")
            logging.info(f"\n{player.name}, please select a red card.")

            # Set the red cards in play
            red_apple = player.choose_red_apple()
            self.red_apples_in_play.append(red_apple)
            logging.info(f"Red card: {red_apple}")

            # Prompt the player to pick up a new red card
            if len(player.red_apples) < 7:
                player.pickup_red_apples()

    def __game_loop(self) -> None:
        # Start the game loop
        while self.winner is None:
            # Increment the round
            self.round += 1

            # Check if the game is over
            self.winner = self.__is_game_over()
            if self.winner is not None:
                print(f"{self.winner.name} has won the game!")
                logging.info(f"{self.winner.name} has won the game!")
                break

            # Play the round
            print(f"Round {self.round}:")
            logging.info(f"Round {self.round}:")

            # Prompt the judge to select a green card
            self.__judge_prompt()

            # Prompt the players to select a red card
            self.__player_prompt()



def main() -> None:
    # Define the command line arguments
    parser = argparse.ArgumentParser(description="Apples to Apples game configuration.")
    parser.add_argument("players", type=int, choices=range(3, 9), help="Total number of players (3-8).")
    parser.add_argument("points", type=int, choices=range(1, 11), help="Total number of points to win (1-10).")
    parser.add_argument("green_expansion", type=str, nargs='?', default='', help="Filename to a green card expansion (optional).")
    parser.add_argument("red_expansion", type=str, nargs='?', default='', help="Filename to a red card expansion (optional).")

    # Parse the command line arguments
    args = parser.parse_args()

    # Configure and initialize the logging module
    configure_logging()
    logging.info("Starting 'Apples to Apples' application.")

    # Log the command line arguments
    logging.info(f"Command line arguments: {args}")
    logging.info(f"Number of players: {args.players}")
    logging.info(f"Points to win: {args.points}")
    logging.info(f"Green card expansion file: {args.green_expansion}")
    logging.info(f"Red card expansion file: {args.red_expansion}")

    # Create the game object
    game = ApplesToApples(args.players, args.points, args.green_expansion, args.red_expansion)

    # Start the game
    game.start()


if __name__ == "__main__":
    main()
