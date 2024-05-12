# Description: Configuration file for the application

# Standard Libraries
import logging
import argparse

# Third-party Libraries

# Local Modules
from config import configure_logging
from results import GameResults, log_results


class Player:
    def __init__(self, name: str) -> None:
        self.name = name
        self.points = 0
        self.judge: bool = False
        self.red_cards: list[str] = []

    def __str__(self) -> str:
        return f"Player(name={self.name}, points={self.points})"


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
        self.green_card_in_play: str = ''
        self.red_cards_in_play: list[str] = []

    def start(self) -> None:
        logging.info("Starting 'Apples to Apples' game.")
        logging.info("Initializing players.")

        # Intro to the game
        print("I assume I am player 1!")
        logging.info("I assume I am player 1!")

        # Initialize the players
        print("The other players are:")
        logging.info("The other players are:")
        self.__initialize_players()

        self.__game_loop()

    def __initialize_players(self) -> None:
        for i in range(self.number_of_players):
            self.players.append(Player(f"Player {i + 1}"))
            print(self.players[i], end=' ')

    def __is_game_over(self) -> bool:
        for player in self.players:
            if player.points >= self.points_to_win:
                self.winner = player
                return True
        return False

    def __game_loop(self) -> None:


        while self.winner is None:
            self.round += 1

            # Check if the game is over
            if self.__is_game_over():
                logging.info("Game over.")
                break

            # Play the round
            print(f"Round {self.round}:")
            logging.info(f"Round {self.round}:")





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
