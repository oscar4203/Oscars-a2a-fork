# Description: Main driver for the 'Apples to Apples' game.

# Standard Libraries
import logging
import argparse

# Third-party Libraries

# Local Modules
from config import configure_logging
from apples import GreenApple, RedApple, Deck
from agent import Player
from results import GameResults, log_results


class ApplesToApples:
    def __init__(self, number_of_players: int, points_to_win: int, green_expansion: str = '', red_expansion: str = '') -> None:
        self.number_of_players: int = number_of_players
        self.points_to_win: int = points_to_win
        self.green_expansion_filename: str = green_expansion
        self.red_expansion_filename: str = red_expansion
        self.green_apples_deck: Deck = Deck()
        self.red_apples_deck: Deck = Deck()
        self.winner: Player | None = None
        self.players: list[Player] = []
        self.round: int = 0
        self.current_judge: Player | None = None
        self.green_apples_in_play: dict[str, GreenApple] | None = None
        self.red_apples_in_play: list[dict[str, RedApple]] = []
        self.discarded_green_apples: list[GreenApple] = []
        self.discarded_red_apples: list[RedApple] = []

    def start(self) -> None:
        print("Starting 'Apples to Apples' game.")
        logging.info("Starting 'Apples to Apples' game.")
        logging.info("Initializing players.")

        # Intro to the game
        print("\nI assume I am player 1!")
        logging.info("I assume I am player 1!")

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
        for i in range(self.number_of_players):
            self.players.append(Player(f"Player {i + 1}"))
            print(self.players[i].name + ",", end=' ')
            logging.info(self.players[i])

            # Have each player pick up 7 red cards
            self.players[i].pickup_red_apples()

    def __choose_judge(self) -> None:
        # Choose the starting judge
        choice = input(f"\nPlease choose the starting judge (1-{self.number_of_players}): ")
        logging.info(f"Please choose the starting judge (1-{self.number_of_players}): {choice}")

        # Validate the user input
        while not choice.isdigit() or int(choice) < 1 or int(choice) > self.number_of_players:
            choice = input(f"Invalid input. Please enter a number (1-{self.number_of_players}): ")
            logging.error(f"Invalid input. Please enter a number (1-{self.number_of_players}): {choice}")

        # Set the judge
        self.current_judge = self.players[int(choice) - 1]
        self.players[int(choice) - 1].judge = True
        print(f"{self.players[int(choice) - 1].name} is the starting judge.")

    def __assign_next_judge(self) -> None:
        # Check if the current judge is None
        if self.current_judge is None:
            logging.error("The current judge is None.")
            raise ValueError("The current judge is None.")

        # Calculate the next judge
        next_judge: Player = self.players[(self.players.index(self.current_judge) + 1) % self.number_of_players]
        print(f"\n{next_judge.name} is the next judge.")
        logging.info(f"{next_judge.name} is the next judge.")

        # Assign the next judge
        self.current_judge.judge = False
        next_judge.judge = True
        self.current_judge = next_judge

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
        self.green_apples_in_play = {self.current_judge.name: self.current_judge.choose_green_apple()}
        print(f"Green card: {self.green_apples_in_play}")
        logging.info(f"Green card: {self.green_apples_in_play}")

    def __player_prompt(self) -> None:
        # Prompt the players to select a red card
        for player in self.players:
            if player.judge:
                continue

            print(f"\n{player.name}, please select a red card.")
            logging.info(f"{player.name}, please select a red card.")

            # Set the red cards in play
            red_apple = player.choose_red_apple()
            self.red_apples_in_play.append({player.name: red_apple})
            logging.info(f"Red card: {red_apple}")

            # Prompt the player to pick up a new red card
            if len(player.red_apples) < 7:
                player.pickup_red_apples()

    def __game_loop(self) -> None:
        # Start the game loop
        while self.winner is None:
            # Increment the round
            self.round += 1

            # Play the round
            print(f"Round {self.round}:")
            logging.info(f"Round {self.round}:")

            # Prompt the judge to select a green card
            self.__judge_prompt()

            # Prompt the players to select a red card
            self.__player_prompt()

            # Prompt the judge to select the winning red card
            if self.current_judge is None:
                logging.error("The current judge is None.")
                raise ValueError("The current judge is None.")
            winning_red_card = self.current_judge.choose_winning_red_apple(self.red_apples_in_play)

            # Award points to the winning player
            for player in self.players:
                if winning_red_card.keys() == player.name:
                    player.points += 1
                    print(f"{player.name} has won the round!")
                    logging.info(f"{player.name} has won the round!")

            # Check for None values
            if self.green_apples_in_play is None:
                logging.error("The green apples in play is None.")
                raise ValueError("The green apples in play is None.")

            if self.red_apples_in_play is None:
                logging.error("The red apples in play is None.")
                raise ValueError("The red apples in play is None.")
            else:
                red_apples_list = []
                for red_apple in self.red_apples_in_play:
                    red_apples_list.append(list(red_apple.values())[0])

            winning_red_card = list(winning_red_card.values())[0]

            # Log the results
            results = GameResults(self.players, self.points_to_win, self.round, self.green_apples_in_play[self.current_judge.name],
                                  red_apples_list, winning_red_card, self.current_judge)
            log_results(results)

            # Discard the green cards
            self.discarded_green_apples.append(self.green_apples_in_play[self.current_judge.name])
            self.green_apples_in_play = None

            # Discard the red cards
            self.discarded_red_apples.extend(red_apples_list)
            self.red_apples_in_play = []

             # Check if the game is over
            self.winner = self.__is_game_over()
            if self.winner is not None:
                print(f"{self.winner.name} has won the game!")
                logging.info(f"{self.winner.name} has won the game!")
                break

            # Assign the next judge
            self.__assign_next_judge()


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
