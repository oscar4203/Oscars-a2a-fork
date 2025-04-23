from src.apples_to_apples import ApplesToApples
import streamlit as st

class GUIWrapper:
    def __init__(self, game: ApplesToApples):
        self.game = game

    def display_game_state(self):
        st.title("Apples to Apples Game")
        st.write("Current Game State:")
        # Display game state information
        st.write(f"Current Game Number: {self.game.__game_log.get_current_game_number()}")
        st.write(f"Total Games: {self.game.__game_log.total_games}")
        st.write("Players:")
        for player in self.game.__game_log.get_game_players():
            st.write(f"- {player.get_name()} (Points: {player.get_points()})")

    def start_new_game(self):
        if st.button("Start New Game"):
            self.game.new_game()
            st.success("New game started!")

    def run(self):
        self.display_game_state()
        self.start_new_game()

if __name__ == "__main__":
    # Initialize the game logic
    game = ApplesToApples(embedding=None, print_in_terminal=False, training_mode=False)
    gui = GUIWrapper(game)
    gui.run()
