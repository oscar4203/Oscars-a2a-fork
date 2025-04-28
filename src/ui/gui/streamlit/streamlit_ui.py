import streamlit as st
import logging # Import logging
from typing import TYPE_CHECKING
import sys # Import sys
import os # Import os

# Add project root to sys.path to allow imports from src
# This ensures that when Streamlit runs this script directly,
# it can find modules in the 'src' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if (project_root not in sys.path):
    sys.path.insert(0, project_root)

# Now try importing (use TYPE_CHECKING)
if TYPE_CHECKING:
    # These imports are just for type hinting, not runtime execution here
    from src.apples_to_apples import ApplesToApples
    from src.agent_model.agent import Agent
    from src.apples.apples import GreenApple, RedApple
    # Import GameLog for type hinting if needed
    from src.data_classes.data_classes import GameLog

class StreamlitUI:
    # --- Modify __init__ to get game from session_state ---
    def __init__(self):
        """Initializes the GUI wrapper, retrieving the game instance from Streamlit's session state."""
        if 'a2a_game_instance' in st.session_state:
            # Retrieve the game instance stored by game_driver.py
            self.game: "ApplesToApples | None" = st.session_state['a2a_game_instance']
            # Basic logging setup for the GUI script itself (optional)
            # Note: Logging configured in game_driver might handle this if process persists
            logging.basicConfig(level=logging.INFO, format='[GUI] %(levelname)s: %(message)s')
            logging.info("GUIWrapper: Found game instance in session state.")
        else:
            # This block executes if the script is run directly without game_driver setting up session state
            st.error("FATAL: Game instance not found in session state. Launch GUI via `python game_driver.py -G`.")
            logging.error("GUIWrapper: Game instance not found in session state.")
            self.game = None # Set game to None to prevent errors later
            st.stop() # Stop the Streamlit script execution

        # Initialize session state variables if they don't exist
        if 'game_started' not in st.session_state:
            # Check if the game log indicates the game has already started (e.g., round > 0)
            # This helps if the GUI is refreshed after the game has begun
            try:
                 # Access game_log safely, checking if self.game is valid
                 if self.game:
                     game_log: "GameLog" = self.game.get_game_log()
                     st.session_state['game_started'] = game_log.get_current_round_number() > 0
                 else:
                     st.session_state['game_started'] = False
            except Exception as e:
                 # Default to False if accessing game state fails
                 logging.warning(f"GUI: Could not determine initial game state: {e}")
                 st.session_state['game_started'] = False
        # Add more state variables as needed for interaction later

    # --- display_game_info remains the same ---
    def display_game_info(self):
        """Displays general game and round information."""
        st.sidebar.header("Game Info")
        # Add check if self.game is valid
        if not self.game:
            st.sidebar.warning("Game object not available.")
            return
        try:
            game_log: "GameLog" = self.game.get_game_log()
            st.sidebar.write(f"Game: {game_log.get_current_game_number()} / {game_log.total_games}")
            st.sidebar.write(f"Round: {game_log.get_current_round_number()}") # Will be 0 if game not started
            st.sidebar.write(f"Points to Win: {game_log.points_to_win}")

            current_judge = game_log.get_current_judge()
            if current_judge:
                st.subheader(f"Judge: {current_judge.get_name()}")
            else:
                st.subheader("Judge: (Not selected yet)")

        except (IndexError, AttributeError, ValueError) as e:
            st.sidebar.warning("Game not fully initialized yet.")
            logging.warning(f"GUI: Error accessing game/round info: {e}")
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred: {e}")
            logging.error(f"GUI: Unexpected error displaying game info: {e}")

    # --- display_green_apple remains the same ---
    def display_green_apple(self):
        """Displays the current green apple."""
        st.header("Green Apple")
        # Add check if self.game is valid
        if not self.game:
            st.write("(Game object not available)")
            return
        try:
            game_log: "GameLog" = self.game.get_game_log()
            # Use get_apples_in_play() which should return the ApplesInPlay object
            apples_in_play = game_log.get_apples_in_play()
            if apples_in_play:
                green_apple_in_play: GreenApple = apples_in_play.get_green_apple()
                if green_apple_in_play:
                    st.markdown(f"**{green_apple_in_play.get_adjective()}**")
                    st.caption(green_apple_in_play.get_synonyms())
                else:
                    st.write("(None drawn yet)")
            else:
                 st.write("(Round not started?)") # If get_apples_in_play returns None or similar
        except (IndexError, AttributeError, ValueError):
            st.write("(None drawn yet)") # Handle cases where round/apples aren't ready
        except Exception as e:
            st.error(f"An unexpected error occurred displaying green apple: {e}")
            logging.error(f"GUI: Unexpected error displaying green apple: {e}")

    # --- display_players_and_hands remains the same ---
    def display_players_and_hands(self):
        """Displays player info and their hands in expanders."""
        st.header("Players")
        # Add check if self.game is valid
        if not self.game:
            st.write("Game object not available.")
            return
        try:
            game_log: "GameLog" = self.game.get_game_log()
            players = game_log.get_game_players()
            if not players:
                st.write("No players initialized yet.")
                return

            # Create columns for players
            num_players = len(players)
            # Adjust column widths if needed, or let Streamlit handle it
            cols = st.columns(num_players)

            for i, player in enumerate(players):
                with cols[i]:
                    is_judge = player.get_judge_status()
                    judge_indicator = " (Judge)" if is_judge else ""
                    # Use markdown for potentially better formatting control
                    st.markdown(f"**{player.get_name()}{judge_indicator}**")
                    st.write(f"Points: {player.get_points()}")

                    # Display hand in an expander
                    with st.expander("Hand"):
                        red_apples_in_hand: list["RedApple"] = player.get_red_apples()
                        if red_apples_in_hand:
                            for card in red_apples_in_hand:
                                st.markdown(f"**{card.get_noun()}**")
                                st.caption(card.get_description())
                                st.divider() # Adds a line between cards
                        else:
                            st.write("(Empty)")

        except (IndexError, AttributeError, ValueError) as e:
            st.warning("Player data not fully available yet.")
            logging.warning(f"GUI: Error accessing player data: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred displaying players: {e}")
            logging.error(f"GUI: Unexpected error displaying players: {e}")

    # --- display_submitted_cards remains the same ---
    def display_submitted_cards(self):
        """Displays red apples submitted this round."""
        st.header("Submitted Red Apples")
        # Add check if self.game is valid
        if not self.game:
            st.write("(Game object not available)")
            return
        try:
            game_log: "GameLog" = self.game.get_game_log()
            apples_in_play = game_log.get_apples_in_play()
            if apples_in_play:
                # Assuming red_apples is a list of dicts like [{player: card}, {player: card}]
                submitted_dicts = apples_in_play.red_apples
                if submitted_dicts:
                    for card_dict in submitted_dicts:
                        # Extract player and card (safer extraction)
                        player = next(iter(card_dict.keys()), None)
                        card = next(iter(card_dict.values()), None)
                        if player and card:
                            st.markdown(f"**{card.get_noun()}** (_submitted by {player.get_name()}_)")
                            st.caption(card.get_description())
                            st.divider()
                        else:
                            st.warning("Could not parse submitted card dictionary.")
                else:
                    st.write("(None yet)")
            else:
                 st.write("(Round not started?)")
        except (IndexError, AttributeError, ValueError):
             st.write("(None yet)") # Handle cases where round/apples aren't ready
        except Exception as e:
            st.error(f"An unexpected error occurred displaying submitted cards: {e}")
            logging.error(f"GUI: Unexpected error displaying submitted cards: {e}")

    # --- Update run method ---
    def run(self):
        """Main function to display the GUI components."""
        # Check if game object is valid (set in __init__)
        if not self.game:
             st.error("GUI cannot run without a valid game instance.")
             return

        st.set_page_config(layout="wide") # Use wider layout
        st.title("Apples to Apples AI Agent")

        # --- Sidebar Controls ---
        st.sidebar.title("Controls")
        # Use session state to track if the game has been started via GUI button
        if not st.session_state.get('game_started', False): # Use .get for safety
             # Check if the underlying game state indicates game 0 / round 0
             try:
                 game_log: "GameLog" = self.game.get_game_log()
                 # Allow start only if game 1, round 0 (or game 0 if init state)
                 allow_start = game_log.get_current_game_number() <= 1 and game_log.get_current_round_number() == 0
             except Exception as e:
                 logging.warning(f"GUI: Could not check initial game state for start button: {e}")
                 allow_start = True # Assume allowed if state check fails

             # Show start button only if allowed
             if allow_start and st.sidebar.button("Start First Game"):
                 try:
                     # Call the game's method to start/initialize the first game/round
                     self.game.new_game()
                     st.session_state.game_started = True # Update session state
                     st.rerun() # Rerun the script to update the display immediately
                 except Exception as e:
                     st.sidebar.error(f"Error starting game: {e}")
                     logging.error(f"GUI: Error calling game.new_game(): {e}")
             elif not allow_start:
                 # If game is already past the starting point, reflect that
                 st.sidebar.write("Game already started.")
                 if not st.session_state.game_started:
                     st.session_state.game_started = True # Ensure state matches reality
                     st.rerun() # Rerun if state was corrected

        else: # If game_started is True
            # Placeholder for future controls (e.g., "Next Round", player actions)
            st.sidebar.write("Game in progress...")
            # Add a button to manually refresh state if needed for debugging
            if st.sidebar.button("Refresh Display"):
                st.rerun()

        # --- Main Display Area ---
        # Always display info if game object exists
        self.display_game_info()
        self.display_green_apple()
        self.display_submitted_cards() # Show submitted cards
        self.display_players_and_hands()


# --- Add a main execution block for when Streamlit runs this script ---
if __name__ == "__main__":
    # This block executes when running `streamlit run src/gui/gui_wrapper.py`
    # It expects 'a2a_game_instance' to be in st.session_state (set by game_driver.py)
    # Basic logging setup for when run directly
    logging.basicConfig(level=logging.INFO, format='[GUI Main] %(levelname)s: %(message)s')
    logging.info("gui_wrapper.py executed as main script.")
    # Instantiate the wrapper, which will get the game object from session state
    gui = StreamlitUI()
    # Run the main display logic
    gui.run()
# --- End Main Execution Block ---
