import tkinter as tk
from src.embeddings.embeddings import Embedding
from src.apples.apples import GreenApple, RedApple, Deck
from src.agent_model.agent import Agent, HumanAgent, RandomAgent, AIAgent
from src.agent_model.model import model_type_mapping, Model

# --- Setup GUI ---
root = tk.Tk()
root.title("Apples to Apples - Simulation")

# --- Setup Decks ---
deck = Deck()
deck.load_deck("Red Apples", "data//apples//red_apples-all.csv")
deck.shuffle()

green_deck = Deck()
green_deck.load_deck("Green Apples", "data//apples//green_apples-all.csv")
green_deck.shuffle()

# --- GUI Frames ---
green_frame = tk.Frame(root)
green_frame.pack(pady=10)

red_frame = tk.Frame(root)
red_frame.pack(pady=20)

status_frame = tk.Frame(root)
status_frame.pack(pady=10)

# --- Global Variables ---
hand = []
current_green_apple = None

# --- Functions ---

def draw_new_round():
    #global hand, current_green_apple

    # Clear previous widgets
    for widget in green_frame.winfo_children():
        widget.destroy()
    for widget in red_frame.winfo_children():
        widget.destroy()
    for widget in status_frame.winfo_children():
        widget.destroy()

    # Draw a new green apple
    green_apple = green_deck.draw_apple()
    current_green_apple = green_apple.get_adjective()

    label = tk.Label(
        green_frame,
        text=f"Green Apple: {current_green_apple}",
        font=("Helvetica", 24, "bold"),
        fg="green"
    )
    label.pack()

    # Draw new red apples (hand)
    while len(hand) < 7:
        hand.append(deck.draw_apple().get_noun())

    for card in hand:
        btn = tk.Button(
            red_frame,
            text=card,
            font=("Helvetica", 14, "bold"),
            width=12,
            height=6,
            bg="red",
            fg="black",
            relief="raised",
            bd=3,
            command=lambda c=card: play_card(c)
        )
        btn.pack(side="left", padx=10)

def play_card(selected_card):

    hand.remove(selected_card)

    # Clear red cards
    for widget in red_frame.winfo_children():
        widget.destroy()

    # Show played card
    played_label = tk.Label(
        status_frame,
        text=f"You played: {selected_card}\nJudge chooses 'x' card. You [win/lose]. Here are the current points: blah blah blah",
        font=("Helvetica", 20),
        fg="blue"
    )
    played_label.pack(pady=10)

    # Show OK button
    ok_button = tk.Button(
        status_frame,
        text="OK",
        font=("Helvetica", 16),
        command=draw_new_round
    )
    ok_button.pack()

# --- Start First Round ---
draw_new_round()

# --- Main Loop ---
root.mainloop()