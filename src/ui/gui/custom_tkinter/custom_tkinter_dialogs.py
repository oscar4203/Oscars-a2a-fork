class CTkMessageBox(ctk.CTkToplevel):
    """Simple message box dialog with customizable buttons."""
    def __init__(self, master, title="Message", message="", icon="info",
                option_1="OK", option_2="", option_3=""):
        super().__init__(master)
        self.title(title)
        self.geometry("400x200")
        self.resizable(False, False)
        self.result = None

        # Set up the UI
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frame to hold content
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        frame.grid_columnconfigure(0, weight=1)

        # Message
        message_label = ctk.CTkLabel(frame, text=message, wraplength=350)
        message_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        # Button frame
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))

        # Add buttons
        col = 0
        if option_1:
            btn1 = ctk.CTkButton(button_frame, text=option_1, command=lambda: self.set_result(option_1))
            btn1.grid(row=0, column=col, padx=10)
            col += 1

        if option_2:
            btn2 = ctk.CTkButton(button_frame, text=option_2, command=lambda: self.set_result(option_2))
            btn2.grid(row=0, column=col, padx=10)
            col += 1

        if option_3:
            btn3 = ctk.CTkButton(button_frame, text=option_3, command=lambda: self.set_result(option_3))
            btn3.grid(row=0, column=col, padx=10)

        # Make modal
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: self.set_result(None))

        # Center the dialog on parent
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - width) // 2
        y = master.winfo_rooty() + (master.winfo_height() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def set_result(self, value):
        """Set the result and close the dialog."""
        self.result = value
        self.grab_release()
        self.destroy()

    def get(self):
        """Wait for the dialog to close and return the result."""
        self.master.wait_window(self)
        return self.result


class InputDialog(ctk.CTkToplevel):
    """Dialog for getting text input."""
    def __init__(self, master, title="Input", prompt="Enter value:"):
        super().__init__(master)
        self.title(title)
        self.geometry("400x150")
        self.resizable(False, False)
        self.result = None

        # Set up the UI
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frame to hold content
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        frame.grid_columnconfigure(0, weight=1)

        # Prompt
        prompt_label = ctk.CTkLabel(frame, text=prompt)
        prompt_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Entry
        self.entry = ctk.CTkEntry(frame, width=300)
        self.entry.grid(row=1, column=0, padx=20, pady=10)
        self.entry.focus_set()

        # Button
        ok_button = ctk.CTkButton(frame, text="OK", command=self.on_ok)
        ok_button.grid(row=2, column=0, padx=20, pady=(10, 20))

        # Make modal
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)

        # Bind Enter key to OK
        self.bind("<Return>", lambda event: self.on_ok())

        # Center the dialog on parent
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - width) // 2
        y = master.winfo_rooty() + (master.winfo_height() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def on_ok(self):
        """Get the input and close the dialog."""
        self.result = self.entry.get()
        self.grab_release()
        self.destroy()

    def on_cancel(self):
        """Cancel and close the dialog."""
        self.result = None
        self.grab_release()
        self.destroy()

    def get_input(self):
        """Wait for the dialog to close and return the input."""
        self.master.wait_window(self)
        return self.result


class PlayerTypeDialog(ctk.CTkToplevel):
    """Dialog for selecting player type."""
    def __init__(self, master, player_number):
        super().__init__(master)
        self.title(f"Player {player_number} Type")
        self.geometry("450x200")
        self.resizable(False, False)
        self.result = None

        # Set up the UI
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Prompt
        label = ctk.CTkLabel(frame, text=f"What type is Player {player_number}?",
                            font=ctk.CTkFont(size=16))
        label.pack(pady=(20, 30))

        # Button frame
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=10)

        # Buttons
        human_btn = ctk.CTkButton(button_frame, text="Human", command=lambda: self.set_result('1'))
        human_btn.pack(side="left", padx=10, expand=True)

        random_btn = ctk.CTkButton(button_frame, text="Random", command=lambda: self.set_result('2'))
        random_btn.pack(side="left", padx=10, expand=True)

        ai_btn = ctk.CTkButton(button_frame, text="AI", command=lambda: self.set_result('3'))
        ai_btn.pack(side="left", padx=10, expand=True)

        # Make modal
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: self.set_result('2'))  # Default to Random

        # Center the dialog on parent
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - width) // 2
        y = master.winfo_rooty() + (master.winfo_height() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Wait for the dialog to close
        master.wait_window(self)

    def set_result(self, value):
        """Set the result and close the dialog."""
        self.result = value
        self.grab_release()
        self.destroy()


class ModelTypeDialog(ctk.CTkToplevel):
    """Dialog for selecting ML model type."""
    def __init__(self, master):
        super().__init__(master)
        self.title("AI Model Type")
        self.geometry("450x200")
        self.resizable(False, False)
        self.result = None

        # Set up the UI
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Prompt
        label = ctk.CTkLabel(frame, text="Please select the machine learning model:",
                            font=ctk.CTkFont(size=16))
        label.pack(pady=(20, 30))

        # Button frame
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=10)

        # Buttons
        lr_btn = ctk.CTkButton(button_frame, text="Linear Regression",
                              command=lambda: self.set_result('1'))
        lr_btn.pack(side="left", padx=10, expand=True)

        nn_btn = ctk.CTkButton(button_frame, text="Neural Network",
                              command=lambda: self.set_result('2'))
        nn_btn.pack(side="left", padx=10, expand=True)

        # Make modal
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: self.set_result('1'))  # Default to LR

        # Center the dialog on parent
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - width) // 2
        y = master.winfo_rooty() + (master.winfo_height() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Wait for the dialog to close
        master.wait_window(self)

    def set_result(self, value):
        """Set the result and close the dialog."""
        self.result = value
        self.grab_release()
        self.destroy()


class ArchetypeDialog(ctk.CTkToplevel):
    """Dialog for selecting AI archetype."""
    def __init__(self, master):
        super().__init__(master)
        self.title("AI Archetype")
        self.geometry("500x200")
        self.resizable(False, False)
        self.result = None

        # Set up the UI
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Prompt
        label = ctk.CTkLabel(frame, text="Please select the AI archetype:",
                            font=ctk.CTkFont(size=16))
        label.pack(pady=(20, 30))

        # Button frame
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=10)

        # Buttons
        lit_btn = ctk.CTkButton(button_frame, text="Literalist",
                                command=lambda: self.set_result('1'))
        lit_btn.pack(side="left", padx=10, expand=True)

        con_btn = ctk.CTkButton(button_frame, text="Contrarian",
                                command=lambda: self.set_result('2'))
        con_btn.pack(side="left", padx=10, expand=True)

        com_btn = ctk.CTkButton(button_frame, text="Comedian",
                                command=lambda: self.set_result('3'))
        com_btn.pack(side="left", padx=10, expand=True)

        # Make modal
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: self.set_result('1'))  # Default to Literalist

        # Center the dialog on parent
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - width) // 2
        y = master.winfo_rooty() + (master.winfo_height() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Wait for the dialog to close
        master.wait_window(self)

    def set_result(self, value):
        """Set the result and close the dialog."""
        self.result = value
        self.grab_release()
        self.destroy()


class StartingJudgeDialog(ctk.CTkToplevel):
    """Dialog for selecting the starting judge."""
    def __init__(self, master, player_count):
        super().__init__(master)
        self.title("Starting Judge Selection")
        self.geometry("600x200")
        self.resizable(False, False)
        self.result = None

        # Set up the UI
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Prompt
        label = ctk.CTkLabel(frame, text="Please select the starting judge:",
                            font=ctk.CTkFont(size=16))
        label.pack(pady=(20, 30))

        # Button frame
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=10)

        # Create buttons for each player
        for i in range(player_count):
            player_btn = ctk.CTkButton(button_frame, text=f"Player {i+1}",
                                      command=lambda j=i+1: self.set_result(j))
            player_btn.pack(side="left", padx=5, expand=True)

        # Make modal
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: self.set_result(1))  # Default to Player 1

        # Center the dialog on parent
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - width) // 2
        y = master.winfo_rooty() + (master.winfo_height() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Wait for the dialog to close
        master.wait_window(self)

    def set_result(self, value):
        """Set the result and close the dialog."""
        self.result = value
        self.grab_release()
        self.destroy()


class RedAppleSelectionDialog(ctk.CTkToplevel):
    """Dialog for selecting a red apple card."""
    def __init__(self, master, player, red_apples, green_apple):
        super().__init__(master)
        self.title(f"{player.get_name()}'s Red Apple Selection")
        self.geometry("750x600")
        self.resizable(False, False)
        self.result = None

        # Set up the UI
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Display green apple at the top
        green_frame = ctk.CTkFrame(frame)
        green_frame.pack(pady=10)

        green_label = ctk.CTkLabel(green_frame, text="Green Apple:",
                                  font=ctk.CTkFont(size=14, weight="bold"))
        green_label.pack(pady=5)

        green_card = GreenAppleCard(green_frame, green_apple.get_adjective(),
                                   ", ".join(green_apple.get_synonyms() or []))
        green_card.pack(pady=5)

        # Prompt
        prompt_label = ctk.CTkLabel(frame, text=f"{player.get_name()}, please select a red apple:",
                                   font=ctk.CTkFont(size=16))
        prompt_label.pack(pady=10)

        # Cards frame
        cards_frame = ctk.CTkFrame(frame)
        cards_frame.pack(fill="both", expand=True, pady=10)

        # Create scrollable frame for cards
        scrollable_frame = ctk.CTkScrollableFrame(cards_frame, width=700, height=300)
        scrollable_frame.pack(fill="both", expand=True)

        # Grid for cards
        for i, apple in enumerate(red_apples):
            row = i // 3
            col = i % 3

            card_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            card_frame.grid(row=row, column=col, padx=10, pady=10)

            card = RedAppleCard(
                card_frame,
                apple.get_noun(),
                apple.get_description(),
                command=lambda idx=i: self.set_result(idx)
            )
            card.pack()

        # Make modal
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: self.set_result(0))  # Default to first card

        # Center the dialog on parent
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = master.winfo_rootx() + (master.winfo_width() - width) // 2
        y = master.winfo_rooty() + (master.winfo_height() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Wait for the dialog to close
        master.wait_window(self)

    def set_result(self, value):
        """Set the result and close the dialog."""
        self.result = value
        self.grab_release()
        self.destroy()


class JudgeSelectionDialog(ctk.CTkToplevel):
    """Dialog for the judge to select the winning red apple."""
    def __init__(self, master, judge, submissions, green_apple):
        super().__init__(master)
        self.title(f"{judge.get_name()}'s Judge Selection")
        self.geometry("750x600")
        self.resizable(False, False)
        self.result = None
        self.players = list(submissions.keys())

        # Set up the UI
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Display green apple at the top
        green_frame = ctk.CTkFrame(frame)
        green_frame.pack(pady=10)
