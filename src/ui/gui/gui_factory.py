"""Factory for creating GUI instances based on configuration."""

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.interface.game_interface import GameInterface

class GUIFactory:
    """Factory for creating GUI instances based on configuration."""

    @staticmethod
    def create_gui(config: Dict[str, Any]) -> "GameInterface":
        """
        Create a GUI instance based on configuration.

        Args:
            config: The GUI configuration dictionary

        Returns:
            An instance of GameInterface
        """
        # Get the framework from config
        framework = config.get("framework", "tkinter")

        # Create the GUI based on the framework
        creators = {
            "tkinter": GUIFactory._create_tkinter_gui,
            "customtkinter": GUIFactory._create_customtkinter_gui,
            "pygame": GUIFactory._create_pygame_gui
        }

        creator = creators.get(framework)
        if not creator:
            logging.warning(f"Unknown GUI framework: {framework}. Falling back to tkinter.")
            creator = GUIFactory._create_tkinter_gui

        return creator(config.get(framework, {}))

    @staticmethod
    def _create_tkinter_gui(config: Dict[str, Any]) -> "GameInterface":
        """Create a Tkinter GUI."""
        try:
            from src.ui.gui.tkinter.tkinter_ui import TkinterUI
            return TkinterUI()
        except ImportError as e:
            logging.error(f"Failed to import TkinterUI: {e}")
            raise

    @staticmethod
    def _create_customtkinter_gui(config: Dict[str, Any]) -> "GameInterface":
        """Create a CustomTkinter GUI."""
        try:
            from src.ui.gui.customtkinter.customtkinter_ui import CustomTkinterUI
            theme = config.get("theme", "dark-blue")
            return CustomTkinterUI(theme=theme)
        except ImportError as e:
            logging.error(f"Failed to import CustomTkinterUI: {e}. Falling back to tkinter.")
            return GUIFactory._create_tkinter_gui({})

    @staticmethod
    def _create_pygame_gui(config: Dict[str, Any]) -> "GameInterface":
        """Create a Pygame GUI."""
        try:
            from src.ui.gui.pygame.pygame_ui import PygameUI
            resolution = config.get("resolution", [1024, 768])
            fps = config.get("fps", 60)
            return PygameUI(resolution=resolution, fps=fps)
        except ImportError as e:
            logging.error(f"Failed to import PygameUI: {e}. Falling back to tkinter.")
            return GUIFactory._create_tkinter_gui({})
