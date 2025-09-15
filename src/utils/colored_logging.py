"""
Utility functions for colored logging output using ANSI escape codes.
"""

import logging


class Colors:
    """ANSI color codes for terminal output."""

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Reset
    RESET = "\033[0m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"


def colorize_text(text: str, color: str) -> str:
    """
    Apply color to text using ANSI escape codes.

    Args:
        text: The text to colorize
        color: The ANSI color code (use Colors class constants)

    Returns:
        Colored text with reset code at the end
    """
    return f"{color}{text}{Colors.RESET}"


def log_error_red(logger: logging.Logger, message: str) -> None:
    """
    Log an error message in bright red color.

    Args:
        logger: The logger instance to use
        message: The error message to log
    """
    colored_message = colorize_text(message, Colors.BRIGHT_RED)
    logger.error(colored_message)


def log_warning_yellow(logger: logging.Logger, message: str) -> None:
    """
    Log a warning message in bright yellow color.

    Args:
        logger: The logger instance to use
        message: The warning message to log
    """
    colored_message = colorize_text(message, Colors.BRIGHT_YELLOW)
    logger.warning(colored_message)


def log_success_green(logger: logging.Logger, message: str) -> None:
    """
    Log a success message in bright green color.

    Args:
        logger: The logger instance to use
        message: The success message to log
    """
    colored_message = colorize_text(message, Colors.BRIGHT_GREEN)
    logger.info(colored_message)


def log_info_blue(logger: logging.Logger, message: str) -> None:
    """
    Log an info message in bright blue color.

    Args:
        logger: The logger instance to use
        message: The info message to log
    """
    colored_message = colorize_text(message, Colors.BRIGHT_BLUE)
    logger.info(colored_message)


def log_debug_cyan(logger: logging.Logger, message: str) -> None:
    """
    Log a debug message in bright cyan color.

    Args:
        logger: The logger instance to use
        message: The debug message to log
    """
    colored_message = colorize_text(message, Colors.BRIGHT_CYAN)
    logger.debug(colored_message)
