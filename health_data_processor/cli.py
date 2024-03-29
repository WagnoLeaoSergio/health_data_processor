"""CLI interface for health_data_processor project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""

from . import processor
from .mlops import init, run_dummy_model


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m health_data_processor` and `$ health_data_processor `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    init()
    # run_dummy_model()
    # processor.train_model()
