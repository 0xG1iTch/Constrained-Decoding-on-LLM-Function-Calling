import json
from pathlib import Path
from typing import Union
from pydantic import ValidationError
from .models import FunctionDefinition, Prompt


def load_prompts(path: Union[str, Path]) -> list[Prompt]:
    """Load and validate test prompts from a JSON file."""
    prompts: list[Prompt] = []
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: input file not found at {path}")
        return prompts
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}")
        return prompts

    for item in data:
        try:
            prompts.append(Prompt(**item))
        except ValidationError as e:
            print(f"Warning: skipping invalid prompt entry: {e}")
    return prompts

def load_function_definitions(path: Union[str, Path]) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    definitions: list[FunctionDefinition] = []
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        return definitions
    except json.JSONDecodeError as e:
        print(f"Error: {path} is not valid JSON: {e}")
        return definitions

    for item in data:
        try:
            definitions.append(FunctionDefinition(**item))
        except ValidationError as e:
            print(f"Skipping invalid function definition: {e}")
    return definitions
