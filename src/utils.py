from src import Small_LLM_Model
import json
from src import Function
from collections.abc import Callable
import time
from functools import wraps


def get_vocab(llm: Small_LLM_Model) -> dict[str, int]:
    with open(llm.get_path_to_vocab_file()) as f:
        return json.load(f)


def get_inverted_vocab(llm: Small_LLM_Model) -> dict[int, str]:
    return {
        value: key.replace(
            'Ġ', ' '
        ).replace(
            'Ċ', '\n'
        ).replace(
            'ĉ', '\t'
        ).replace('Ď', '\r')
        for key, value
        in get_vocab(llm).items()
        }


def get_function(function_name: str, functions: list[Function]) -> Function:
    for fn in functions:
        if fn.name == function_name:
            return fn
    raise ValueError(f"'{function_name}' not found in definitions")


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper() -> None:
        start: float = time.perf_counter()
        func()
        end: float = time.perf_counter()
        minutes: int = int((end - start) // 60)
        seconds: int = int((end - start) % 60)
        print(f"Run completed in {minutes} minutes and {seconds} seconds")
    return wrapper
