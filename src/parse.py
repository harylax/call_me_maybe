from pydantic import BaseModel  # type: ignore
import json
from typing import Any
import sys
from argparse import ArgumentParser


class Function(BaseModel):
    name: str
    params: dict[str, str]
    descr: str
    returns: str


def _json_load(path) -> list[dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except OSError as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as err:
        print(f"JSONDecodeError: {err}")
        sys.exit(1)


def parse_functions_definition(path: str) -> list[Function]:
    res: list[Function] = []
    for func in _json_load(path):
        params_dict: dict[str, str] = {}
        for param, type in func['parameters'].items():
            params_dict[param] = type['type']
        res.append(Function(
            name=func['name'],
            params=params_dict,
            descr=func['description'],
            returns=func['returns']['type']
        ))
    return res


def parse_prompts(path: str) -> list[str]:
    return [prompt['prompt'] for prompt in _json_load(path)]


def parse_args() -> tuple[str, str, str]:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--input",
        default='data/input/function_calling_tests.json'
        )
    parser.add_argument(
        "--functions_definition",
        default='data/input/functions_definition.json'
        )
    parser.add_argument(
        "--output",
        default='data/output/function_calling_results.json'
        )
    args = parser.parse_args()
    return args.input, args.functions_definition, args.output
