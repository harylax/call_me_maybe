from llm_sdk.llm_sdk import Small_LLM_Model  # type: ignore
import json
from typing import Any
from src import (
    timer, parse_args, get_vocab, get_inverted_vocab,
    Function, parse_functions_definition, parse_prompts,
    function_name_from_llm, params_from_llm, get_function
)
from pathlib import Path
import sys


@timer
def main() -> None:
    llm: Small_LLM_Model = Small_LLM_Model()

    input_path, functions_definition_path, output_path = parse_args()

    output: list[dict[str, Any]] = []

    vocab: dict[str, int] = get_vocab(llm)
    inv_vocab: dict[int, str] = get_inverted_vocab(llm)

    functions: list[Function] = parse_functions_definition(
        functions_definition_path
        )

    for user_prompt in parse_prompts(input_path):
        llm_fn_name: str = function_name_from_llm(
            user_prompt, llm, inv_vocab, functions
            )

        try:
            function: Function = get_function(llm_fn_name, functions)
        except ValueError as err:
            print(f"Value Error: {err}")
            raise SystemExit()

        llm_params: dict[str, str | int | float] = params_from_llm(
            user_prompt, llm, vocab, inv_vocab, function
        )

        output.append({
            'prompt': user_prompt,
            'name': llm_fn_name,
            'parameters': llm_params
        })
        print(f"prompt: {user_prompt}")
        print(f"name: {llm_fn_name}")
        print(f"parameters: {llm_params}")
    path: Path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
    except OSError as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
