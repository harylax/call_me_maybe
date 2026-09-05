from llm_sdk.llm_sdk import Small_LLM_Model
import json
from typing import Any
from pydantic import BaseModel


class Function(BaseModel):
    name: str
    params: dict[str, str]
    descr: str
    returns: str


def parse_functions_definition(path: str) -> list[Function]:
    content: list[dict[str, Any]] = []
    with open(path) as f:
        content = json.load(f)
    res: list[Function] = []
    for func in content:
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


def _functions_definition(functions: list[Function]) -> str:
    res: list[str] = []
    for func in functions:
        params_str: str = ', '.join(
            f"{key}: {value}"
            for key, value
            in func.params.items()
            )
        res.append(
            f" - function name: {func.name}, "
            f"parameter(s): {params_str}, "
            f"returns: {func.returns}, "
            f"description: {func.descr}"
            )
    return '\n'.join(fn for fn in res)


def build_function_calling_prompt(
        prompt: str,
        functions: list[Function]
        ) -> str:
    return (
        "You are a function calling assistant...\n\n"
        f"Available functions:\n{_functions_definition(functions)}\n\n"
        f"User main prompt: {prompt}\n"
        "Function to call: "
    )


def build_params_prompt(
        prompt: str,
        function: Function
        ) -> str:
    params_str: str = ', '.join(
        f"parameter '{key}' (type: {value})"
        for key, value
        in function.params.items()
        )
    return (
        f"User main prompt: {prompt}\n"
        f"Function called: {_functions_definition([function])}\n"
        f"Extract the value of the following parameters: {params_str}.\n"
        # "Don't add extra informations.\n"
        # "If the value to be extracted is part of the user main prompt: just exract the value, don't add extra informations.\n"
        # "Preserve uppercase and lowercase.\n"
        # "Preserve spelling, capitalisation, uppercase, lowercase, "
        # "mixed case, numbers in the user main prompt.\n"
        # "Don't correct user's mistakes.\n"
        # "As reminder, vowels are: 'aeiouAEIOU' "
        # "and numbers are: '0123456789'.\n"
        "Parameter(s) value(s):\n"
    )


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


def function_name_from_llm(
        user_prompt: str,
        llm: Small_LLM_Model,
        inv_vocab: dict[int, str],
        functions: list[Function]
        ) -> str:
    full_prompt: str = build_function_calling_prompt(
        user_prompt, functions
    )
    targets: list[str] = [func.name for func in functions]
    input_ids: list[int] = llm.encode(full_prompt)[0].tolist()
    tokens: str = ''
    while tokens not in targets:
        left_to_find_options: list[str] = []
        for target in targets:
            if target.startswith(tokens):
                left_to_find: str = target[len(tokens):]
                left_to_find_options.append(left_to_find)
        logits: list[float] = llm.get_logits_from_input_ids(input_ids)
        for token_id in range(len(logits)):
            token_str: str = inv_vocab.get(token_id, '')
            if not token_str:
                logits[token_id] = float('-inf')
                continue
            if not any(
                left_to_find.startswith(token_str)
                for left_to_find
                in left_to_find_options
            ):
                logits[token_id] = float('-inf')
        best_logit: float = max(logits)
        best_id: int = logits.index(best_logit)
        best_token: str = inv_vocab[best_id]
        input_ids.append(best_id)
        tokens += best_token
    return tokens


def params_from_llm(
        user_prompt: str,
        llm: Small_LLM_Model,
        vocab: dict[str, int],
        inv_vocab: dict[int, str],
        function: Function
        ) -> dict[str, str | int | float]:
    full_prompt: str = build_params_prompt(
        user_prompt, function
    )
    input_ids: list[int] = llm.encode(full_prompt)[0].tolist()
    res: dict[str, str | int | float] = {}
    for param, type in function.params.items():
        add_str: str = (
            f"\nkey=\"{param}\" (type: {type})\n"
            "value="
            )
        add_token_ids: list[int] = llm.encode(add_str)[0].tolist()
        input_ids.extend(add_token_ids)
        # for id in add_token_ids:
        #     input_ids.append(id)
        if type == 'string':
            input_ids.append(vocab['"'])
            tokens: str = ''
            while not tokens.endswith('"'):
                logits: list[float] = llm.get_logits_from_input_ids(input_ids)
                for token_id in range(len(logits)):
                    token_str: str = inv_vocab.get(token_id, '')
                    if not token_str:
                        logits[token_id] = float('-inf')
                        continue
                    if '"' in token_str:
                        if not token_str.endswith('"'):
                            logits[token_id] = float('-inf')
                best_logit: float = max(logits)
                best_id: int = logits.index(best_logit)
                best_token: str = inv_vocab[best_id]
                input_ids.append(best_id)
                tokens += best_token
                if len(tokens) > 50:
                    break
            res[param] = tokens.rstrip('"')
        elif type == 'number':
            input_ids.append(vocab["'"])
            tokens = ''
            while not tokens.endswith("'"):
                logits = llm.get_logits_from_input_ids(input_ids)
                for token_id in range(len(logits)):
                    token_str = inv_vocab.get(token_id, '')
                    if not token_str:
                        logits[token_id] = float('-inf')
                        continue
                    if token_str[0] in ['-', '+']:
                        if tokens != '':
                            logits[token_id] = float('-inf')
                            continue
                    if '.' in token_str:
                        if tokens == '':
                            logits[token_id] = float('-inf')
                            continue
                        if '.' in tokens:
                            logits[token_id] = float('-inf')
                            continue
                    if "'" in token_str:
                        if not token_str.endswith("'"):
                            logits[token_id] = float('-inf')
                            continue
                    if not all(c in "0123456789+-.'" for c in token_str):
                        logits[token_id] = float('-inf')
                best_logit = max(logits)
                best_id = logits.index(best_logit)
                best_token = inv_vocab[best_id]
                input_ids.append(best_id)
                tokens += best_token
            res[param] = float(tokens.rstrip("'"))
        elif type == 'integer':
            input_ids.append(vocab["'"])
            tokens = ''
            while not tokens.endswith("'"):
                logits = llm.get_logits_from_input_ids(input_ids)
                for token_id in range(len(logits)):
                    token_str = inv_vocab.get(token_id, '')
                    if not token_str:
                        logits[token_id] = float('-inf')
                        continue
                    if token_str[0] in ['-', '+']:
                        if tokens != '':
                            logits[token_id] = float('-inf')
                            continue
                    if "'" in token_str:
                        if not token_str.endswith("'"):
                            logits[token_id] = float('-inf')
                            continue
                    if not all(c in "0123456789+-'" for c in token_str):
                        logits[token_id] = float('-inf')
                best_logit = max(logits)
                best_id = logits.index(best_logit)
                best_token = inv_vocab[best_id]
                input_ids.append(best_id)
                tokens += best_token
            res[param] = int(tokens.rstrip("'"))
        elif type == 'boolean':
            true_id: int = vocab['true']
            false_id: int = vocab['false']
            logits = llm.get_logits_from_input_ids(input_ids)
            res[param] = logits[true_id] > logits[false_id]

    return res


def get_function(function_name: str, functions: list[Function]) -> Function:
    for fn in functions:
        if fn.name == function_name:
            return fn
    raise ValueError(f"'{function_name}' not found in definitions")


def parse_prompts(path: str) -> list[str]:
    content: list[dict[str, str]] = []
    with open(path) as f:
        content = json.load(f)
    return [prompt['prompt'] for prompt in content]


def parse_args() -> tuple[str, str, str]:
    from argparse import ArgumentParser
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

from collections.abc import Callable
import time
def timer(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> None:
        start: float = time.perf_counter()
        func(*args, **kwargs)
        end: float = time.perf_counter()
        minutes: int = int((end - start) // 60)
        seconds: int = int((end - start) % 60)
        print(f"Run completed in {minutes} minutes and {seconds} seconds")
    return wrapper

@timer
def run() -> None:
    llm: Small_LLM_Model = Small_LLM_Model()

    input_path, functions_definition_path, output_path = parse_args()

    output: list[dict[str, Any]] = []

    vocab: dict[str, int] = get_vocab(llm)
    inv_vocab: dict[int, str] = get_inverted_vocab(llm)

    functions: list[Function] = parse_functions_definition(
        functions_definition_path
        )

    for user_prompt in parse_prompts(input_path):
    # for user_prompt in ["Greet shrek"]:
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
    from pathlib import Path
    path: Path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


# def main() -> None:
#     llm: Small_LLM_Model = Small_LLM_Model()

#     input_path, functions_definition_path, output_path = parse_args()

#     output: list[dict[str, Any]] = []

#     vocab: dict[str, int] = get_vocab(llm)
#     inv_vocab: dict[int, str] = get_inverted_vocab(llm)

#     functions: list[Function] = parse_functions_definition(
#         functions_definition_path
#         )

#     # for user_prompt in parse_prompts(input_path):
#     for user_prompt in ["Greet shrek"]:
#         llm_fn_name: str = function_name_from_llm(
#             user_prompt, llm, inv_vocab, functions
#             )

#         try:
#             function: Function = get_function(llm_fn_name, functions)
#         except ValueError as err:
#             print(f"Value Error: {err}")
#             raise SystemExit()

#         llm_params: dict[str, str | int | float] = params_from_llm(
#             user_prompt, llm, vocab, inv_vocab, function
#         )

#         output.append({
#             'prompt': user_prompt,
#             'name': llm_fn_name,
#             'parameters': llm_params
#         })
#         print(f"prompt: {user_prompt}")
#         print(f"name: {llm_fn_name}")
#         print(f"parameters: {llm_params}")
#     from pathlib import Path
#     path: Path = Path(output_path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(output_path, 'w') as f:
#         json.dump(output, f, indent=2)


if __name__ == "__main__":
    # main()
    run()