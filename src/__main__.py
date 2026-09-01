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


def functions_definition(functions: list[Function]) -> str:
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
            f"description: {func.descr}"
            )
    return '\n'.join(fn for fn in res)


def build_function_calling_prompt(
        prompt: str,
        functions: list[Function]
        ) -> str:
    return (
        "You are a function calling assistant...\n\n"
        f"Available functions:\n{functions_definition(functions)}\n\n"
        f"User question: {prompt}\n"
        "Function to call: "
    )


def _get_vocab(llm: Small_LLM_Model) -> dict[str, int]:
    with open(llm.get_path_to_vocab_file()) as f:
        return json.load(f)


def get_inverted_vocab(llm: Small_LLM_Model) -> dict[int, str]:
    return {
        value: key.replace('Ġ', ' ')
        for key, value
        in _get_vocab(llm).items()
        }


def main() -> None:
    llm: Small_LLM_Model = Small_LLM_Model()
    vocab: dict[int, str] = get_inverted_vocab(llm)

    path_to_functions_definition: str = 'data/input/functions_definition.json'
    functions: list[Function] = parse_functions_definition(
        path_to_functions_definition
        )
    user_prompt: str = "What is the sum of 265 and 345?"

    full_prompt: str = build_function_calling_prompt(
        user_prompt,
        functions
        )

    input_ids: list[int] = llm.encode(full_prompt)[0].tolist()
    targets: list[str] = [func.name for func in functions]
    tokens: str = ""
    while tokens not in targets:
        left_to_find_options: list[str] = []
        for target in targets:
            if target.startswith(tokens):
                left_to_find: str = target[len(tokens):]
                left_to_find_options.append(left_to_find)

        logits: list[float] = llm.get_logits_from_input_ids(input_ids)

        for i in range(len(logits)):
            token_id: int = i
            token_str: str = vocab.get(token_id, '')
            if not token_str:
                logits[token_id] = float("-inf")
                continue
            if not any(
                left_to_find.startswith(token_str)
                for left_to_find
                in left_to_find_options
            ):
                logits[token_id] = float("-inf")

        best_logit: float = max(logits)
        best_id: int = logits.index(best_logit)
        best_token: str = llm.decode([best_id])
        input_ids.append(best_id)
        tokens += best_token

    print(tokens)


if __name__ == "__main__":
    main()
