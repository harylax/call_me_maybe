from llm_sdk.llm_sdk import Small_LLM_Model
import json
from typing import Any


def functions_definition() -> str:
    content: list[dict[str, Any]] = []
    with open('data/input/functions_definition.json') as f:
        content = json.load(f)
    res: list[str] = []
    for df in content:
        params_dict: dict[str, str] = {}
        for param, type in df['parameters'].items():
            params_dict[param] = type['type']
        params_str: str = ', '.join(
            f"{key}: {value}"
            for key, value
            in params_dict.items()
            )
        res.append(
            f" - function name: {df['name']}, "
            f"parameter(s): {params_str}, "
            f"description: {df['description']}"
            )
    return '\n'.join(fn for fn in res)


def build_function_calling_prompt(prompt: str) -> str:
    return (
        "You are a function calling assistant...\n\n"
        f"Available functions:\n{functions_definition()}\n\n"
        f"User question: {prompt}\n"
        "Function to call: "
    )


def main() -> None:
    llm: Small_LLM_Model = Small_LLM_Model()
    user_prompt: str = "What is the sum of 265 and 345?"

    full_prompt: str = build_function_calling_prompt(user_prompt)

    input_ids: list[int] = llm.encode(full_prompt)[0].tolist()
    targets: list[str] = [
        "fn_substitute_string_with_regex", "fn_get_square_root",
        "fn_reverse_string", "fn_greet", "fn_add_numbers"
        ]
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
            token_str: str = llm.decode([token_id])
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
