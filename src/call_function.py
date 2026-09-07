from src import Small_LLM_Model, FunctionDef, Prompt, build_function_calling_prompt


def _get_remaining_suffixes(functions_names: list[str], generated: str) -> list[str]:
    remaining_suffixes: list[str] = []
    for name in functions_names:
        if name.startswith(generated):
            step: int = len(generated)
            suffix: str = name[step:]
            remaining_suffixes.append(suffix)
    return remaining_suffixes

def _is_token_prefix_of_any_suffix(token_str: str, remaining_suffixes: list[str]) -> bool:
    return any(suffix.startswith(token_str) for suffix in remaining_suffixes)

def function_name_from_llm(
        user_prompt: Prompt,
        llm: Small_LLM_Model,
        inv_vocab: dict[int, str],
        functions: list[FunctionDef]
        ) -> str:
    full_prompt: str = build_function_calling_prompt(
        user_prompt, functions
    )

    functions_names: list[str] = [func.name for func in functions]
    input_ids: list[int] = llm.encode(full_prompt)[0].tolist()
    generated: str = ''

    while generated not in functions_names:

        remaining_suffixes: list[str] = _get_remaining_suffixes(functions_names, generated)

        logits: list[float] = llm.get_logits_from_input_ids(input_ids)
        for token_id in range(len(logits)):
            token_str: str = inv_vocab.get(token_id, '')
            if not token_str:
                logits[token_id] = float('-inf')
                continue
            if not _is_token_prefix_of_any_suffix(token_str, remaining_suffixes):
                logits[token_id] = float('-inf')

        best_logit: float = max(logits)
        best_id: int = logits.index(best_logit)
        best_token: str = inv_vocab[best_id]
        input_ids.append(best_id)
        generated += best_token

    return generated