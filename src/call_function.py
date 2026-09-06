from src import Small_LLM_Model, Function, build_function_calling_prompt


def function_name_from_llm(
        user_prompt: str,
        llm: Small_LLM_Model,
        inv_vocab: dict[int, str],
        functions: list[Function]
        ) -> str:
    full_prompt: str = build_function_calling_prompt(
        user_prompt, functions
    )
    functions_names: list[str] = [func.name for func in functions]
    input_ids: list[int] = llm.encode(full_prompt)[0].tolist()
    generated: str = ''
    while generated not in functions_names:
        remaining_suffixes: list[str] = []
        for name in functions_names:
            if name.startswith(generated):
                remaining_suffix: str = name[len(generated):]
                remaining_suffixes.append(remaining_suffix)
        logits: list[float] = llm.get_logits_from_input_ids(input_ids)
        for token_id in range(len(logits)):
            token_str: str = inv_vocab.get(token_id, '')
            if not token_str:
                logits[token_id] = float('-inf')
                continue
            if not any(
                suffix.startswith(token_str)
                for suffix
                in remaining_suffixes
            ):
                logits[token_id] = float('-inf')
        best_logit: float = max(logits)
        best_id: int = logits.index(best_logit)
        best_token: str = inv_vocab[best_id]
        input_ids.append(best_id)
        generated += best_token
    return generated
