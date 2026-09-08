from src import Small_LLM_Model, FunctionDef, build_params_prompt, Prompt


def _mask_logits(
        logits: list[float],
        param_type: str,
        inv_vocab: dict[int, str],
        generated: str = '',
        seen: dict[int, int] | None = None
        ) -> None:
    if param_type == 'string':
        for token_id in range(len(logits)):
            if seen is not None:
                if token_id in seen:
                    logits[token_id] -= 2.0 * seen[token_id]
            token_str: str = inv_vocab.get(token_id, '')
            if not token_str:
                logits[token_id] = float('-inf')
                continue
            if '"' in token_str:
                if not token_str.endswith('"'):
                    logits[token_id] = float('-inf')

    elif param_type == 'number':
        for token_id in range(len(logits)):
            token_str = inv_vocab.get(token_id, '')
            if not token_str:
                logits[token_id] = float('-inf')
                continue
            if token_str[0] in ['-', '+']:
                if generated != '':
                    logits[token_id] = float('-inf')
                    continue
            if '.' in token_str:
                if generated == '':
                    logits[token_id] = float('-inf')
                    continue
                if '.' in generated:
                    logits[token_id] = float('-inf')
                    continue
            if "'" in token_str:
                if not token_str.endswith("'"):
                    logits[token_id] = float('-inf')
                    continue
            if not all(c in "0123456789+-.'" for c in token_str):
                logits[token_id] = float('-inf')

    elif param_type == 'integer':
        for token_id in range(len(logits)):
            token_str = inv_vocab.get(token_id, '')
            if not token_str:
                logits[token_id] = float('-inf')
                continue
            if token_str[0] in ['-', '+']:
                if generated != '':
                    logits[token_id] = float('-inf')
                    continue
            if "'" in token_str:
                if not token_str.endswith("'"):
                    logits[token_id] = float('-inf')
                    continue
            if not all(c in "0123456789+-'" for c in token_str):
                logits[token_id] = float('-inf')


def _constrained_gen(
        param_type: str,
        vocab: dict[str, int],
        inv_vocab: dict[int, str],
        llm: Small_LLM_Model,
        input_ids: list[int],
        closing_char: str,
        seen: dict[int, int] | None = None
        ) -> str:
    input_ids.append(vocab[closing_char])
    generated: str = ''
    while not generated.endswith(closing_char):
        logits: list[float] = llm.get_logits_from_input_ids(
                    input_ids
                    )
        _mask_logits(logits, param_type, inv_vocab, generated, seen)
        best_logit: float = max(logits)
        best_id: int = logits.index(best_logit)
        best_token: str = inv_vocab[best_id]

        input_ids.append(best_id)
        generated += best_token

        if seen is not None:
            if best_id in seen:
                seen[best_id] += 1
            else:
                seen[best_id] = 1

        if len(generated) > 50:
            break
    return generated.strip()


def params_from_llm(
        user_prompt: Prompt,
        llm: Small_LLM_Model,
        vocab: dict[str, int],
        inv_vocab: dict[int, str],
        function: FunctionDef
        ) -> dict[str, str | int | float | bool]:
    full_prompt: str = build_params_prompt(
        user_prompt, function
    )
    input_ids: list[int] = llm.encode(full_prompt)[0].tolist()
    res: dict[str, str | int | float | bool] = {}
    for i, (param, param_def) in enumerate(function.parameters.items()):

        add_prompt: str = (
            f"\nThe parameter number {i} is "
            f"\"{param}\" and its type '{param_def.type}'\n"
            f"\n{param}="
            )
        add_token_ids: list[int] = llm.encode(add_prompt)[0].tolist()
        input_ids.extend(add_token_ids)

        if param_def.type == 'string':
            seen: dict[int, int] = {}
            generated: str = _constrained_gen(
                'string', vocab, inv_vocab, llm, input_ids, '"', seen
            )
            res[param] = generated.rstrip('"').strip()

        elif param_def.type == 'number':
            generated = _constrained_gen(
                'number', vocab, inv_vocab, llm, input_ids, "'"
            )
            res[param] = float(generated.rstrip("'"))

        elif param_def.type == 'integer':
            generated = _constrained_gen(
                'integer', vocab, inv_vocab, llm, input_ids, "'"
            )
            res[param] = int(generated.rstrip("'"))

        elif param_def.type == 'boolean':
            true_id: int = vocab['true']
            false_id: int = vocab['false']
            logits = llm.get_logits_from_input_ids(input_ids)
            res[param] = logits[true_id] > logits[false_id]

    return res
