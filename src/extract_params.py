from src import Small_LLM_Model, FunctionDef, build_params_prompt, Prompt


# def params_from_llm(
#         user_prompt: Prompt,
#         llm: Small_LLM_Model,
#         vocab: dict[str, int],
#         inv_vocab: dict[int, str],
#         function: FunctionDef
#         ) -> dict[str, str | int | float]:
#     full_prompt: str = build_params_prompt(
#         user_prompt, function
#     )
#     input_ids: list[int] = llm.encode(full_prompt)[0].tolist()
#     res: dict[str, str | int | float | bool] = {}

#     invalid_quote_string: list[int] = []
#     is_empty: list[int] = []
#     has_sign: list[int] = []
#     has_dot: list[int] = []
#     invalid_quote_number: list[int] = []
#     invalid_number: list[int] = []
#     invalid_integer: list[int] = []

#     for token_id, token_str in inv_vocab.items():
#         if not token_str:
#             is_empty.append(token_id)
#             continue
#         if '"' in token_str:
#             if not token_str.endswith('"'):
#                 invalid_quote_string.append(token_id)
#         if token_str[0] in ['-', '+']:
#             has_sign.append(token_id)
#         if '.' in token_str:
#             has_dot.append(token_id)
#         if "'" in token_str:
#             if not token_str.endswith("'"):
#                 invalid_quote_number.append(token_id)
#         if not all(c in "0123456789+-.'" for c in token_str):
#             invalid_number.append(token_id)
#         if not all(c in "0123456789+-'" for c in token_str):
#             invalid_integer.append(token_id)

#     for i, (param, param_def) in enumerate(function.parameters.items()):
#         add_str: str = (
#             f"\nThe parameter number {i} is "
#             f"\"{param}\" and its type '{param_def.type}'\n"
#             f"\n{param}="
#             )
#         add_token_ids: list[int] = llm.encode(add_str)[0].tolist()
#         input_ids.extend(add_token_ids)
#         if param_def.type == 'string':
#             input_ids.append(vocab['"'])
#             tokens: str = ''
#             seen: dict[int, int] = {}
#             while not tokens.endswith('"'):
#                 logits: list[float] = llm.get_logits_from_input_ids(input_ids)
#                 for token_id, count in seen.items():
#                     logits[token_id] -= 2.0 * count
#                 for token_id in is_empty:
#                     logits[token_id] = float('-inf')
#                 for token_id in invalid_quote_string:
#                     logits[token_id] = float('-inf')

#                 best_logit: float = max(logits)
#                 best_id: int = logits.index(best_logit)
#                 best_token: str = inv_vocab[best_id]
#                 input_ids.append(best_id)
#                 tokens += best_token
#                 if best_id in seen:
#                     seen[best_id] += 1
#                 else:
#                     seen[best_id] = 1
#                 if len(tokens) > 50:
#                     break
#             res[param] = tokens.rstrip('"')
#         elif param_def.type == 'number':
#             input_ids.append(vocab["'"])
#             tokens = ''
#             while not tokens.endswith("'"):
#                 logits = llm.get_logits_from_input_ids(input_ids)

#                 for token_id in is_empty:
#                     logits[token_id] = float('-inf')
#                 for token_id in invalid_quote_number:
#                     logits[token_id] = float('-inf')
#                 for token_id in invalid_number:
#                     logits[token_id] = float('-inf')
#                 if tokens != '':
#                     for token_id in has_sign:
#                         logits[token_id] = float('-inf')
#                 if tokens == '' or '.' in tokens:
#                     for token_id in has_dot:
#                         logits[token_id] = float('-inf')

#                 best_logit = max(logits)
#                 best_id = logits.index(best_logit)
#                 best_token = inv_vocab[best_id]
#                 input_ids.append(best_id)
#                 tokens += best_token
#             res[param] = float(tokens.rstrip("'"))
#         elif param_def.type == 'integer':
#             input_ids.append(vocab["'"])
#             tokens = ''
#             while not tokens.endswith("'"):
#                 logits = llm.get_logits_from_input_ids(input_ids)
#                 for token_id in is_empty:
#                     logits[token_id] = float('-inf')
#                 for token_id in invalid_quote_number:
#                     logits[token_id] = float('-inf')
#                 for token_id in invalid_integer:
#                     logits[token_id] = float('-inf')
#                 if tokens != '':
#                     for token_id in has_sign:
#                         logits[token_id] = float('-inf')

#                 best_logit = max(logits)
#                 best_id = logits.index(best_logit)
#                 best_token = inv_vocab[best_id]
#                 input_ids.append(best_id)
#                 tokens += best_token
#             res[param] = int(tokens.rstrip("'"))
#         elif param_def.type == 'boolean':
#             true_id: int = vocab['true']
#             false_id: int = vocab['false']
#             logits = llm.get_logits_from_input_ids(input_ids)
#             res[param] = logits[true_id] > logits[false_id]
#     return res

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

        add_str: str = (
            f"\nThe parameter number {i} is "
            f"\"{param}\" and its type '{param_def.type}'\n"
            f"\n{param}="
            )
        add_token_ids: list[int] = llm.encode(add_str)[0].tolist()
        input_ids.extend(add_token_ids)

        if param_def.type == 'string':
            input_ids.append(vocab['"'])
            tokens: str = ''
            seen: dict[int, int] = {}
            while not tokens.endswith('"'):
                logits: list[float] = llm.get_logits_from_input_ids(
                    input_ids
                    )
                for token_id in range(len(logits)):
                    if token_id in seen:
                        logits[token_id] -= 2.0 * seen[token_id]
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
                if best_id in seen:
                    seen[best_id] += 1
                else:
                    seen[best_id] = 1
                if len(tokens) > 50:
                    break
            res[param] = tokens.rstrip('"')

        elif param_def.type == 'number':
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

        elif param_def.type == 'integer':
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
        elif param_def.type == 'boolean':
            true_id: int = vocab['true']
            false_id: int = vocab['false']
            logits = llm.get_logits_from_input_ids(input_ids)
            res[param] = logits[true_id] > logits[false_id]

    return res