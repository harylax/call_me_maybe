from llm_sdk.llm_sdk import Small_LLM_Model

def main() -> None:
    llm: Small_LLM_Model = Small_LLM_Model()
    prompt: str = "What is the sum of 265 and 345?"

    input_ids: list[int] = llm.encode(prompt)[0].tolist()
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
