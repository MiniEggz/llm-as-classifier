JSON_PREFILLS = {
    "google/gemma-4-E2B-it": "".join(
        ["```", "json", "\n", "{", "\n", '"', "label", '":', ' "']
    ),
    "google/gemma-4-E2B": "".join(["{", "\n", "  ", '"', "label", '":', ' "']),
    "google/gemma-4-E4B-it": "".join(['{"', "label", '":', ' "']),
    "google/gemma-4-E4B": "".join(['{"', "label", '":', ' "']),
    "google/gemma-4-31B-it": "".join(
        [
            "<|channel>",
            "thought",
            "\n",
            "<channel|>",
            "```",
            "json",
            "\n",
            '{"',
            "label",
            '":',
            ' "',
        ]
    ),
    "google/gemma-4-31B": "".join(['{"', "label", '":', ' "']),
    "meta-llama/Llama-3.2-1B-Instruct": "{",
    "meta-llama/Llama-3.2-1B": "".join(
        ["```", "json", "\n", "{\n", " ", ' "', "label", '":', ' "']
    ),
    "meta-llama/Llama-3.2-3B-Instruct": "".join(['{"', "label", '":', ' "']),
    "meta-llama/Llama-3.2-3B": "".join(['{"', "label", '":', ' "']),
    "Qwen/Qwen2.5-7B-Instruct": "".join(['{"', "label", '":', ' "']),
    "Qwen/Qwen2.5-7B": "".join(['{"', "label", '":', ' "']),
    "Qwen/Qwen3-8B": "".join(
        ["<think>", "\n\n", "</think>", "\n\n", '{"', "label", '":', ' "']
    ),
    "Qwen/Qwen3-8B-Base": "".join(['{"', "label", '":', ' "']),
    "Qwen/Qwen3-14B": "".join(
        ["<think>", "\n\n", "</think>", "\n\n", '{"', "label", '":', ' "']
    ),
    "Qwen/Qwen3-14B-Base": "".join(['{"', "label", '":', ' "']),
    "Qwen/Qwen3-32B": "".join(
        ["<think>", "\n\n", "</think>", "\n\n", '{"', "label", '":', ' "']
    ),
    "Qwen/Qwen3.5-9B": "".join(
        ["<think>", "\n\n", "</think>", "\n\n", '{"', "label", '":', ' "']
    ),
    "Qwen/Qwen3.5-9B-Base": "".join(["{", "\n", " ", ' "', "label", '":', ' "']),
    "Qwen/Qwen3.5-4B": "".join(
        ["<think>", "\n\n", "</think>", "\n\n", '{"', "label", '":', ' "']
    ),
    "Qwen/Qwen3.5-4B-Base": "".join(['{"', "label", '":', ' "']),
    "Qwen/Qwen3.5-2B": "".join(
        ["<think>", "\n\n", "</think>", "\n\n", '{"', "label", '":', ' "']
    ),
    "Qwen/Qwen3.5-2B-Base": "".join(['{"', "label", '":', ' "']),
}
