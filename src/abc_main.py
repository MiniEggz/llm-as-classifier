import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import disable_caching
from tqdm import tqdm
from transformers import AutoTokenizer

from classification_tree_builder import TreeBuilder
from data_loaders import (
    Banking77Adapter,
    CLINC150Adapter,
    FeverAdapter,
    MMLUProAdapter,
    PubMedQAAdapter,
    SST5Adapter,
    TwentyNGAdapter,
)
from json_prefills import JSON_PREFILLS
from prompt_generators import (
    Banking77BasePromptGenerator,
    Banking77PromptGenerator,
    CLINC150BasePromptGenerator,
    CLINC150PromptGenerator,
    FeverBasePromptGenerator,
    FeverPromptGenerator,
    MMLUPromptGenerator,
    SST5BasePromptGenerator,
    SST5PromptGenerator,
    TwentyNGBasePromptGenerator,
    TwentyNGPromptGenerator,
)
from tree_aggregators import TreeAggregator
from utils import get_class_labels

disable_caching()


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))


def find_over_length(dataset, max_model_len, tokenizer, patience=20):
    char_lens = np.array([len(p) for p in dataset["input_prompt"]])
    order = np.argsort(-char_lens)

    over, under_streak = [], 0
    for idx in order:
        count = count_tokens(dataset[idx]["input_prompt"], tokenizer)
        if count > max_model_len:
            over.append(int(idx))
            under_streak = 0
        else:
            under_streak += 1
        if under_streak >= patience:
            break
    return over


def get_token_id(tok_str, tokenizer):
    encoded = tokenizer.encode(tok_str, add_special_tokens=False)
    print(encoded)
    if len(encoded) != 1:
        raise ValueError(f"Encoded token is too long. For {tok_str}, got {encoded}.")
    return encoded[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="File path to config yaml file.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="prompts.yaml",
        help="File path to all prompts.",
    )
    return parser.parse_args()


DATA_ADAPTERS = {
    "mmlu": MMLUProAdapter,
    "pmqa": PubMedQAAdapter,
    "fever": FeverAdapter,
    "banking77": Banking77Adapter,
    "twentyng": TwentyNGAdapter,
    "sst5": SST5Adapter,
    "clinc150": CLINC150Adapter,
}
PROMPT_GENERATORS = {
    "mmlu": MMLUPromptGenerator,
    "fever": FeverPromptGenerator,
    "banking77": Banking77PromptGenerator,
    "twentyng": TwentyNGPromptGenerator,
    "sst5": SST5PromptGenerator,
    "clinc150": CLINC150PromptGenerator,
}
BASE_PROMPT_GENERATORS = {
    "fever": FeverBasePromptGenerator,
    "banking77": Banking77BasePromptGenerator,
    "twentyng": TwentyNGBasePromptGenerator,
    "sst5": SST5BasePromptGenerator,
    "clinc150": CLINC150BasePromptGenerator,
}
EOA_TOKENS = {
    "google/gemma-4-E2B-it": [1, 106],  # <eos>, <turn|>
    "google/gemma-4-E2B": [1, 107],  # <eos>, \n
    "google/gemma-4-E4B-it": [1, 106],  # <eos>, <turn|>
    "google/gemma-4-E4B": [1, 107],  # <eos>, \n
    "google/gemma-4-31B-it": [1, 106],  # <eos>, <turn|>
    "google/gemma-4-31B": [1, 107],  # <eos>, \n
    "meta-llama/Llama-3.2-3B-Instruct": [128009],  # <|eot_id|>
    "meta-llama/Llama-3.2-3B": [128009, 198],  # <|eot_id|>, \n
    "meta-llama/Llama-3.2-1B-Instruct": [128009],  # <|eot_id|>
    "meta-llama/Llama-3.2-1B": [128009, 198],  # <|eot_id|>, \n
    "Qwen/Qwen2.5-7B-Instruct": [151645],  # <|im_end|>
    "Qwen/Qwen2.5-7B": [151643, 198],  # <|endoftext|>, \n
    "Qwen/Qwen3-8B": [151645],  # <|im_end|>
    "Qwen/Qwen3-8B-Base": [151643, 198],  # <|endoftext|>, \n
    "Qwen/Qwen3-14B": [151645],  # <|im_end|>
    "Qwen/Qwen3-14B-Base": [151643, 198],  # <|endoftext|>, \n
    "Qwen/Qwen3-32B": [151645],  # <|im_end|>
    "Qwen/Qwen3.5-4B": [248046],  # <|im_end|>
    "Qwen/Qwen3.5-4B-Base": [248044],  # <|endoftext|>
    "Qwen/Qwen3.5-9B": [248046],  # <|im_end|>
    "Qwen/Qwen3.5-9B-Base": [248044],  # <|endoftext|>
}

if __name__ == "__main__":
    # parse args and load config
    args = parse_args()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    with open(args.prompts, "r") as prompt_file:
        prompts = yaml.safe_load(prompt_file)

    # get options (whether abc prompting)
    adapter_dict = config.get("adapter_dict", None)

    print("initialising data adapter...")
    dataset_name = config["dataset"]
    if dataset_name not in DATA_ADAPTERS.keys():
        raise ValueError(
            f"Dataset {dataset_name} not recognised. Please use one of: {', '.join(list(DATA_ADAPTERS.keys()))}."
        )
    data_adapter = DATA_ADAPTERS[dataset_name](
        split=config["split"], adapter_dict=adapter_dict
    )

    # load dataset
    print("loading dataset...")
    dataset = data_adapter.dataset

    # load model tokeniser
    model_name = config["model_name"]
    print(f"initialising tokeniser for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create prompts for each data point
    # TODO: create method data_adapter.add_input_prompts(tokenizer, template)
    #       prompt will be different for each dataset.
    print("creating input prompt for each row...")
    #  sys_prompt = "SYSTEM PROMPT"
    #  prompt_template = "The options are {OPTIONS}.\n\nInput text:\n{INPUT_TEXT}\n\nContext:\n{CONTEXT}"

    if not config.get("base_model", False):
        sys_prompt = prompts[dataset_name][config["sys_prompt"]]
        prompt_template = prompts[dataset_name][config["user_prompt"]]

        prompt_generator = PROMPT_GENERATORS[dataset_name](
            tokenizer, sys_prompt, prompt_template
        )
        dataset = dataset.map(prompt_generator.map_messages)

        # cut dataset for testing
        if config["testing"]:
            dataset = dataset.select(range(20))

        ds_size = len(dataset)
        print("Length of dataset:", ds_size)

        # add tokenised messages
        enable_thinking = config.get("enable_thinking", False)

        def messages_to_prompts(row):
            prompt = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
            return {"input_prompt": prompt}

        dataset = dataset.map(messages_to_prompts)
    else:  # if using base model
        prompt_template = prompts[dataset_name][config["user_prompt"]]
        prompt_generator = BASE_PROMPT_GENERATORS[dataset_name](
            tokenizer, prompt_template
        )
        dataset = dataset.map(prompt_generator.map_prompt)

        # cut dataset for testing
        if config["testing"]:
            dataset = dataset.select(range(20))

        ds_size = len(dataset)
        print("Length of dataset:", ds_size)

    prefill_json = config.get("prefill_json", False)
    prompt_appendix = config.get("prompt_appendix", None)
    if prompt_appendix is not None:
        if prefill_json:
            prompt_appendix += JSON_PREFILLS[model_name]

        def append_to_prompt(row):
            return {"input_prompt": row["input_prompt"] + prompt_appendix}

        dataset = dataset.map(append_to_prompt)

    # check whether class labels are all the same
    class_labels = get_class_labels(list(dataset["options"]))
    cls_class_labels = list(adapter_dict.keys())

    # find best estimate for max_input_len
    # TODO: make override in config
    num_to_sample = len(dataset) // 20 + 5
    top_rows = sorted(dataset, key=lambda x: len(x["input_prompt"]), reverse=True)[
        :num_to_sample
    ]
    token_counts = [len(tokenizer.encode(row["input_prompt"])) for row in top_rows]
    max_token_count = max(token_counts) + 50  # TODO: make this buffer in config
    print("Max token count:")
    print(max_token_count)
    max_model_len = 2 ** (max_token_count - 1).bit_length()

    max_model_len = min(max_model_len, 16384)

    print("Max model length is:")
    print(max_model_len)

    num_over = sum(1 if tok_count > max_model_len else 0 for tok_count in token_counts)
    print(f"Data points over {max_model_len} (of the top 50):")
    print(num_over)

    over_len_idx = find_over_length(dataset, max_model_len, tokenizer)

    if not config.get("base_model", False):

        def truncate_over_len(row, idx):
            # convert to set for faster lookups
            over_set = set(over_len_idx)
            if idx not in over_set:
                return {
                    "messages": row["messages"],
                    "input_prompt": row["input_prompt"],
                }

            row["messages"] = prompt_generator.map_messages(
                row, max_model_len=max_model_len
            )["messages"]
            return {"messages": row["messages"], **messages_to_prompts(row)}

        dataset = dataset.map(truncate_over_len, with_indices=True)

    else:

        def truncate_over_len_base(row, idx):
            # convert to set for faster lookups
            over_set = set(over_len_idx)
            if idx not in over_set:
                return {
                    "input_prompt": row["input_prompt"],
                }

            return prompt_generator.map_prompt(row)

        dataset = dataset.map(truncate_over_len_base, with_indices=True)

    preds_csv_path = config["preds_csv"]
    eoa_tokens = EOA_TOKENS[model_name]
    # add " and "} as an end token for json output
    if "json" in preds_csv_path:
        eoa_tokens.append(get_token_id('"', tokenizer))
        eoa_tokens.append(get_token_id('"}', tokenizer))

    # build all trees
    tree_builder = TreeBuilder(
        model_name,
        cls_class_labels,
        max_model_len=max_model_len,
        single_token_classes=True,
    )

    prompts = list(dataset["input_prompt"])
    data_points = zip(list(dataset["id"]), list(dataset["input_prompt"]))
    aggregations = {}
    for id, input_prompt in tqdm(
        data_points, total=len(dataset), desc="Running inference"
    ):
        oom = False
        try:
            tree_builder.score_tree(input_prompt, eoa_tokens)
            aggregator = TreeAggregator(tree_builder.tree)
            pred, conf, probs = aggregator.predict("probability_tree")
            probs = {adapter_dict[chars]: prob for chars, prob in probs.items()}
            aggregations[id] = {
                "pred": adapter_dict[pred],
                "conf": conf,
                "probs": probs,
                "raw_output": pred,
            }
        except torch.cuda.OutOfMemoryError:
            oom = True

        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # save inference
    # modify slightly to for full probs
    file_name, _ = preds_csv_path.split(".")
    file_name += "-logit"
    preds_csv_path = f"{file_name}.csv"
    print(f"Saving inference to {preds_csv_path}...")
    base_df = pd.DataFrame(dataset)
    agg_df = pd.DataFrame(
        [
            {
                "id": id,
                "pred": r["pred"],
                "confidence": r["conf"],
                "probs": r["probs"],
                "raw_output": r["raw_output"],
            }
            for id, r in aggregations.items()
        ]
    )
    df = base_df.merge(agg_df, on="id", how="left")
    df.to_csv(preds_csv_path, index=False)
    print("Saved.")
