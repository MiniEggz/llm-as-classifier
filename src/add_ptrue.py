import argparse
import re

import pandas as pd
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from classification_tree_builder import TreeBuilder
from prompt_generators import IndependentJudge
from tree_aggregators import TreeAggregator


# will start with the independent confidence judge
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="conf_config.yaml",
        help="File path to config yaml file.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="prompts.yaml",
        help="File path to all prompts.",
    )
    return parser.parse_args()


def parse_percentage_output(text):
    try:
        match = re.search(r"(\d+\.?\d*)\s*%", text)

        if match:
            return float(match.group(1)) / 100.0
    except:
        print(text)

    return -1


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

    model_name = config["model_name"]
    dataset_name = config["dataset"]
    # already have data in the results file
    # load results file
    results_path = config["results_file"]
    dataset = load_dataset("csv", data_files=results_path, split="train")

    if config.get("testing", False):
        dataset = dataset.select(range(10))

    # generate all prompts for confidence judging
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sys_prompt = prompts[dataset_name][config.get("sys_prompt", "ptrue_sys_prompt")]
    user_prompt = prompts[dataset_name][config.get("user_prompt", "ptrue_prompt")]

    prompt_generator = IndependentJudge(tokenizer, sys_prompt, user_prompt)
    dataset = dataset.map(prompt_generator.map_messages)

    def tokenize_messages(row):
        prompt = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return {"input_prompt": prompt}

    dataset = dataset.map(tokenize_messages)

    # get estimate at max number of tokens
    # TODO: pull out into utils
    num_to_sample = len(dataset) // 20 + 5
    top_rows = sorted(dataset, key=lambda x: len(x["input_prompt"]), reverse=True)[
        :num_to_sample
    ]
    token_counts = [len(tokenizer.encode(row["input_prompt"])) for row in top_rows]
    max_token_count = max(token_counts) + 50  # TODO: make this buffer in config
    print("Max token count:")
    print(max_token_count)
    max_model_len = 2 ** (max_token_count - 1).bit_length()

    print("Initialising prompts.")
    input_prompts = list(dataset["input_prompt"])

    # initialise tree builder
    eoa_tokens = EOA_TOKENS[model_name]
    cls_class_labels = ["True", "False"]
    tree_builder = TreeBuilder(model_name, cls_class_labels, single_token_classes=True)

    data_points = zip(list(dataset["id"]), list(dataset["input_prompt"]))
    aggregations = {}
    print("Running inference.")
    for id, input_prompt in tqdm(
        data_points, total=len(dataset), desc="Running inference"
    ):
        tree_builder.score_tree(input_prompt, eoa_tokens)
        aggregator = TreeAggregator(tree_builder.tree)
        _, _, probs = aggregator.predict("probability_tree")
        aggregations[id] = {
            "conf": probs["True"],
        }

    # save results
    csv_save_path = config["conf_csv"]
    print(f"Saving confidence inference to {csv_save_path}...")
    base_df = pd.DataFrame(dataset)
    base_df = base_df.drop(columns=["confidence"])
    agg_df = pd.DataFrame(
        [{"id": id, "confidence": r["conf"]} for id, r in aggregations.items()]
    )
    df = base_df.merge(agg_df, on="id", how="left")

    df.to_csv(csv_save_path, index=False)
    print("Saved.")
