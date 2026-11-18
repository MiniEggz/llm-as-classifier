import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

from data_loaders import (
    Banking77Adapter,
    CLINC150Adapter,
    FeverAdapter,
    SST5Adapter,
    TwentyNGAdapter,
)
from prompt_generators import (
    Banking77JevPromptGenerator,
    CLINC150JevPromptGenerator,
    FeverJevPromptGenerator,
    SST5JevPromptGenerator,
    TwentyNGJevPromptGenerator,
)

API_URL = "https://api.typesafe.ai/v1/systemone"
RETRY_STATUSES = {429, 529}

DATA_ADAPTERS = {
    "fever": FeverAdapter,
    "banking77": Banking77Adapter,
    "twentyng": TwentyNGAdapter,
    "sst5": SST5Adapter,
    "clinc150": CLINC150Adapter,
}
PROMPT_GENERATORS = {
    "fever": FeverJevPromptGenerator,
    "banking77": Banking77JevPromptGenerator,
    "twentyng": TwentyNGJevPromptGenerator,
    "sst5": SST5JevPromptGenerator,
    "clinc150": CLINC150JevPromptGenerator,
}


def all_options_identical(options: list) -> bool:
    """Whether all lists of options are each identical.

    Args:
        options (list): list of option lists -- 2d list
            where the 1st index is the set of options for
            data point i; 2nd index is the individual label
            option within the data point.

    Returns:
        bool: whether all options are identical.
    """
    if not options:
        return True
    first = options[0]
    return all(option == first for option in options)


def get_class_labels(options: list) -> list:
    """Get class labels from a list of options.

    Args:
        options (list): list of option lists -- 2d list
            where the 1st index is the set of options for
            data point i; 2nd index is the individual label
            option within the data point.

    Returns:
        list: list of options, reduced if possible.
    """
    if all_options_identical(options):
        return options[0]
    else:
        raise NotImplementedError(
            "The case for varying options has not yet been implemented."
        )


def get_criteria(dataset_name, class_labels):
    if dataset_name == "fever":
        return {
            "SUPPORTS": "The evidence provided supports the claim.",
            "REFUTES": "The evidence provided refutes the claim.",
            "NOT ENOUGH INFO": "There is not enough evidence to make a clear decision.",
        }
    return {class_label: None for class_label in class_labels}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="prompts.yaml",
        help="File path to all prompts.",
    )
    parser.add_argument(
        "--overs-file",
        type=Path,
        default="over_2k.json",
        help="File containing dict for over X tokens for all datasets.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="How often to save the outputs.",
    )
    return parser.parse_args()


def classify(
    body: dict, api_key: str | None = None, max_retries: int = 5, timeout: float = 60.0
):
    api_key = api_key or os.environ["TYPESAFE_API_KEY"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries + 1):
        resp = requests.post(API_URL, json=body, headers=headers, timeout=timeout)
        if resp.status_code in RETRY_STATUSES and attempt < max_retries:
            time.sleep(min(2**attempt, 30))  # 1, 2, 4, 8, 16... seconds
            continue
        if resp.status_code == 422:
            # Validation error: the body says which field is wrong.
            raise ValueError(f"Jev rejected the request: {resp.text}")
        resp.raise_for_status()
        break

    data = resp.json()

    answer = data["answers"]["label"]
    label = answer["choice"]
    probs = answer["probabilities"]
    conf = max(probs.values())
    api_confidence = answer.get("confidence")
    return {
        "pred": label,
        "confidence": conf,
        "probs": probs,
        "api_confidence": api_confidence,
        "raw_output": label,
    }


if __name__ == "__main__":
    args = parse_args()

    with open(args.prompts, "r") as prompt_file:
        prompts = yaml.safe_load(prompt_file)

    print("initialising data adapter...")
    dataset_name = args.dataset
    csv_path = Path(f"results/jev-1-13-0_{dataset_name}_jev.csv")
    if dataset_name not in DATA_ADAPTERS.keys():
        raise ValueError(
            f"Dataset {dataset_name} not recognised. Please use one of: {', '.join(list(DATA_ADAPTERS.keys()))}."
        )
    data_adapter = DATA_ADAPTERS[dataset_name](split="test")

    # load dataset
    print("loading dataset...")
    dataset = data_adapter.dataset

    print(f"filtering with {args.overs_file}...")
    og_len = len(dataset)
    over_ids = dict(json.loads(Path(args.overs_file).read_text()))
    over_ids = {key: set(values) for key, values in over_ids.items()}
    dataset = dataset.filter(
        lambda row: int(row["id"]) not in list(over_ids[dataset_name])
    )
    new_len = len(dataset)
    if new_len < og_len:
        print(f"Dataset reduced from {og_len} data points to {new_len}.")
    else:
        print(f"Dataset size: {len(dataset)}.")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        to_finish = df[df["pred"].isna() | (df["pred"] == "")]
        if len(to_finish) == 0:
            print("All data points have already been processed.")
            exit()
        # filter dataset
        ids_to_run = list(int(id) for id in list(to_finish["id"]))
        dataset = dataset.filter(lambda row: int(row["id"]) in ids_to_run)
        if len(dataset) < new_len:
            print(f"Dataset reduced from {new_len} data points to {len(dataset)}.")
        else:
            print(f"Dataset size: {len(dataset)}.")
    else:
        df = pd.DataFrame(dataset)

    print("creating input prompt for each row...")
    jev_instructions = prompts[dataset_name]["jev_instructions"]
    class_labels = get_class_labels(list(dataset["options"]))
    prompt_generator = PROMPT_GENERATORS[dataset_name](
        jev_instructions, get_criteria(dataset_name, class_labels)
    )
    dataset = dataset.map(prompt_generator.map_body)

    result_cols = ["pred", "conf", "probs", "api_confidence", "raw_output"]
    for col in result_cols:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)

    row_index = {int(i): idx for idx, i in zip(df.index, df["id"])}

    def save():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    ids = list(dataset["id"])
    bodies = list(dataset["body"])
    count = 0
    try:
        for id, body in tqdm(
            zip(ids, bodies), total=len(ids), desc="Running inference:"
        ):
            outputs = classify(body)
            idx = row_index[int(id)]
            for col, value in outputs.items():
                df.at[idx, col] = value
            count += 1
            if count >= args.save_every:
                save()
                count = 0
    finally:
        save()
