import argparse
import re
import warnings

import numpy as np
import pandas as pd
import yaml
from datasets import disable_caching
from transformers import AutoTokenizer

from data_loaders import (
    Banking77Adapter,
    CLINC150Adapter,
    FeverAdapter,
    SST5Adapter,
    TwentyNGAdapter,
)
from greedy_inference import GreedyInference, GreedyInferenceConf, GreedyInferenceJson
from greedy_inference_base import (
    GreedyInferenceBase,
    GreedyInferenceBaseConf,
    GreedyInferenceBaseJson,
)
from greedy_inference_cot import (
    GreedyInferenceConfCoT,
    GreedyInferenceCoT,
    GreedyInferenceJsonCoT,
)
from our_classifier import (
    FullPathGLNClassifier,
    FullPathGLNJsonClassifier,
    FullPathNLNClassifier,
    FullPathNLNJsonClassifier,
    FullPathSLNClassifier,
    FullPathSLNJsonClassifier,
    OnDivergenceClassifier,
    OnDivergenceJsonClassifier,
)
from outlines_classifier import (
    OutlinesClassifier,
    OutlinesClassifierConfPost,
    OutlinesClassifierConfPostLikert,
    OutlinesClassifierConfPre,
    OutlinesClassifierConfPreLikert,
    OutlinesClassifierJson,
)
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
from sampling_classifier import SamplingClassifier
from utils import get_class_labels

disable_caching()

# silence space regex warnings
warnings.filterwarnings("ignore", message=".*Regex parsing warning*")


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
CLASSIFIERS = {
    "outlines-no-conf": OutlinesClassifier,
    "outlines-json-no-conf": OutlinesClassifierJson,
    "outlines-json-pre": OutlinesClassifierConfPre,
    "outlines-json-post": OutlinesClassifierConfPost,
    "outlines-json-pre-likert": OutlinesClassifierConfPreLikert,
    "outlines-json-post-likert": OutlinesClassifierConfPostLikert,
    "our-classifier-full-nln": FullPathNLNClassifier,  # no length normalisation
    "our-classifier-full-nln-json": FullPathNLNJsonClassifier,  # no length normalisation
    "our-classifier-full-gln": FullPathGLNClassifier,  # google NMT length normalisation
    "our-classifier-full-gln-json": FullPathGLNJsonClassifier,  # google NMT length normalisation
    "our-classifier-full-sln": FullPathSLNClassifier,  # standard length normalisation
    "our-classifier-full-sln-json": FullPathSLNJsonClassifier,  # standard length normalisation
    "our-classifier-diverge": OnDivergenceClassifier,
    "our-classifier-diverge-json": OnDivergenceJsonClassifier,
    "greedy-inference-no-conf": GreedyInference,
    "greedy-inference-verbal": GreedyInferenceConf,
    "greedy-inference-json-no-conf": GreedyInferenceJson,
    "greedy-inference-cot": GreedyInferenceCoT,
    "greedy-inference-conf-cot": GreedyInferenceConfCoT,
    "greedy-inference-json-cot": GreedyInferenceJsonCoT,
    "greedy-inference-base-no-conf": GreedyInferenceBase,
    "greedy-inference-base-json-no-conf": GreedyInferenceBaseJson,
    "greedy-inference-base-verbal": GreedyInferenceBaseConf,
    "sampling-classifier": SamplingClassifier,
}


def parse_percentage_output(conf):
    if isinstance(conf, float):
        if conf >= 0 and conf <= 1:
            return conf
    if isinstance(conf, int):
        if conf >= 0 and conf <= 100:
            return float(conf) / 100.0
        else:
            return -1.0
    match = re.search(r"(\d+\.?\d*)\s*%", conf)

    try:
        if match:
            return float(match.group(1)) / 100.0
    except:
        pass

    return -1.0


likert_to_float = {
    "Not at all confident": 0.2,
    "Slightly confident": 0.4,
    "Moderately confident": 0.6,
    "Very confident": 0.8,
    "Completely confident": 1,
}


def parse_likert_conf(conf):
    return likert_to_float.get(conf, -1)


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

    buffer_len = config.get("buffer_len", None)

    if not config.get("base_model", False):
        sys_prompt = prompts[dataset_name][config["sys_prompt"]]
        prompt_template = prompts[dataset_name][config["user_prompt"]]

        prompt_generator = PROMPT_GENERATORS[dataset_name](
            tokenizer, sys_prompt, prompt_template, buffer_len=buffer_len
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
            tokenizer, prompt_template, buffer_len=buffer_len
        )
        dataset = dataset.map(prompt_generator.map_prompt)

        # cut dataset for testing
        if config["testing"]:
            dataset = dataset.select(range(20))

        ds_size = len(dataset)
        print("Length of dataset:", ds_size)

    prompt_appendix = config.get("prompt_appendix", None)
    if prompt_appendix:

        def append_to_prompt(row):
            return {"input_prompt": row["input_prompt"] + prompt_appendix}

        dataset = dataset.map(append_to_prompt)

    # check whether class labels are all the same
    class_labels = get_class_labels(list(dataset["options"]))

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

    # add ceiling of 32768 tokens
    max_model_len = min(max_model_len, 32768)

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

    # run inference / classification
    print("initialising classifier...")

    # TODO: make this variable based on config
    Classifier = CLASSIFIERS[config["classifier_name"]]
    if adapter_dict is None:
        cls_class_labels = class_labels
    else:
        cls_class_labels = list(adapter_dict.keys())

    classifier = Classifier(
        model_name=model_name,
        class_labels=cls_class_labels,
        max_model_len=max_model_len,
    )

    print("running inference...")
    input_prompts = list(dataset["input_prompt"])
    preds, raw_confs, raw_outputs = classifier.classify(input_prompts)

    if adapter_dict is not None:
        preds = [adapter_dict.get(pred, pred) for pred in preds]

    # TODO: maybe change this depending on how handling confidence
    if not set(raw_confs).isdisjoint(set(likert_to_float.keys())):
        confs = [parse_likert_conf(rc) for rc in raw_confs]
    elif not all(c == 1.0 for c in raw_confs):
        confs = [parse_percentage_output(rc) for rc in raw_confs]
    else:
        confs = raw_confs

    # save inference
    preds_csv_path = config["preds_csv"]
    print(f"Saving inference to {preds_csv_path}...")
    df = pd.DataFrame(dataset)
    df["pred"] = preds
    df["raw_confidence"] = raw_confs
    df["confidence"] = confs
    if raw_outputs:
        df["raw_output"] = raw_outputs

    df.to_csv(preds_csv_path, index=False)
    print("Saved.")
