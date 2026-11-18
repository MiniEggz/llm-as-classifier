import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from classification_tree_builder import GenerationTree
from data_loaders import (
    Banking77Adapter,
    CLINC150Adapter,
    FeverAdapter,
    MMLUProAdapter,
    PubMedQAAdapter,
    SST5Adapter,
    TwentyNGAdapter,
)
from tree_aggregators import AGGREGATION_METHODS, TreeAggregator

DATA_ADAPTERS = {
    "mmlu": MMLUProAdapter,
    "pmqa": PubMedQAAdapter,
    "fever": FeverAdapter,
    "banking77": Banking77Adapter,
    "twentyng": TwentyNGAdapter,
    "sst5": SST5Adapter,
    "clinc150": CLINC150Adapter,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="File path to config yaml file.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Aggregate all trees available.",
    )
    return parser.parse_args()


def aggregate_from_config(config):
    # get options (whether abc prompting)
    adapter_dict = config.get("adapter_dict", None)
    # NEED tree path (and then the indices from dataset)

    # set up dataset adapter
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
    # NEED dataset_name

    # load model tokeniser
    model_name = config["model_name"]
    print(f"initialising tokeniser for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    run_name = config["run_name"]
    trees_dir = Path(f"./results/trees/{run_name}")

    # for each data point, do each aggregation
    agg_names = list(AGGREGATION_METHODS.keys())
    aggregations = {agg_name: {} for agg_name in agg_names}
    for idx in tqdm(dataset["id"]):
        tree = GenerationTree.load(trees_dir / f"{idx}.json", tokenizer)
        aggregator = TreeAggregator(tree)
        for agg_method in agg_names:
            pred, conf, probs = aggregator.predict(agg_method)
            aggregations[agg_method][idx] = {
                "pred": pred,
                "conf": conf,
                "probs": probs,
            }

    model_save_name, _, run_appendix = run_name.split("_")
    run_appendix = run_appendix.replace("create-tree-greedy", "greedy-tree")

    # save each to a file
    base_df = pd.DataFrame(dataset)

    save_dir = Path("./results/")

    for agg_name in agg_names:
        save_agg_name = agg_name.replace("_", "-")
        records = aggregations[agg_name]
        agg_df = pd.DataFrame(
            [
                {
                    "id": idx,
                    "pred": r["pred"],
                    "confidence": r["conf"],
                    "probs": r["probs"],
                }
                for idx, r in records.items()
            ]
        )
        df = base_df.merge(agg_df, on="id", how="left")
        df.to_csv(
            save_dir
            / f"{model_save_name}_{dataset_name}_{run_appendix}-{save_agg_name}.csv",
            index=False,
        )


if __name__ == "__main__":
    # load config
    args = parse_args()

    not_aggregated = []

    if args.all:
        trees_dir = Path("./results/trees")
        for dir_path in trees_dir.iterdir():
            print(f"Looking at {dir_path}.")
            if not dir_path.is_dir():
                continue
            dir_name = dir_path.name
            model, dataset, method_name = dir_name.split("_")
            cfg_path = Path("./cfg") / Path(method_name) / f"{dir_name}.yaml"
            with open(cfg_path, "r") as config_file:
                config = yaml.safe_load(config_file)
            try:
                aggregate_from_config(config)
            except:
                not_aggregated.append(config)

        print("Not aggregated:")
        for na in not_aggregated:
            print(na)
    else:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)

        aggregate_from_config(config)
