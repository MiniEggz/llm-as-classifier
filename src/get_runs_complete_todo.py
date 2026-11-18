import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

DATASETS = ["banking77", "clinc150", "fever", "sst5", "twentyng"]
MODELS = [
    "llama3-2-1b",
    "llama3-2-3b",
    "gemma4-e2b",
    "gemma4-e4b",
    "gemma4-31b",
    "qwen2-5-7b",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "qwen3-5-4b",
    "qwen3-5-9b",
]

METHOD_BLACKLIST = [
    "our-method",
    "cot-tp",
    "cot",
    "judge-json",
    "greedy-inference-json-no-conf-abc-logit",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/",
        help="Results directory.",
    )
    parser.add_argument(
        "--shell-script",
        action="store_true",
        help="Whether to write files, or just for information.",
    )
    parser.add_argument(
        "--file-out",
        type=str,
        default="to_run.sh",
        help="Output file path.",
    )
    return parser.parse_args()


@dataclass
class ModelRun:
    model_name: str
    dataset_name: str
    method_name: str
    is_base: bool

    @classmethod
    def from_stem(cls, stem: str):
        if not isinstance(stem, str):
            raise ValueError("stem must be a string.")

        split = stem.split("_")
        if len(split) != 3:
            raise ValueError("stem must contain {MODEL}_{DATASET}_{METHOD_NAME} only.")

        model_name, dataset_name, method_name = split
        if "base" in method_name:
            is_base = True
        else:
            is_base = False

        return cls(
            model_name=model_name,
            dataset_name=dataset_name,
            method_name=method_name,
            is_base=is_base,
        )

    @property
    def stem(self):
        return "_".join([self.model_name, self.dataset_name, self.method_name])

    @property
    def model_ds_str(self):
        return f"{self.model_name}_{self.dataset_name}"


@dataclass
class ModelRuns:
    model_runs: list[ModelRun]

    def get_by_model(
        self, model_name: str, model_runs: Optional[list[ModelRun]] = None
    ) -> list[ModelRun]:
        if model_runs is None:
            model_runs = self.model_runs
        return [mr for mr in model_runs if mr.model_name == model_name]

    def get_by_dataset(
        self, dataset_name: str, model_runs: Optional[list[ModelRun]] = None
    ) -> list[ModelRun]:
        if model_runs is None:
            model_runs = self.model_runs
        return [mr for mr in model_runs if mr.dataset_name == dataset_name]

    def get_by_model_and_ds(self, model_name: str, dataset_name: str) -> list[ModelRun]:
        return self.get_by_dataset(dataset_name, self.get_by_model(model_name))


def get_all_methods(results_dir: Path):
    # get all combinations of DATASETS and MODELS
    model_ds_combos = list(product(MODELS, DATASETS))
    model_ds_strs = [f"{model}_{ds}" for model, ds in model_ds_combos]
    model_ds_methods = {md_str: set() for md_str in model_ds_strs}

    # get all files in results
    model_runs = [ModelRun.from_stem(p.stem) for p in results_dir.rglob("*.csv")]

    # look at the start of each file (first 2 _)
    # add to dict of dataset+model combos
    for mr in model_runs:
        mr_set = model_ds_methods.get(mr.model_ds_str, None)
        if mr_set is None:
            continue
        if any(blacklisted in mr.method_name for blacklisted in METHOD_BLACKLIST):
            continue
        mr_set.add(mr.method_name)

    # all methods (run on any model+ds combination)
    return set.union(*list(model_ds_methods.values()))


def check_all_runs(results_dir: Path, add_stops: bool = False):
    # get all combinations of DATASETS and MODELS
    model_ds_combos = list(product(MODELS, DATASETS))
    model_ds_strs = [f"{model}_{ds}" for model, ds in model_ds_combos]
    model_ds_methods = {md_str: set() for md_str in model_ds_strs}

    # get all files in results
    model_runs = [ModelRun.from_stem(p.stem) for p in results_dir.rglob("*.csv")]

    # look at the start of each file (first 2 _)
    # add to dict of dataset+model combos
    for mr in model_runs:
        mr_set = model_ds_methods.get(mr.model_ds_str, None)
        if mr_set is None:
            continue
        if any(blacklisted in mr.method_name for blacklisted in METHOD_BLACKLIST):
            continue
        mr_set.add(mr.method_name)

    # all methods (run on any model+ds combination)
    all_methods = set.union(*list(model_ds_methods.values()))

    # check which ones appear in everything
    complete_methods = set.intersection(*list(model_ds_methods.values()))

    # print
    print("Union")
    print(all_methods)
    print("Intersection")
    print(complete_methods)

    for key, mr_set in model_ds_methods.items():
        print(f"Model/dataset combo: {key}")
        print(all_methods - mr_set)
        if add_stops:
            input("Enter for next.")


def py_file_from_config(config_path: Path) -> str:
    # different rules for different configs...
    if "create-tree" in str(config_path):
        return "create_tree.py"
    elif "ptrue" in str(config_path):
        return "add_ptrue.py"
    elif "judge" in str(config_path):
        return "add_confidence_tf.py"
    # because every method name will have
    # -logit appended if it existed somewhere
    elif "-logit" in str(config_path):
        return "abc_main.py"
    else:
        return "main.py"


TREE_METHODS = {
    "create-tree-greedy": {
        "greedy-tree-full-log-prob",
        "greedy-tree-length-norm",
        "greedy-tree-google-length-norm",
        "greedy-tree-full-log-prob-decision",
        "greedy-tree-length-norm-decision",
        "greedy-tree-google-length-norm-decision",
        "greedy-tree-probability-tree",
        "greedy-tree-greedy",
    },
    "create-tree-greedy-json": {
        "greedy-tree-json-full-log-prob",
        "greedy-tree-json-length-norm",
        "greedy-tree-json-google-length-norm",
        "greedy-tree-json-full-log-prob-decision",
        "greedy-tree-json-length-norm-decision",
        "greedy-tree-json-google-length-norm-decision",
        "greedy-tree-json-probability-tree",
        "greedy-tree-json-greedy",
    },
}


def get_model_todos(results_dir: Path, add_stops: bool = False) -> list[str]:
    """Get all the model todos as a list containing all
    shell commands to run.

    Args:
        results_dir (Path): path to the results directory.
        add_stops (bool): whether to stop as going through things.

    Returns:
        list[str]: list of shell script lines.
    """
    config_dir = Path("./cfg")
    # get all files in results
    model_runs = ModelRuns(
        [ModelRun.from_stem(p.stem) for p in results_dir.rglob("*.csv")]
    )

    all_methods = get_all_methods(results_dir)

    shell_script_lines = []
    # loop through all models
    for model in MODELS:
        print(f"For model {model}:")
        # for each dataset per model
        for dataset in DATASETS:
            print(f"\tFor dataset {dataset}")
            reduced = model_runs.get_by_model_and_ds(model, dataset)

            method_set = set(mr.method_name for mr in reduced)
            if not method_set:
                print("\t\tComplete.")
                continue

            methods_to_run = all_methods - method_set

            for tree_method, aggregates in TREE_METHODS.items():
                if aggregates.issubset(methods_to_run):
                    methods_to_run = methods_to_run - aggregates
                    methods_to_run.add(tree_method)

            # get full path to config
            configs_to_run = [
                config_dir / method / f"{model}_{dataset}_{method}.yaml"
                for method in methods_to_run
                if "base" not in method
            ]

            # based on config path, get python run name
            commands = [
                f'python {py_file_from_config(config_file)} --config="{str(config_file)}"'
                for config_file in configs_to_run
            ]

            if len(commands) > 0:
                commands.insert(0, f"# {model} + {dataset}")
                commands.append("\n")
                commands = [command.replace("-logit", "") for command in commands]

            for command in commands:
                print(f"\t\t\t{command}")
            # add to shell script
            shell_script_lines.extend(commands)

    return shell_script_lines


def write_shell_script(script_name, sh_lines):
    with open(script_name, "w") as f:
        f.write("\n".join(sh_lines) + "\n")


def sh_script_len(sh_script):
    return len([line for line in sh_script if (line != "\n" and line[0] != "#")])


if __name__ == "__main__":
    args = parse_args()

    results_dir = Path(args.results_dir)

    if args.shell_script:
        sh_lines = get_model_todos(results_dir)
        write_shell_script(args.file_out, sh_lines)
        print(f"Written {sh_script_len(sh_lines)} runs to {args.file_out}.")
    else:
        check_all_runs(results_dir, True)
