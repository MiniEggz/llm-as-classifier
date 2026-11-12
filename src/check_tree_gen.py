import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from data_loaders import (
    Banking77Adapter,
    CLINC150Adapter,
    FeverAdapter,
    SST5Adapter,
    TwentyNGAdapter,
)

DATA_ADAPTERS = {
    "fever": FeverAdapter,
    "banking77": Banking77Adapter,
    "twentyng": TwentyNGAdapter,
    "sst5": SST5Adapter,
    "clinc150": CLINC150Adapter,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-files",
        action="store_true",
        help="Aggregate all trees available.",
    )
    parser.add_argument(
        "--under-2k",
        action="store_true",
        help="Only check ones under 2k.",
    )
    return parser.parse_args()


@dataclass
class ModelRun:
    model_name: str
    dataset_name: str
    method_name: str
    finished: bool = False
    rerun_ids: Optional[list] = None

    @classmethod
    def from_dir_name(cls, dir_name: str):
        if not isinstance(dir_name, str):
            raise ValueError("dir_name must be a string.")

        split = dir_name.split("_")
        if len(split) != 3:
            raise ValueError(
                "dir_name must contain {MODEL}_{DATASET}_{METHOD_NAME} only."
            )

        model_name, dataset_name, method_name = split

        return cls(
            model_name=model_name, dataset_name=dataset_name, method_name=method_name
        )

    @property
    def dir_name(self):
        return "_".join([self.model_name, self.dataset_name, self.method_name])


def get_runs(tree_path: Path) -> tuple[list[str], list[ModelRun]]:
    """Get all runs that are either complete or partially complete.

    Args:
        tree_path (Path): path to the trees.

    Returns:
        list[str]: list of all datasets that have run.
        list[ModelRun]: list of model runs.
    """
    tree_path = Path(tree_path)

    run_dirs = [p for p in tree_path.iterdir() if p.is_dir()]
    model_runs = [ModelRun.from_dir_name(dir.name) for dir in run_dirs]
    datasets = sorted(list(set([model_run.dataset_name for model_run in model_runs])))

    return datasets, model_runs


if __name__ == "__main__":
    args = parse_args()
    tree_path = Path("./results/trees")
    dataset_names, model_runs = get_runs(tree_path)

    over_ids = {}
    if args.under_2k:
        invalid_ids_file = Path("over_2k.json")
        over_ids = dict(json.loads(invalid_ids_file.read_text()))
        over_ids = {key: set(values) for key, values in over_ids.items()}

    runs_by_ds = {dataset_name: [] for dataset_name in dataset_names}
    for model_run in model_runs:
        runs_by_ds[model_run.dataset_name].append(model_run)

    for dataset_name in dataset_names:
        data_adapter = DATA_ADAPTERS[dataset_name](split="test")
        dataset = data_adapter.dataset

        ids = list(dataset["id"])
        id_set = set(ids)

        assert len(dataset) == len(ids)

        ids_is_ordered = ids == sorted(ids)
        print(f"IDs in order? {ids_is_ordered}")

        for model_run in runs_by_ds[dataset_name]:
            print(f"Checking {model_run.dir_name}...")
            completed_ids = {
                int(p.stem) for p in (tree_path / model_run.dir_name).rglob("*.json")
            }

            overlaps = len(id_set & completed_ids)
            if overlaps == len(ids):
                print("Model run fully complete.")
                model_run.finished = True
                continue

            remaining_id_set = id_set - completed_ids
            if args.under_2k:
                remaining_id_set = remaining_id_set - over_ids.get(dataset_name, set())
            remaining_ids = list(remaining_id_set)

            print(f"Missed data points: {len(remaining_ids)}.")
            if args.write_files:
                print("Writing files.")
                rerun_dir = Path("./tree_reruns")
                rerun_dir.mkdir(exist_ok=True)
                Path(f"{rerun_dir / model_run.dir_name}.json").write_text(
                    json.dumps(sorted(remaining_ids))
                )
                print(f"Written to ./tree_reruns/{model_run.dir_name}.json")
            input("ENTER to keep going.")

        print(f"Dataset '{dataset_name}' complete.")
