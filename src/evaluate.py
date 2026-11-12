import argparse
import ast
import json
import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils import get_class_labels, get_clean_pred, get_clean_preds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="eval_config.yaml",
        help="File path to config yaml file.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="prompts.yaml",
        help="File path to all prompts.",
    )
    return parser.parse_args()


def confidence_analysis(confs: np.ndarray):
    """Provides general statistics about the confidence scores from
    the model.

    Args:
        confs (np.ndarray): 1d numpy array containing float confidence scores.

    Returns:
        dict: dictionary containing a number of confidence statistics.
    """
    confs = np.asarray(confs, dtype=float)
    finite = confs[np.isfinite(confs)]
    failed = (confs == -1) | ~np.isfinite(confs)
    valid_finite = confs[~failed]
    unique, counts = np.unique(np.round(finite, 2), return_counts=True)
    conf_counts = dict(zip(unique, counts))

    return {
        "conf_parse_rate": (conf_counts.get(-1, 0) + len(confs) - len(finite))
        / len(confs),
        "n_distinct_confidences": len(unique),
        "mean_conf": float(finite.mean()) if finite.size else np.nan,
        "mean_valid_conf": float(valid_finite.mean()) if valid_finite.size else np.nan,
        "std_conf": float(finite.std()) if finite.size else np.nan,
        "std_valid_conf": float(valid_finite.std()) if valid_finite.size else np.nan,
        "min_conf": float(finite.min()) if finite.size else np.nan,
        "min_valid_conf": float(valid_finite.min()) if valid_finite.size else np.nan,
        "max_conf": float(finite.max()) if finite.size else np.nan,
        "max_conf": float(valid_finite.max()) if valid_finite.size else np.nan,
    }


def confidence_accuracy_gap(correct: np.ndarray, confs: np.ndarray) -> float:
    """Confidence-accuracy gap to indicate whether the model is under- or
    over-confident. Over-confidence will lead to a higher score, and
    under-confidence will lead to a negative score (range=-1..1).

    Args:
        correct (np.ndarray): 1d numpy array containing whether each prediction
            was correct (0=incorrect, 1=correct).
        confs (np.ndarray): 1d numpy array of floats containing confidence in
            each prediction.

    Returns:
        float: calibration gap (in range -1..1).
    """
    return float(np.mean(confs) - np.mean(correct))


def top_label_brier(correct: np.ndarray, confs: np.ndarray) -> float:
    """Binary implementation of the brier score.

    Args:
        correct (np.ndarray): 1d numpy array containing whether each prediction
            was correct (0=incorrect, 1=correct).
        confs (np.ndarray): 1d numpy array of floats containing confidence in
            each prediction.

    Returns:
        float: binary (or top-label) brier score (range of 0..1).
    """
    correct = np.asarray(correct, dtype=float)
    confs = np.asarray(confs, dtype=float)
    return float(np.mean((confs - correct) ** 2))


def multiclass_brier(y_true: np.ndarray, probs: Optional[np.ndarray]) -> float:
    """Calculate multiclass brier score.

    Args:
        y_true (np.ndarray): correct integer label.
        probs (np.ndarray): probabilities for each data point (shape of NxK where
            N is number of data points and K is number of classes).

    Returns:
        float: multiclass brier score (range of 0..2, where lower is better).
    """
    if probs is None:
        return np.nan

    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), np.asarray(y_true, dtype=int)] = 1.0

    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


# probably don't use NLL because of how quickly it degrades
def negative_log_likelihood(
    y_true: np.ndarray, probs: Optional[np.ndarray], clip: float = 1e-12
) -> float:
    if probs is None:
        return np.nan

    probs = np.asarray(probs, dtype=float)
    p_true = probs[np.arange(len(y_true)), np.asarray(y_true, dtype=int)]

    return float(-np.mean(np.log(np.clip(p_true, clip, 1.0))))


def calculate_ece(
    preds: list[int] | np.ndarray,
    confs: list[float] | np.ndarray,
    gold_labels: list[int] | np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate expected calibration error (ECE).

    Args:
        preds (list[int] | np.ndarray): predictions from the model.
        conf (list[float] | np.ndarray): model confidence between 0-1.
        gold_labels: list[int] | np.ndarray: true labels to compare preds against.
        bins: number of bins, defaults to 10.

    Returns:
        float: ECE score.
    """
    acc = (np.array(preds) == np.array(gold_labels)).astype(float)
    confs = np.array(confs)

    bin_boundaries = np.linspace(0, 1, bins + 1)
    pos = np.digitize(confs, bin_boundaries, right=True)
    bin_ids = np.clip(pos - 1, 0, bins - 1)

    bin_counts = np.bincount(bin_ids, minlength=bins)
    sum_conf = np.bincount(bin_ids, weights=confs, minlength=bins)
    sum_acc = np.bincount(bin_ids, weights=acc, minlength=bins)

    diffs = np.abs(sum_acc - sum_conf)
    ece = diffs.sum() / len(confs)

    return ece


def correctness_auroc(correct: np.ndarray, confs: np.ndarray) -> float:
    """Calculate the AUROC (or ROC-AUC) score.

    Args:
        correct (np.ndarray): 1d numpy array containing whether each prediction
            was correct (0=incorrect, 1=correct).
        confs (np.ndarray): 1d numpy array of floats containing confidence in
            each prediction.

    Returns:
        float: AUROC score.
    """
    correct = np.asarray(correct, dtype=int)
    confs = np.asarray(confs, dtype=float)

    if correct.min() == correct.max():
        return np.nan

    return float(roc_auc_score(correct, confs))


# TODO: write is_degenerate so we can see where things go wrong


def distribute_bad_preds(
    preds: list[int],
    num_labels: int,
    gold_labels: list[int],
    use_tiebreak: Optional[str] = None,
):
    """Distributes incorrectly generated predictions (that aren't in the
    label set) among all labels that are not the gold label (i.e. all
    incorrect answers).

    Args:
        preds (list[int]): list of predictions as integer classes.
        num_labels (int): number of labels, expected to be equal to the index
            of the badly generated predictions.
        gold_labels (list[int]): list of gold labels to know what label cannot
            be generated for that instance.
        use_tiebreak (Optional[str]): if there is a tiebreak, it should be either
            "gold" or "preds" to decide whether to use the distribution
            of the true prior (from gold_labels) or the predicted prior (from
            preds).

    Returns:
        list[int]: list of predictions as integer classes, where the bad_pred_idx
            is no longer present, and has been replaced uniformly by the
            available incorrect answers.
    """
    distributed = np.zeros(num_labels)

    # set tiebreak (choose larger classes first)
    if use_tiebreak is not None:
        if use_tiebreak == "gold":
            support = np.bincount(gold_labels, minlength=num_labels)
        elif use_tiebreak == "preds":
            support = np.bincount(preds, minlength=num_labels)[:num_labels]
        else:
            raise ValueError("use_tiebreak must be: None, 'gold', or 'preds'.")
        tiebreak = -support / (support.sum() + 1) * 0.5
    else:
        tiebreak = np.zeros(num_labels)

    def get_new_pred(idx, tiebreak):
        # mask the gold label out of the decision
        masked = distributed.astype(float) + tiebreak
        masked[gold_labels[idx]] = np.inf
        # select the new prediction (not gold)
        new_pred = int(np.argmin(masked))
        # add counter in distributed
        distributed[new_pred] += 1

        return new_pred

    new_preds = [
        pred if pred != num_labels else get_new_pred(idx, tiebreak)
        for idx, pred in enumerate(preds)
    ]

    return new_preds


def eval_csv(csv_path: Union[str, Path], invalid_ids_file=Path("over_2k.json")) -> dict:
    """Evaluate model performance from one file.

    Args:
        eval_csv (str | Path): path to csv results file.

    Returns:
        dict: dictionary containing model name and results.
    """
    # extract model name
    run_name = str(Path(csv_path).stem)
    model_name, dataset_name, method_name = run_name.split("_")

    # load results file
    df = pd.read_csv(csv_path)
    df["options"] = df["options"].apply(ast.literal_eval)

    over_ids = dict(json.loads(invalid_ids_file.read_text()))
    over_ids = {key: list(set(values)) for key, values in over_ids.items()}
    df = df[~df["id"].astype(int).isin(over_ids[dataset_name])]

    # load class labels
    # TODO: make general case for variable labels
    class_labels = get_class_labels(list(df["options"]))
    label_to_idx = {label: i for i, label in enumerate(class_labels)}

    if "raw_output" in df.columns:
        oom_count = (
            df["pred"].str.contains("OOM", na=False, regex=False)
            | df["raw_output"].str.contains("OOM", na=False, regex=False)
        ).sum()
    else:
        oom_count = (df["pred"].str.contains("OOM", na=False, regex=False)).sum()

    # extract labels and preds
    print(f"Model name: {model_name}")
    labels = [label_to_idx[label] for label in df["gold_label"]]
    # TODO: make a method to get the preds
    # needs: df["pred"], class_labels
    preds = [
        label_to_idx.get(clean_pred, len(label_to_idx))
        for clean_pred in get_clean_preds(list(df["pred"]), class_labels)
    ]
    error_idx = len(label_to_idx)
    gen_errors = preds.count(error_idx)
    print(f"Num gen errors: {gen_errors}")
    # preds = [label_to_idx[pred] for pred in df["pred"]]

    # calculate performance metrics
    acc = accuracy_score(labels, preds)
    valid = np.asarray(preds) != error_idx

    valid_acc = (
        accuracy_score(np.asarray(labels)[valid], np.asarray(preds)[valid])
        if valid.any()
        else 0.0
    )
    valid_indices = list(range(len(class_labels)))
    adjusted_preds = distribute_bad_preds(
        preds, len(label_to_idx), labels, use_tiebreak="gold"
    )
    f1_macro = f1_score(
        labels, adjusted_preds, labels=valid_indices, average="macro", zero_division=0.0
    )

    # load confidences
    if "confidence" in df.columns:
        confs = list(df["confidence"])
    else:
        confs = [1 for _ in range(len(df))]

    # confidence statistics
    conf_stats = confidence_analysis(np.asarray(confs))

    incorrect_conf = confs.count(-1)
    confidences = np.array([1 if conf == -1 else conf for conf in confs])
    under_conf = np.sum(confidences < 0)
    over_conf = np.sum(confidences > 1)

    # find correctness
    correct = np.asarray(preds) == np.asarray(labels)

    ca_gap = confidence_accuracy_gap(correct, confidences)

    # brier scores
    binary_brier = top_label_brier(correct, confidences)
    probs = None
    if "probs" in df.columns:
        prob_dicts = [ast.literal_eval(p) for p in df["probs"]]
        probs = np.array(
            [[d[label] for label in class_labels] for d in prob_dicts], dtype=float
        )
    #  probs = []
    #  probs = df.get("probs", None)
    #  if probs is not None:
    #  if isinstance(probs, pd.Series):
    #  probs = list(probs)
    #  probs = [list(ast.literal_eval(p).values()) for p in probs]
    #  probs = np.asarray(probs)
    #  else:
    #  print(type(probs))
    #  input("EH?")
    brier = multiclass_brier(np.asarray(labels), probs)

    # auc
    auroc = correctness_auroc(correct, np.asarray(confidences))

    if (under_conf + over_conf) != 0:
        print("WARNING:")
        print(f"  under-confident (<0): {under_conf}")
        print(f"  over-confident (>1): {over_conf}")

    confidences = list(np.clip(confidences, 0, 1))
    ece = calculate_ece(preds, confidences, labels)

    # get bins for confidence scores
    counts, bins = np.histogram(confidences, bins=np.linspace(0, 1, 11))

    return {
        "accuracy": acc,
        "selective_accuracy": valid_acc,
        "f1_macro": f1_macro,
        "ece": ece,
        "binary_brier": binary_brier,
        "brier": brier,
        "auroc": auroc,
        "confidence_accuracy_gap": ca_gap,
        **conf_stats,
        "OOM_count": oom_count,
        "class_output_errors": gen_errors,
        "confidence_errors": incorrect_conf,
        "under_confident": under_conf,
        "over_confident": over_conf,
        "hist_bins": bins,
        "hist_counts": counts,
    }


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    if config["res_dir"] is not None:
        res_dir_path = Path(config["res_dir"])
        csv_files = sorted(list(res_dir_path.rglob("*.csv")))
    else:
        csv_files = config["csv_file"]

    rows_for_df = []
    for csv_file in csv_files:
        model_name, dataset_name, method_name = str(csv_file.stem).split("_")
        row = {
            "key": csv_file.stem,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "method_name": method_name,
        }
        results = eval_csv(csv_file)
        row.update(results)
        rows_for_df.append(row)

        print(f"Model name: {model_name}")
        print(f"  Accuracy: {results['accuracy']}")
        print(f"  F1-macro: {results['f1_macro']}")
        print(f"  ECE: {results['ece']}")
        print(f"  Histogram bins: {results['hist_bins']}")
        print(f"  Histogram counts: {results['hist_counts']}")
        print()

    print(f"Saving dataframe to {config['save_path']}...")
    df = pd.DataFrame(rows_for_df)
    # only keep ones without standard (vLLM)
    df = df[~df["key"].str.contains("standard", na=False)]
    df.to_csv(config["save_path"], index=False)
    print("Dataframe saved.")
