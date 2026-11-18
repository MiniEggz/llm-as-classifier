import json
import os
import random
import re

import numpy as np
import torch
from transformers import set_seed


def set_all_seeds(seed=42):
    # basic + os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # hf transformers
    set_seed(seed)


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


def extract_from_gen(output: str):
    """Extract individual results for label and confidence from
    JSON output (hopefully) from LLM.

    Args:
        output (str): string output of the LLM - hopefully in JSON format.

    Returns:
        tuple[str, float]: label and confidence for the given output.
    """
    try:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            return output, -1

        data = json.loads(match.group())

        label = data.get("label", output)
        conf = data.get("confidence", -1)

        return label, conf

    except (json.JSONDecodeError, ValueError, TypeError):
        return output, -1


def get_clean_pred(pred_text: str, class_labels: list[str]) -> str:
    """Get cleaned predictions from raw prediction text.

    Args:
        pred_text (str): prediction directly from the classifier.
        class_labels (list[str]): list of valid class labels.

    Returns:
        str: cleaned label
    """
    # catch instances of nan being generated
    if isinstance(pred_text, float) or isinstance(pred_text, int):
        return "None"

    # if it is equal, directly take it
    # TODO: catch cases where the output isn't exact
    pred_text = pred_text.strip()
    if pred_text in class_labels:
        return pred_text

    lookup = {label.lower(): label for label in class_labels}
    regex_labels = "|".join(
        re.escape(label.lower())
        for label in sorted(
            class_labels, key=len, reverse=True
        )  # longest-first is load-bearing
    )
    pattern = re.compile(rf"(?<!\w)({regex_labels})(?!\w)")

    found = {lookup[m.group(1)] for m in pattern.finditer(pred_text.lower())}
    return found.pop() if len(found) == 1 else "None"


def get_clean_preds(pred_texts: list[str], class_labels: list[str]) -> list[str]:
    """Get cleaned predictions from raw prediction text.

    Args:
        pred_text (str): prediction directly from the classifier.
        class_labels (list[str]): list of valid class labels.

    Returns:
        list[str]: list of cleaned labels, same size as pred_text.
    """
    return [get_clean_pred(pred_text, class_labels) for pred_text in pred_texts]
