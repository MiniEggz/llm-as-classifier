"""Map single-token prompt identifiers (e.g. A/B/C) in `pred` back to label names.

Usage:
    python map_prompt_options.py results.csv                 # writes results.csv in place (keeps a .bak)
    python map_prompt_options.py results.csv -o fixed.csv    # writes to a new file
"""

import argparse
import ast
import re
import shutil
from pathlib import Path

import pandas as pd


def map_pred(pred, options: list[str], prompt_options: list[str]):
    """Return the label name for `pred`, or `pred` unchanged if it can't be mapped."""
    if not isinstance(pred, str):
        return pred

    text = pred.strip()

    # already a label name -> leave it
    if text in options:
        return text

    ident_to_label = dict(zip(prompt_options, options))

    # exact identifier, e.g. "C" (also tolerates "C." / "c")
    key = text.rstrip(".").strip()
    for ident, label in ident_to_label.items():
        if key.upper() == ident.upper():
            return label

    # identifier inside longer text, e.g. "Answer: C" or "A (SUPPORTS)";
    # only accepted if exactly one identifier appears
    idents = sorted(prompt_options, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(i) for i in idents) + r")(?!\w)", re.IGNORECASE
    )
    found = {m.group(1).upper() for m in pattern.finditer(text)}
    if len(found) == 1:
        hit = found.pop()
        return next(l for i, l in ident_to_label.items() if i.upper() == hit)

    # unparseable: leave as-is so evaluate.py counts it as a generation error
    return pred


PROMPT_OPTION_LINE = re.compile(r"^\s*-\s*(\S+)\s+\((.+)\)\s*$", re.MULTILINE)


def prompt_options_from_prompt(prompt: str, options: list[str]) -> list[str]:
    """Rebuild identifiers from the '- A (SUPPORTS)' lines the model was shown,
    aligned with `options`. Raises if any option is missing from the prompt."""
    label_to_ident = {}
    for ident, label in PROMPT_OPTION_LINE.findall(prompt):
        label_to_ident.setdefault(label.strip(), ident)
    missing = [o for o in options if o not in label_to_ident]
    if missing:
        raise ValueError(f"Options not found in prompt: {missing[:5]}")
    return [label_to_ident[o] for o in options]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    options = df["options"].apply(ast.literal_eval)

    if "prompt_options" in df.columns:
        prompt_options = df["prompt_options"].apply(ast.literal_eval)
        source = "prompt_options column"
    else:
        prompt_col = "input_prompt" if "input_prompt" in df.columns else "messages"
        prompt_options = pd.Series(
            [
                prompt_options_from_prompt(str(p), o)
                for p, o in zip(df[prompt_col], options)
            ],
            index=df.index,
        )
        df["prompt_options"] = prompt_options.apply(str)
        source = f"rebuilt from {prompt_col}"

    first = dict(zip(prompt_options.iloc[0], options.iloc[0]))
    print(
        f"Mapping ({source}), first row: {dict(list(first.items())[:5])}"
        f"{' ...' if len(first) > 5 else ''}"
    )

    new_pred = [
        map_pred(p, o, po) for p, o, po in zip(df["pred"], options, prompt_options)
    ]
    mapped_ok = sum(isinstance(p, str) and p in o for p, o in zip(new_pred, options))
    changed = sum(str(a) != str(b) for a, b in zip(df["pred"], new_pred))

    df["pred"] = new_pred

    out = args.out or args.csv
    if out == args.csv:
        shutil.copy(args.csv, args.csv.with_suffix(".csv.bak"))
    df.to_csv(out, index=False)

    print(
        f"{args.csv.name}: {changed} preds changed, "
        f"{mapped_ok}/{len(df)} now valid labels, "
        f"{len(df) - mapped_ok} unparsed -> {out}"
    )


if __name__ == "__main__":
    main()
