"""
Usage:
    python duplicate_configs.py <config_dir> <old_dataset> <new_dataset>

Example:
    python duplicate_configs.py configs/ fever banking77
"""

import sys
from pathlib import Path


def duplicate_configs(config_dir: str, old_dataset: str, new_dataset: str):
    config_path = Path(config_dir)

    if not config_path.is_dir():
        print(f"Error: {config_dir} is not a directory")
        return

    files_created = 0
    for yaml_file in config_path.rglob("*.yaml"):
        if old_dataset not in yaml_file.name:
            continue

        new_name = yaml_file.name.replace(old_dataset, new_dataset)
        new_file = yaml_file.parent / new_name

        if new_file.exists():
            print(f"Skipping {new_file} (already exists)")
            continue

        content = yaml_file.read_text()
        new_content = content.replace(old_dataset, new_dataset)
        new_file.write_text(new_content)

        print(f"Created {new_file}")
        files_created += 1

    print(f"\nDone. {files_created} files created.")


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return

    duplicate_configs(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
