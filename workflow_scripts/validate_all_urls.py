# SPDX-License-Identifier: Apache-2.0

import os
from sys import exit

import pathspec

from url_validator import validate_file


VALID_EXTENSIONS = {
    ".md",
    ".py",
    ".rst",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
}


def load_gitignore():
    """
    Load .gitignore patterns using pathspec.
    """
    if not os.path.exists(".gitignore"):
        return None

    with open(".gitignore", "r", encoding="utf-8") as gitignore:
        patterns = gitignore.readlines()

    return pathspec.PathSpec.from_lines(
        "gitwildmatch",
        patterns
    )


def should_validate(file_name):
    _, ext = os.path.splitext(file_name)
    return ext.lower() in VALID_EXTENSIONS


def validate_urls_under_directory(directory):
    total_count = 0
    passed_count = 0
    failed_count = 0

    gitignore_spec = load_gitignore()

    for root, _, files in os.walk(directory):

        for file_name in files:

            file_path = os.path.join(root, file_name)

            relative_path = os.path.relpath(file_path, ".")

            # Skip .gitignore ignored files
            if gitignore_spec and gitignore_spec.match_file(relative_path):
                continue

            # Skip unsupported file types
            if not should_validate(file_name):
                continue

            total_count += 1

            print(f"\n[Validating]: {relative_path}")

            try:
                if validate_file(file_path):
                    print("[PASSED]")
                    passed_count += 1
                else:
                    print("[FAILED]")
                    failed_count += 1

            except Exception as error:
                print(f"[ERROR]: {error}")
                failed_count += 1

    print("\n============= SUMMARY =============")
    print(f"Total checked : {total_count}")
    print(f"Passed        : {passed_count}")
    print(f"Failed        : {failed_count}")

    if failed_count > 0:
        exit(1)

    print("\n[SUCCESS] All URL validations passed!")


if __name__ == '__main__':
    validate_urls_under_directory(".")