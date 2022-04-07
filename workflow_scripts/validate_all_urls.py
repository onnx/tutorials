# SPDX-License-Identifier: Apache-2.0

import os
from sys import exit
from url_validator import validate_file


def validate_urls_under_directory(directory):
    total_count = 0
    invalid_url_count = 0

    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            total_count += 1
            print(f"-----------validate {file_path}")
            if not validate_file(file_path):
                invalid_url_count += 1

    if invalid_url_count == 0:
        print(f"{total_count} files passed. ")
    else:
        print(f"{invalid_url_count} files failed in {total_count} files. ")
        exit(1)


if __name__ == '__main__':
    validate_urls_under_directory('.')
