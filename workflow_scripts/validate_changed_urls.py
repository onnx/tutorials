from os.path import exists
from url_validator import validate_file
import subprocess


def get_changed_files():
    try:
        files = subprocess.check_output(["git", "diff", "--name-only", "origin/main", "--"])
    except subprocess.CalledProcessError as ex:
        return ex.output
    return files.decode("utf-8").split("\n")


def validate_changed_urls():
    files = get_changed_files()
    total_count = 0
    invalid_url_count = 0

    for file_path in files:
        total_count += 1
        if not exists(file_path):
            print(f"Skip because {file_path} does not exist. ")
            break
        print(f"-----------validate {file_path}")
        if not validate_file(file_path):
            invalid_url_count += 1

    if invalid_url_count == 0:
        print(f"{total_count} updated files passed. ")
    else:
        print(f"{invalid_url_count} files failed in updated {total_count} files. ")
        exit(1)


if __name__ == '__main__':
    validate_changed_urls()
