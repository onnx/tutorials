import os
from re import findall
from urllib.request import urlopen
from sys import exit


def validate_url(url):
    try:
        with urlopen(url) as response:
            status_code = response.getcode()
            # if the request succeeds
            if status_code == 200:
                return True
            else:
                print(f"{url}: is Not reachable, status_code: {status_code}.")
                return False

    except Exception as e:
        print(f"{url}: is Not reachable, Exception: {e}")
        return False


def polish_url(url):
    """ Trim ,\n.) in the end """
    url = url.replace("\\n", "")
    for i in range(len(url)):
        if url[len(url) - i - 1].isalpha() or url[len(url) - i - 1].isdigit():
            return url[:len(url) - i]
    return url


def validate_file(file_path):
    has_invalid_url = False
    if file_path.endswith(".ipynb") or file_path.endswith(".md"):
        with open(file_path, "r") as f:
            for line in f:
                url_list = findall(
                    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                    line)
                for url in url_list:
                    url = polish_url(url)
                    if not validate_url(url):
                        has_invalid_url = True

                if "http://" in line:
                    print(f"File {file_path} contains an insecure url: {line}")
                    has_invalid_url = True

    return not has_invalid_url


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
