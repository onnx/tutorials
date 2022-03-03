from re import findall
from urllib.request import Request, urlopen


SKIP_URLS_LIST = ["https://127.0.0",
                  # Used for server demo code
                  "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet20_CIFAR10_CNTK.model"
                  # [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: Hostname mismatch, certificate is not valid
                  ]


def validate_url(url):
    for skip_url in SKIP_URLS_LIST:
        if skip_url in url:
            return True
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
        request = Request(url, headers=headers)
        with urlopen(request) as response:
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
    """ Trim , \n . ) 's in the end """
    url = url.replace("\\n", "")
    url = url.replace("'s", "")
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
