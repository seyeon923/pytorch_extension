import hashlib
import pathlib
import logging

from typing import BinaryIO

import requests


def download_from_url(url: str, dst: str | pathlib.Path | BinaryIO,
                      expected_hash: str = "", hash_algorithm: str = "sha256"):
    logging.info(f"Downloading file from '{url}' ...")
    response = requests.get(url)

    if response.status_code == 200:
        hash_pass = True
        if expected_hash:
            h = hashlib.new(hash_algorithm, response.content)
            actual_hash = h.hexdigest()
            hash_pass = expected_hash == actual_hash

        if hash_pass:
            if isinstance(dst, (str, pathlib.Path)):
                with open(dst, "wb") as f:
                    f.write(response.content)
            else:
                dst.write(response.content)
            logging.info(f"The file from '{
                         url}' has been successfully downloaded")
        else:
            raise RuntimeError(f"The {hash_algorithm} hash of the downloaded file '{
                               actual_hash}' is different from the value '{expected_hash}' that you expected!")
    else:
        raise RuntimeError(f"Failed to download the file from the url '{url}'")


if __name__ == "__main__":
    from tempfile import TemporaryFile
    url = "https://upload.wikimedia.org/wikipedia/ko/2/24/Lenna.png"
    hash_algorithm = "md5"
    expected_hash = "814a0034f5549e957ee61360d87457e5"

    with TemporaryFile() as temp_file:
        download_from_url(url, temp_file,
                          expected_hash=expected_hash,
                          hash_algorithm=hash_algorithm)

    download_from_url(url, "lena1.png",
                      expected_hash=expected_hash,
                      hash_algorithm=hash_algorithm)

    download_from_url(url, pathlib.Path("./lena2.png"),
                      expected_hash=expected_hash,
                      hash_algorithm=hash_algorithm)

    download_from_url(url, pathlib.Path("./lena3.png"))
