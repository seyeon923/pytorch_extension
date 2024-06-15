import hashlib
import pathlib
import logging
import io

import tqdm

from typing import BinaryIO, Iterable

import requests


def __check_hash_and_write_from_bytes(content: bytes, file: str | pathlib.Path | BinaryIO,
                                      expected_hash: str = "", hash_algorithm: str = "sha256"):
    hash_pass = True
    if expected_hash:
        h = hashlib.new(hash_algorithm, content)
        actual_hash = h.hexdigest()
        hash_pass = expected_hash == actual_hash

    if hash_pass:
        if isinstance(file, (str, pathlib.Path)):
            with open(file, "wb") as f:
                f.write(content)
        else:
            file.write(content)
        logging.info(f"The file from '{url}' has been successfully downloaded")
    else:
        raise RuntimeError(f"The {hash_algorithm} hash of the downloaded file '{
                           actual_hash}' is different from the value '{expected_hash}' that you expected!")


def __check_hash_and_write_from_iterable(content_iterable: Iterable[bytes],
                                         file: str | pathlib.Path | BinaryIO,
                                         datasize: int,
                                         expected_hash: str = "", hash_algorithm: str = "sha256"):
    tmp_io = io.BytesIO()

    hash_pass = True
    if expected_hash:
        h = hashlib.new(hash_algorithm)

        with tqdm.tqdm(total=datasize) as pbar:
            for content in content_iterable:
                h.update(content)
                tmp_io.write(content)
                pbar.update(len(content))

        actual_hash = h.hexdigest()
        hash_pass = expected_hash == actual_hash
    else:
        with tqdm.tqdm(total=datasize) as pbar:
            for content in content_iterable:
                tmp_io.write(content)
                pbar.update(len(content))

    if hash_pass:
        if isinstance(file, (str, pathlib.Path)):
            with open(file, "wb") as f:
                f.write(tmp_io.getvalue())
        else:
            file.write(tmp_io.getvalue())
        logging.info(f"The file from '{url}' has been successfully downloaded")
    else:
        raise RuntimeError(f"The {hash_algorithm} hash of the downloaded file '{
                           actual_hash}' is different from the value '{expected_hash}' that you expected!")


def download_from_url(url: str, dst: str | pathlib.Path | BinaryIO,
                      expected_hash: str = "", hash_algorithm: str = "sha256"):
    logging.info(f"Downloading file from '{url}' ...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        datasize = response.headers.get("content-length")

        if datasize is None:
            __check_hash_and_write_from_bytes(
                response.content, dst, expected_hash=expected_hash, hash_algorithm=hash_algorithm)
        else:
            datasize = int(datasize)
            __check_hash_and_write_from_iterable(response.iter_content(
                chunk_size=4096), dst, datasize, expected_hash=expected_hash, hash_algorithm=hash_algorithm)
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
