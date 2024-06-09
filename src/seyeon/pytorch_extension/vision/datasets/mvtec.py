import os
import tarfile
import hashlib

from pathlib import Path
from typing import Callable, Any, Optional

import torch

from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import to_grayscale

from ...utils import download_from_url


class MVTecAD(Dataset):
    URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
    ARCHIVE_FILENAME = "mvtec_anomaly_detection.tar.xz"
    HASH = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
    HASH_ALGORITHM = "sha256"

    EXTRACT_FINISH_FLAG_FILENAME = ".extract_flag_file.txt"
    EXTRACT_FINISH_FLAG_FILE_CONTENT = \
        "This file is for marking that MVTect dataset already downloaded and extracted."

    DEFAULT_DATA_DIR = Path.home() / ".datasets" / "mvtec"

    AVAILABLE_CATEGORIES = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    def __init__(self, category: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 dataset_dir: str | Path = DEFAULT_DATA_DIR):
        """Create MVTecAD Dataset

        Args:
        - `category` : Specific category of MVTecAD dataset to be created as `Dataset`.
        - `train`: If `True`, creates dataset for training which consists of only anomaly-free images.
        - `transform` : A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        - `target_transform` : A function/transform that takes in the
            target and transforms it.
        - `loader` : A function to load an image given its path.
        - `dataset_dir` : Directory where MVTecAD dataset to be downloaded.
        """
        self.__category = category
        self.__dataset_dir = Path(dataset_dir).absolute()
        self.__category_dir = (self.dataset_dir / category).absolute()
        self.__archive_path = (
            self.dataset_dir / self.ARCHIVE_FILENAME).absolute()

        self.__loader = loader
        self.__transform = transform
        self.__target_transform = target_transform

        if not self.__is_already_extracted():
            self.__download_mvtect_dataset_archive()
            self.__extract_archive()

        if not self.category_dir.exists():
            raise RuntimeError(
                f"No directory found for category {self.category_dir}, you might pass the invalid category.")

        self.__train = train
        if train:
            image_folder = ImageFolder(self.category_dir / "train")
            self.__masks = None
        else:
            image_folder = ImageFolder(self.category_dir / "test")
            self.__masks = [self.__get_mask_path_from_img_path(
                path, image_folder.classes[label]).absolute().as_posix()
                for path, label in image_folder.imgs]

        self.__classes = image_folder.classes
        self.__class_to_index = image_folder.class_to_idx
        self.__imgs = [(Path(path).absolute().as_posix(), idx)
                       for path, idx in image_folder.imgs]

        self.__good_idx_to_zero()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img = tv_tensors.Image(self.loader(img_path))
        if label > 0:
            mask = self.loader(self.masks[idx])
            mask = to_grayscale(mask, num_output_channels=1)
            mask = tv_tensors.Mask(mask)
        else:
            img_h, img_w = img.shape[-2:]
            mask = tv_tensors.Mask(torch.zeros((1, img_h, img_w)))

        if self.transform:
            img, mask = self.transform(img, mask)
        if self.target_transform:
            label = self.target_transform(label)

        return {"image_path": img_path, "image": img, "mask": mask}

    @property
    def classes(self):
        return self.__classes

    @property
    def class_to_index(self):
        return self.__class_to_index

    @property
    def imgs(self):
        return self.__imgs

    @property
    def masks(self):
        return self.__masks

    @property
    def loader(self):
        return self.__loader

    @property
    def transform(self):
        return self.__transform

    @property
    def target_transform(self):
        return self.__target_transform

    @property
    def category(self):
        return self.__category

    @property
    def dataset_dir(self):
        return self.__dataset_dir

    @property
    def category_dir(self):
        return self.__category_dir

    @property
    def archive_path(self):
        return self.__archive_path

    @property
    def train(self):
        """`True` if it's train set."""
        return self.__train

    def __download_mvtect_dataset_archive(self):
        if self.__is_archive_exists():
            return

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        print(f"Downloading MVTecAD dataset to '{self.archive_path}' ...")
        download_from_url(self.URL, self.archive_path,
                          expected_hash=self.HASH,
                          hash_algorithm=self.HASH_ALGORITHM)

    def __extract_archive(self):
        print(f"Extracting MVTecAD dataset to '{self.dataset_dir}' ...")
        with tarfile.open(self.archive_path) as f:
            f.extractall(self.dataset_dir)

        self.__mark_as_mvtec_extracted()

    def __is_already_extracted(self):
        flag_file_path = Path(self.dataset_dir) / \
            self.EXTRACT_FINISH_FLAG_FILENAME

        return flag_file_path.exists()

    def __is_archive_exists(self):
        if self.archive_path.exists():
            h = hashlib.new(self.HASH_ALGORITHM)
            BUF_SIZE = 1024 * 1024
            with open(self.archive_path, "rb") as f:
                data = f.read(BUF_SIZE)
                while data:
                    h.update(data)
                    data = f.read(BUF_SIZE)

            return h.hexdigest() == self.HASH
        else:
            return False

    def __get_mask_path_from_img_path(self, img_path: str | Path, class_name: str):
        img_filename = os.path.basename(img_path)
        img_filename_wo_ext = os.path.splitext(img_filename)[0]
        return \
            self.category_dir / "ground_truth" / \
            class_name / f"{img_filename_wo_ext}_mask.png"

    def __mark_as_mvtec_extracted(self):
        flag_file_path = self.dataset_dir / self.EXTRACT_FINISH_FLAG_FILENAME

        with open(flag_file_path, "w") as f:
            f.write(self.EXTRACT_FINISH_FLAG_FILE_CONTENT)

    def __good_idx_to_zero(self):
        orig_good_idx = self.class_to_index["good"]
        orig_zero_idx_class = self.classes[0]

        self.classes[0], self.classes[orig_good_idx] = self.classes[orig_good_idx], self.classes[0]
        self.class_to_index["good"] = 0
        self.class_to_index[orig_zero_idx_class] = orig_good_idx

        new_imgs = []
        for img_path, idx in self.imgs:
            if idx == 0:
                idx = orig_good_idx
            elif idx == orig_good_idx:
                idx = 0
            new_imgs.append((img_path, idx))
        self.__imgs = new_imgs


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision.transforms.v2 import Compose, RandomAffine, Resize, PILToTensor
    import torchvision.transforms.v2.functional as F
    from torchvision.utils import make_grid

    tfms = Compose([PILToTensor(), RandomAffine(
        degrees=10, translate=(0.2, 0.2)), Resize(100)])

    dataset = MVTecAD("bottle", train=False, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        imgs = batch["image"]
        masks = batch["mask"]

        imgs = make_grid(imgs)
        imgs = F.to_pil_image(imgs)
        masks = make_grid(masks)
        masks = F.to_pil_image(masks)

        plt.imshow(imgs)
        plt.imshow(masks, alpha=0.25)
        plt.show()
        plt.close()
