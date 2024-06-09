import unittest
import os
import logging

from itertools import chain

from torch.utils.data import DataLoader
from torchvision.transforms.v2 import PILToTensor, Resize, Compose

from seyeon.pytorch_extension.vision.datasets.mvtec import MVTecAD

logging.basicConfig(level=logging.INFO)


class MVTecADTestCase(unittest.TestCase):
    def get_all_datasets(self, transform=None):
        return chain(self.train_datasets(transform=transform),
                     self.test_datasets(transform=transform))

    def train_datasets(self, transform=None):
        for category in MVTecAD.AVAILABLE_CATEGORIES:
            yield MVTecAD(category, train=True, transform=transform)

    def test_datasets(self, transform=None):
        for category in MVTecAD.AVAILABLE_CATEGORIES:
            yield MVTecAD(category, train=False, transform=transform)

    def test_is_good_index_zero(self):
        for dataset in self.get_all_datasets():
            self.assertEqual(dataset.classes[0], "good")
            self.assertEqual(dataset.class_to_index["good"], 0)

    def test_class_indices_validity(self):
        for dataset in self.get_all_datasets():
            num_classes = len(dataset.classes)
            self.assertListEqual(list(range(num_classes)), sorted(
                list(dataset.class_to_index.values())))
            self.assertTrue(list(range(num_classes)), sorted(
                list(set([idx for _, idx in dataset.imgs]))))

    def test_train_dataset_has_one_class(self):
        for dataset in self.train_datasets():
            self.assertTrue(len(dataset.classes), 1)
            self.assertTrue(len(dataset.class_to_index), 1)
            self.assertEqual(dataset.classes[0], "good")
            self.assertEqual(dataset.class_to_index["good"], 0)

    def test_test_dataset_has_mask(self):
        for dataset in self.test_datasets():
            self.assertTrue(dataset.masks)

    def test_train_dataset_has_no_mask(self):
        for dataset in self.train_datasets():
            self.assertIsNone(dataset.masks)

    def test_consistent_class_index(self):
        for dataset in self.get_all_datasets():
            for class_name in dataset.classes:
                self.assertTrue(
                    dataset.classes[dataset.class_to_index[class_name]], class_name)

    def test_img_mask_matched(self):
        for dataset in self.test_datasets():
            for (img_path, _), mask_path in zip(dataset.imgs, dataset.masks):
                img_filename = os.path.basename(img_path)
                mask_filename = os.path.basename(mask_path)
                img_filename, img_ext = os.path.splitext(img_filename)
                mask_filename, mask_ext = os.path.splitext(mask_filename)

                self.assertEqual(img_ext, mask_ext)
                self.assertEqual(img_filename, mask_filename[:-5])

                img_classname = os.path.basename(os.path.relpath(
                    os.path.dirname(img_path), dataset.category_dir / "test"))
                mask_classname = os.path.basename(os.path.relpath(
                    os.path.dirname(mask_path), dataset.category_dir / "ground_truth"))

                self.assertTrue(img_classname, mask_classname)

    def test_img_path_and_label_matched(self):
        for dataset in self.get_all_datasets():
            for img_path, label in dataset.imgs:
                class_name = dataset.classes[label]
                if dataset.train:
                    dataset_dir = dataset.category_dir / "train" / class_name
                else:
                    dataset_dir = dataset.category_dir / "test" / class_name
                self.assertTrue(img_path.startswith(
                    dataset_dir.absolute().as_posix()))

    def test_batchable(self):
        tfms = Compose([PILToTensor(), Resize((100, 100))])
        for dataset in self.get_all_datasets(transform=tfms):
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
            for batch in dataloader:
                img_paths = batch["image_path"]
                imgs = batch["image"]
                masks = batch["mask"]

                self.assertEqual(
                    (3, 100, 100), imgs.shape[-3:], f"Unexpected image shape with images {img_paths}")
                self.assertEqual(
                    (1, 100, 100), masks.shape[-3:], f"Unexpected mask shape with images {img_paths}")
                self.assertEqual(imgs.shape[0], masks.shape[0])
