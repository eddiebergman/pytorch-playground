from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import requests
from torch import Tensor

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class MNIST:
    """MNIST dataset."""

    URL: ClassVar = "https://github.com/pytorch/tutorials/raw/main/_static/mnist.pkl.gz"
    x_train: Tensor
    x_valid: Tensor
    y_train: Tensor
    y_valid: Tensor

    @classmethod
    def get(
        cls,
        path: Path = DATA_PATH / "mnist.pkl.gz",
        *,
        download: bool = True,
    ) -> MNIST:
        """Get MNIST dataset.

        Args:
            path: Path to dataset.
            download: Download dataset if not found.

        Returns:
            MNIST dataset.
        """
        if not path.exists() and download:
            content = requests.get(cls.URL, timeout=5).content
            with path.open("wb") as f:
                f.write(content)

        if not path.exists():
            raise FileNotFoundError(f"File {path} not found.")

        with gzip.open(path, "rb") as f:
            train, valid, test = pickle.load(f, encoding="latin-1")  # noqa: S301

        return cls()
