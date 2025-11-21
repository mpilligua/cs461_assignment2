import torchvision.transforms as T
from torch.utils.data import Dataset
import pickle
from typing import Tuple
import torch

DEFAULT_TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2471, 0.2435, 0.2616),
        ),
    ]
)


class TestTimeAdaptationDataset(Dataset):
    """
    Dataset for test-time adaptation.
    Each data point is a tuple of (image, label, corruption_type).
    Note:
        - label can be None for unlabeled data (e.g., exploratory set).
        - corruption_type is a string indicating the type of corruption applied to the image.
        - The dataset can be filtered by corruption type. The use of _shared_data allows multiple
          filtered datasets to share the same underlying data loaded from disk, improving efficiency.

    """
    def __init__(
        self,
        dataset_path,
        kind="public_test_bench",
        transform="default",
        corruption=None,
        _shared_data=None,
    ):
        self.kind = kind
        self.dataset_path = dataset_path
        if transform == "default":
            self.transform = DEFAULT_TRANSFORM
        else:
            self.transform = transform
        self.corruption = corruption

        # Use shared data if provided, otherwise load fresh
        if _shared_data is not None:
            self.data = _shared_data
        else:
            self._load_data()

        # Filter if corruption specified
        if self.corruption is not None:
            self._filter_by_corruption()

    def _load_data(self):
        raw_data = pickle.load(open(f"{self.dataset_path}/{self.kind}_data.pkl", "rb"))
        if self.kind == "exploratory":
            self.data = [
                (self.transform(img), None, corruption) for img, corruption in raw_data
            ]
        else:
            self.data = [
                (self.transform(img), label, corruption)
                for img, label, corruption in raw_data
            ]

    def _filter_by_corruption(self):
        all_corruptions = set([corruption for _, _, corruption in self.data])
        assert self.corruption in all_corruptions, (
            f"Corruption {self.corruption} not found in dataset corruptions: {all_corruptions}"
        )
        self.data = [item for item in self.data if item[2] == self.corruption]

    def get_available_corruptions(self):
        """Get all unique corruptions in the dataset."""
        return sorted(set([corruption for _, _, corruption in self.data]))

    def filter_by_corruption(self, corruption):
        """
        Create a new dataset instance filtered by corruption, sharing the same underlying data.

        Args:
            corruption: Corruption type to filter by

        Returns:
            New TestTimeAdaptationDataset instance with filtered data
        """
        return TestTimeAdaptationDataset(
            dataset_path=self.dataset_path,
            kind=self.kind,
            transform=self.transform,
            corruption=corruption,
            _shared_data=self.data,  # Share the loaded data
        )

    def get_corruption_datasets(self):
        """
        Get a dictionary of datasets, one for each corruption type.

        Returns:
            Dict[str, TestTimeAdaptationDataset]: Mapping from corruption name to filtered dataset
        """
        corruptions = self.get_available_corruptions()
        return {corr: self.filter_by_corruption(corr) for corr in corruptions}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int | None, str]:
        return self.data[idx]
