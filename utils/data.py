import typing

from torchvision import transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.unlabeled.iwildcam_unlabeled_dataset import IWildCamUnlabeledDataset

def get_iwildcam_datasets() -> tuple[IWildCamDataset, IWildCamUnlabeledDataset]:
    """Retrieves the iWildCam labeled and unlabeled datasets."""
    dataset = typing.cast(  # explicit cast to make type checker happy
        IWildCamDataset | None,
        get_dataset(dataset="iwildcam", download=True),
    )
    dataset_unlabeled = typing.cast(
        IWildCamUnlabeledDataset | None,
        get_dataset(dataset="iwildcam", download=True, unlabeled=True),
    )
    # error out if dataset not found
    if not dataset or not dataset_unlabeled:
        raise RuntimeError("Dataset not found, or failed to download.")
    return dataset, dataset_unlabeled


def create_loader(
    dataset: IWildCamDataset | IWildCamUnlabeledDataset,
    subset_type: str,
    tfms: transforms.Compose,
    batch_size: int,
    num_workers=4,
):
    """Creates a train dataloader."""
    if subset_type not in dataset.split_dict:
        raise ValueError(f"subset_type must be one of {dataset.split_dict.keys()}")

    data = dataset.get_subset(split=subset_type, transform=tfms)
    loader = get_train_loader if subset_type == "train" else get_eval_loader

    return loader(
        loader="standard",
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
