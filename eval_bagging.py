import argparse
import os
import re

import lightning as L
import pandas as pd
import torch

from models.domain_moe import DomainMoE
from utils.data import create_loader, get_iwildcam_datasets
from utils.mappings import MODEL_MAP, TFMS_MAP

L.seed_everything(42, workers=True)  # for reproducability
torch.set_float32_matmul_precision("high")

EVAL_SPLIT_TYPES = [
    "val",
    "test",
    "id_val",
    "id_test",
]


def parse_checkpoint_filename(checkpoint_path: str):
    """Parse the checkpoint filename to extract model name, variant, and batch size."""
    pattern = r"(?P<model_name>.+)-(?P<model_variant>.+)_(?P<date>\d{8}-\d{6})_lr(?P<lr>[e0-9.-]+)_bs(?P<batch_size>\d+)(?:_(?P<ensemble>[a-z-]+)(?P<n_estimators>\d+))?(?:\[.+\])?\.ckpt"
    match = re.match(pattern, checkpoint_path.split("/")[-1])

    if not match:
        raise ValueError("Checkpoint filename does not match the expected format.")

    return {
        "model_name": match.group("model_name"),
        "model_variant": match.group("model_variant"),
        "lr": float(match.group("lr")),
        "batch_size": int(match.group("batch_size")),
        "ensemble": match.group("ensemble"),
        "n_estimators": int(match.group("n_estimators") or -1),
    }

def get_experts_from_folder(
    expert_checkpoints_folder: str, out_classes: int
) -> tuple[torch.nn.ModuleList, dict]:
    checkpoints = os.listdir(expert_checkpoints_folder)
    checkpoints_info = []
    expert_models = torch.nn.ModuleList()

    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(expert_checkpoints_folder, checkpoint)

        checkpoint_info = parse_checkpoint_filename(checkpoint)
        expert_model_name = checkpoint_info["model_name"]
        expert_model_name_base = expert_model_name.split("-")[0]
        expert_model_variant = checkpoint_info["model_variant"]

        expert_model_class = MODEL_MAP.get(
            expert_model_name, MODEL_MAP[expert_model_name_base]
        )

        expert_model = expert_model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            variant=expert_model_variant,
            out_classes=out_classes,
        )

        expert_models.append(expert_model)
        checkpoints_info.append(checkpoint_info)
        
    return (expert_models, checkpoints_info[0])

def collect_eval_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument(
        "--checkpoints-folder-path",
        required=True,
        help="Path to the folder containing the saved model checkpoint files",
    )
    return parser.parse_args()


def main():
    args = collect_eval_arguments()

    # Load the labeled dataset and create evaluation DataLoaders
    labeled_dataset, _ = get_iwildcam_datasets()
    out_classes = labeled_dataset.n_classes
    num_domains = len(torch.unique(labeled_dataset.metadata_array[:, 0]))
    assert out_classes is not None

    # Parse checkpoint file name to get model details
    expert_models, checkpoint_info = get_experts_from_folder(args.checkpoints_folder_path, out_classes)
    model_name = checkpoint_info["model_name"]
    model_name_base = model_name.split("-")[0]  # strip model subtype
    batch_size = checkpoint_info["batch_size"]

    model_tfms = TFMS_MAP[model_name_base]

    # init trainer class just to use predict method
    dummy_trainer = L.Trainer(
        devices=1, logger=[], deterministic=True
    )  # suppress logging

    

    all_y_preds = []
    print("====EVAL-PER-EXPERT====")
    for i, expert in enumerate(expert_models):
        loader = create_loader(
            labeled_dataset, subset_type='test', tfms=model_tfms, batch_size=batch_size
        )
        assert loader is not None

        # infer on the loader
        all_y_pred = dummy_trainer.predict(model=expert, dataloaders=loader) # type: ignore
        assert all_y_pred is not None
        # find predictions for each data example
        all_y_pred = torch.vstack(all_y_pred).argmax(dim=-1).flatten()  # type: ignore
        all_y_preds.append(all_y_pred)

        # ok to skip transforms, X is not used
        subset_no_tfms = labeled_dataset.get_subset('test')
        all_y_true = subset_no_tfms.y_array
        all_metadata = subset_no_tfms.metadata_array

        print(f"===Expert #{i}===")
        res, _ = labeled_dataset.eval(all_y_pred, all_y_true, all_metadata)
        print(res)
    all_y_preds = torch.stack(all_y_preds, dim=0)
    corr_matrix = torch.corrcoef(all_y_preds)
    print("====CORRELATION_MATRIX====")
    print(corr_matrix)


if __name__ == "__main__":
    main()
