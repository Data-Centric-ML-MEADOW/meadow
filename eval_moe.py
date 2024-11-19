import argparse
import os
import re

import lightning as L
import pandas as pd
import torch

from models.domain_moe import DomainMoE
from utils.data import create_loader, get_iwildcam_datasets
from utils.mappings import TFMS_MAP

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
    pattern = r"MoE(?P<n_estimators>\d+)_(?P<model_name>.+)-(?P<model_variant>.+)_(?P<domain_mapper_name>.+)-(?P<domain_mapper_variant>.+)_(?P<date>\d{8}-\d{6})_lr(?P<lr>[e0-9.-]+)_bs(?P<batch_size>\d+)(?:\[.+\])?\.ckpt"
    match = re.match(pattern, checkpoint_path.split("/")[-1])

    if not match:
        raise ValueError("Checkpoint filename does not match the expected format.")

    return {
        "model_name": match.group("model_name"),
        "model_variant": match.group("model_variant"),
        "domain_mapper_name": match.group("domain_mapper_name"),
        "domain_mapper_variant": match.group("domain_mapper_variant"),
        "lr": float(match.group("lr")),
        "batch_size": int(match.group("batch_size")),
        "n_estimators": int(match.group("n_estimators")),
    }


def collect_eval_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the saved model checkpoint file",
    )
    return parser.parse_args()


def main():
    args = collect_eval_arguments()

    # Parse checkpoint file name to get model details
    checkpoint_info = parse_checkpoint_filename(args.checkpoint_path)
    model_name = checkpoint_info["model_name"]
    model_name_base = model_name.split("-")[0]  # strip model subtype
    batch_size = checkpoint_info["batch_size"]

    model_tfms = TFMS_MAP[model_name_base]

    # init trainer class just to use predict method
    dummy_trainer = L.Trainer(
        devices=1, logger=[], deterministic=True
    )  # suppress logging

    # Load the labeled dataset and create evaluation DataLoaders
    labeled_dataset, _ = get_iwildcam_datasets()
    out_classes = labeled_dataset.n_classes
    num_domains = len(torch.unique(labeled_dataset.metadata_array[:, 0]))
    assert out_classes is not None

    # init moe model
    model = DomainMoE.load_from_checkpoint(
        args.checkpoint_path,
        out_classes=out_classes,
        num_domains=num_domains,
        num_estimators=checkpoint_info["n_estimators"],
    )
    model.eval()

    agg_res = {}
    for split in EVAL_SPLIT_TYPES:
        loader = create_loader(
            labeled_dataset, subset_type=split, tfms=model_tfms, batch_size=batch_size
        )
        assert loader is not None

        # infer on the loader
        all_y_pred = dummy_trainer.predict(model=model, dataloaders=loader)
        assert all_y_pred is not None
        # find predictions for each data example
        all_y_pred = torch.vstack(all_y_pred).argmax(dim=-1).flatten()  # type: ignore

        # ok to skip transforms, X is not used
        subset_no_tfms = labeled_dataset.get_subset(split)
        all_y_true = subset_no_tfms.y_array
        all_metadata = subset_no_tfms.metadata_array

        # Run evaluation with all accumulated predictions and metadata
        print(f"==={split}===")
        res, _ = labeled_dataset.eval(all_y_pred, all_y_true, all_metadata)
        print(res)
        agg_res[split] = res

    print("====SUMMARY====")
    df = pd.DataFrame.from_dict(agg_res)
    print(df)
    stripped_fn = args.checkpoint_path.rsplit(".", 1)[0]
    stripped_fn = stripped_fn.split("/")[-1]
    if not os.path.exists("results"):
        os.mkdir("results")
    df.to_csv(f"results/{stripped_fn}.csv")  # save to file

    all_y_preds = []
    print("====EVAL-PER-EXPERT====")
    for i, expert in enumerate(model.expert_models):
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
