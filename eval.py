import argparse
import os
import torch
import re
import lightning as L
from utils.mappings import MODEL_MAP, TFMS_MAP
from utils.data import get_iwildcam_datasets, create_loader
import pandas as pd
from models.snapshot_ensemble import SnapshotEnsemble

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
    match = re.match(pattern, checkpoint_path.split('/')[-1])

    if not match:
        raise ValueError("Checkpoint filename does not match the expected format.")

    return {
        "model_name": match.group("model_name"),
        "model_variant": match.group("model_variant"),
        "lr": float(match.group("lr")),
        "batch_size": int(match.group("batch_size")),
        "ensemble": match.group("ensemble"),
        "n_estimators": int(match.group("n_estimators") or -1)
    }

def collect_eval_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the saved model checkpoint file")
    return parser.parse_args()

def main():
    args = collect_eval_arguments()

    # Parse checkpoint file name to get model details
    checkpoint_info = parse_checkpoint_filename(args.checkpoint_path)
    model_name = checkpoint_info["model_name"]
    model_name_base = model_name.split("-")[0] # strip model subtype
    model_variant = checkpoint_info["model_variant"]
    batch_size = checkpoint_info["batch_size"]

    # Select the model from MODEL_MAP based on the name and variant
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model name must be one of {MODEL_MAP.keys()}")
    model_class = MODEL_MAP.get(model_name, MODEL_MAP[model_name_base])

    model_tfms = TFMS_MAP[model_name_base]

    # init trainer class just to use predict method
    dummy_trainer = L.Trainer(devices=1, logger=[], deterministic=True)  # suppress logging

    # Load the labeled dataset and create evaluation DataLoaders
    labeled_dataset, _ = get_iwildcam_datasets()

    classifying_domains = checkpoint_info["model_name"].split("-")[-1] == "domain"
    if classifying_domains:
        # count the number of domains in the labeled dataset
        out_classes = len(torch.unique(labeled_dataset.metadata_array[:, 0]))
    else:
        out_classes = labeled_dataset.n_classes
    assert out_classes is not None


    # Initialize the model from the checkpoint
    if checkpoint_info["ensemble"]:
        if checkpoint_info["ensemble"] == "snapshot-pl":
            model = SnapshotEnsemble.load_from_checkpoint(
                args.checkpoint_path,
                out_classes=out_classes,
                train_loader_len=-1,
                num_estimators=checkpoint_info["n_estimators"],
                base_model=model_class,
                base_model_args={
                    "variant": model_variant,
                    "out_classes":out_classes
                }
            )
        else:
            raise ValueError("Use the jupyter notebook to evaulate torchensemble models") # TODO: add support for torchenesemble
    else:
        model = model_class.load_from_checkpoint(
            args.checkpoint_path,
            variant=model_variant,
            out_classes=out_classes,
        )
    model.eval()

    agg_res = {}
    for split in EVAL_SPLIT_TYPES:
        loader = create_loader(
            labeled_dataset,
            subset_type=split,
            tfms=model_tfms,
            batch_size=batch_size
        )
        assert loader is not None

        # infer on the loader
        all_y_pred = dummy_trainer.predict(model=model, dataloaders=loader)
        assert all_y_pred is not None
        # find predictions for each data example
        all_y_pred = torch.vstack(all_y_pred).argmax(dim=-1).flatten() # type: ignore

        # ok to skip transforms, X is not used
        subset_no_tfms = labeled_dataset.get_subset(split)
        all_y_true = (
            subset_no_tfms.y_array
            if not classifying_domains
            else subset_no_tfms.metadata_array[:, 0] # use domain labels
        )
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
    df.to_csv(f"results/{stripped_fn}.csv") # save to file


if __name__ == "__main__":
    main()
