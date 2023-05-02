#!/usr/bin/env python3
"""Script for dataset creation"""

import argparse
import logging
import os
from typing import List, Tuple

import pandas as pd

from utils.utils import set_logger


def arg_parser() -> Tuple[argparse.Namespace, List[str]]:
    """Parse CLI argments"""
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument(
        "-c",
        "--castors",
        default="gs://hm-images-bucket/annotations/castors.csv",
        type=str,
        help="Castors file",
    )
    parser.add_argument(
        "-p",
        "--pim",
        default="gs://hdl-tables/dim/dim_pim/",
        type=str,
        help="Directory for Pim table",
    )
    parser.add_argument(
        "-d",
        "--padma",
        default="gs://hdl-tables/dma/product_article_datamart/",
        type=str,
        help="Directory for Padma table",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="gs://hm-images-bucket/annotations/",
        type=str,
        help="Output directory",
    )

    return parser.parse_known_args()


def main() -> None:
    """Main function"""
    known_args, _ = arg_parser()
    set_logger()

    # Read input data
    logging.info("Read input tables")
    padma = pd.read_parquet(known_args.padma, columns=["product_code", "article_code", "castor"])
    pim = pd.read_parquet(known_args.pim, columns=["product_code", "article_code", "product_fit"])
    castors = pd.read_csv(known_args.castors)

    # Clean data tables
    logging.info("Clean data")
    padma = padma.drop_duplicates()
    padma.castor = padma.castor.astype(int)
    assert not padma.isna().any().any()

    pim = pim.dropna(axis=0, subset=["article_code", "product_fit"])
    pim = pim.drop_duplicates()
    assert not pim.isna().any().any()

    # Merge pim and padma table
    data = pim.merge(padma, on=["product_code", "article_code"], how="left")
    data = data.drop(axis=1, labels=["product_code", "article_code"])
    data = data[~data["product_fit"].str.contains("[", regex=False)]
    assert not data.isna().any().any()

    # Merge castor data to get output
    logging.info("Create labels")
    out = castors.merge(data, on="castor", how="inner")
    out["labels"] = out["product_fit"].astype("category").cat.codes
    assert not out.isna().any().any()

    # Split data into training and test dataset
    logging.info("Split data")
    tmp = out[["product_fit", "castor"]].drop_duplicates()
    sub_train = tmp.groupby("product_fit").sample(frac=0.8)
    sub_train["is_train"] = True
    final = out.merge(sub_train[["castor", "is_train"]], on="castor", how="left")
    final.fillna(False, inplace=True)
    assert not final.isna().any().any()
    train_fit = final.loc[final.is_train, ["path", "castor", "product_fit", "labels"]]
    test_fit = final.loc[~final.is_train, ["path", "castor", "product_fit", "labels"]]
    assert not set(train_fit.castor) & set(test_fit.castor)

    # Create automl data
    logging.info("Create gcp ai specific data")
    out_gcp = out[["path", "product_fit"]].copy()
    out_gcp["path"] = "gs://hm-images-bucket/images/" + out_gcp["path"]
    out_gcp["mode"] = "VALIDATION"
    out_gcp.loc[final.is_train, "mode"] = "TRAINING"
    out_gcp = out_gcp[["mode", "path", "product_fit"]]

    # Write output files
    logging.info("Write to file")
    out.to_csv(os.path.join(known_args.out_dir, "full_fit.csv"), index=False)
    train_fit.to_csv(os.path.join(known_args.out_dir, "train.csv"), index=False)
    test_fit.to_csv(os.path.join(known_args.out_dir, "test.csv"), index=False)
    out_gcp.to_csv(
        os.path.join(known_args.out_dir, "full_fit_gcai.csv"),
        index=False,
        header=False,
    )


if __name__ == "__main__":
    main()
