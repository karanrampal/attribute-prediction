#!/usr/bin/env python
"""Script for dataset creation"""

import argparse
import json
import logging
import os
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils.utils import set_logger


def arg_parser() -> Tuple[argparse.Namespace, List[str]]:
    """Parse CLI argments"""
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument(
        "-c",
        "--castors",
        default="gs://hm_images/annotations/castors.csv",
        type=str,
        help="Castors file",
    )
    parser.add_argument(
        "-p",
        "--pim",
        default="gs://hdl_tables/dim/dim_pim.parquet",
        type=str,
        help="Directory for Pim table",
    )
    parser.add_argument(
        "-d",
        "--padma",
        default="gs://hdl_tables/dma/product_article_datamart.parquet",
        type=str,
        help="Directory for Padma table",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="gs://hm_images/annotations/",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "-s",
        "--cols",
        action="extend",
        default=[
            "product_code",
            "article_code",
            "product_age_group",
            "product_waist_rise",
            "product_sleeve_length",
            "product_garment_length",
            "product_fit",
            "product_sleeve_style",
            "product_neck_line_style",
            "product_collar_style",
        ],
        nargs="+",
        type=str,
        help="Columns to use for labels",
    )

    return parser.parse_known_args()


def main() -> None:
    """Main function"""
    known_args, _ = arg_parser()
    set_logger()

    # Read input data
    logging.info("Read input tables")
    padma = pd.read_parquet(known_args.padma, columns=["product_code", "article_code", "castor"])
    pim = pd.read_parquet(known_args.pim, columns=known_args.cols)
    castors = pd.read_csv(known_args.castors)

    # Clean data tables
    logging.info("Clean data")
    padma = padma.drop_duplicates()
    padma.castor = padma.castor.astype(int)
    assert not padma.isna().any().any()

    pim = pim.dropna(axis=0, subset=["article_code"])
    pim = pim.drop_duplicates()

    # Process PIM data
    logging.info("Process pim")
    out = []
    for col in known_args.cols:
        out.append(pim[col].apply(lambda x: json.loads(x) if x and "[" in x else x))
    tmp = pd.concat(out, axis=1)
    out = []
    for col in known_args.cols[2:]:
        out.append(pd.get_dummies(tmp[col].explode()).reset_index().groupby("index").max())
    res = pd.concat(out, axis=1)
    res = pd.concat([pim[known_args.cols[:2]], res], axis=1)
    assert not res.isna().any().any()

    # Merge pim and padma table
    logging.info("Create labels")
    data = res.merge(padma, on=["product_code", "article_code"], how="left")
    data.dropna(inplace=True)
    data = data.drop(axis=1, labels=["product_code", "article_code"])
    assert not data.isna().any().any()

    # Merge castor data to get output
    out = castors.merge(data, on="castor", how="inner")
    assert not out.isna().any().any()

    # Split data into training and test dataset
    logging.info("Split data")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
    train_idxs, test_idxs = next(gss.split(X=out.path, groups=out.castor))
    assert not set(out.castor[train_idxs]) & set(out.castor[test_idxs])
    out.drop(columns=["castor"], axis=1, inplace=True)
    train = out.iloc[train_idxs, :]
    test = out.iloc[test_idxs, :]

    # Write output files
    logging.info("Write to file")
    train.to_csv(os.path.join(known_args.out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(known_args.out_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()
