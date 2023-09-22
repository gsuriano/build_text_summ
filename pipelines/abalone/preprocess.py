"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

feature_columns_names = ["sentence","abstract"]

label_column = "r2f"
# Since we get a headerless CSV file we specify the column names here.

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/input").mkdir(parents=True, exist_ok=True)
    

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/input/dataset.csv"
    
    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=None,
        names=feature_columns_names + [label_column],
    )
    os.unlink(fn)

    train, split = train_test_split(df, test_size=0.30, random_state=42)
    validation, test = train_test_split(split, test_size=0.5, random_state=42)
    
    logger.info("Writing out datasets to %s.", base_dir)
    train.to_csv(f"{base_dir}/train/train.csv", index=False)
    #s3.Bucket(bucket).download_file("data/train/train.csv", f"{base_dir}/train/train.csv")
    validation.to_csv(
        f"{base_dir}/validation/validation.csv", index=False
    )
    #s3.Bucket(bucket).download_file("data/validation/validation.csv", f"{base_dir}/validation/validation.csv")
    test.to_csv(f"{base_dir}/test/test.csv", index=False)
    #s3.Bucket(bucket).download_file("data/test/test.csv", f"{base_dir}/test/test.csv")
