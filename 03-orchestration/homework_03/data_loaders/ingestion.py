import requests
from io import BytesIO
from typing import List
import os
import sys
import pandas as pd
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    df = pd.read_parquet(f"/home/src/mlops/homework_03/data_loaders/data/yellow_tripdata_2023-03.parquet")
    return df
