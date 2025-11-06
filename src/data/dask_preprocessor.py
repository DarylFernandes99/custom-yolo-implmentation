import os
import ijson
import dask.bag as db
import pandas as pd
import dask.dataframe as ddf
from dask.diagnostics import ProgressBar

class DaskPreprocessor:
    """
    Converts Large JSON annotation files into Dask Dataframes and export them as Parquet files
    """

    def __init__(self, file_path, output_dir="data/processed/parquet/", meta_dir="data/processed/metadata/"):
        self.file_path = file_path
        self.output_dir = output_dir
        self.meta_dir = meta_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        ProgressBar().register()

    def stream_array(self, key, chunk_size=10_000):
        with open(self.file_path, "rb") as f:
            items = ijson.items(f, f"{key}.item")
            batch = []
            for obj in items:
                batch.append(obj)
                if len(batch) >= chunk_size:
                    yield pd.Dataframe(batch)
                    batch.clear()
            if batch:
                yield pd.DataFrame(batch)

    def load_images_metadata(self):
        print("Loading image metadata...")
        df_parts = list(self.stream_array("images", chunk_size=10_000))
        image_ddf = ddf.from_pandas(pd.concat(df_parts, ignore_index=True), npartitions=len(df_parts))    
        image_ddf = image_ddf.drop(columns=["license", "coco_url", "date_captured"], errors="ignore")
        image_ddf = image_ddf.rename(columns={"id": "image_id"})[["image_id", "file_name", "height", "width", "flickr_url"]]
        print(f"Loaded {len(image_ddf)} image metadata records after cleaning.")
        return image_ddf

if __name__ == "__main__":
    preprocessor = DaskPreprocessor(
        file_path = '../../dataset/annotation/instances_val2017.json',
        output_dir="./processed/parquet/",
        meta_dir="./processed/metadata/"
    )

df = preprocessor.load_images_metadata()