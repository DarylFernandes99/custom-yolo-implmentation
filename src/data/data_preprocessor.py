import os
import cv2
import ast
import dask
import ijson
import pandas as pd
import pyarrow as pa
import dask.dataframe as ddf
from pycocotools import mask as maskUtils
from typing import List, Optional, Dict, Union, Any

class DataPreprocess:
    def __init__(self, annotations_dir: str, output_dir: str, is_test: bool = False):
        """
        Preprocessor for dataset annotation files used in image segmentation tasks.
        This class centralizes configuration for preprocessing annotation data (for example,
        bounding boxes, polygons) and for writing any converted or normalized
        annotation artifacts to an output location. It is intended to be used by higher-level
        methods that perform validation, parsing, format conversion, and saving of preprocessed
        annotation data.

        Parameters
        ----------
        annotations_dir : str
            Path to the directory containing the raw annotation files. Expected contents and
            formats depend on the dataset (e.g., JSON, XML, PNG masks); individual processing
            methods will document supported formats. This path is recorded as `input_dir` for
            internal use.
        output_dir : str
            Path to the directory where preprocessed annotation files and any derived outputs
            will be written. If the directory does not exist, calling code or specific methods
            may create it prior to writing.
        is_test : bool, optional
            Flag indicating whether the preprocessor is operating in test mode. When True,
            enables reduced processing scope. Defaults to False for standard preprocessing
            operations.
        """

        self.input_dir = annotations_dir
        self.output_dir = output_dir
        self.is_test = is_test

    def load_annotations_file(self, file_names: List[str], key: str, columns: Optional[List[str]] = None, chunk_size=10_000) -> ddf.DataFrame:
        """
        Load and process annotations from JSON files into a Dask DataFrame.
        This method reads JSON files containing nested data structures, streams them
        efficiently using ijson to handle large files, and chunks them into Pandas
        DataFrames before converting to a Dask DataFrame for distributed processing.
        Special handling is applied based on the data key type:
        - "images": Returns deduplicated data
        - "categories": Returns deduplicated, sorted data with reset index
        - Other keys: Returns concatenated data without modification

        Parameters:
        -------
        file_names : List[str]
            List of JSON file names to load from input_dir.
        key : str
            The root key in the JSON structure to extract items from.
            Used to stream specific data types (e.g., "images", "categories", "annotations").
        columns : Optional[List[str]], optional
            Column names for the resulting DataFrame.
            If None, column names are inferred from the data. Defaults to None.
        chunk_size : int, optional
            Number of items to accumulate before creating
            a DataFrame chunk. Helps control memory usage for large files.
            Defaults to 10,000.
        
        Returns:
        -------
        dask.dataframe.DataFrame
            A Dask DataFrame containing the loaded and processed data.
            Returns an empty DataFrame with specified columns if no data is found.
        """

        chunks = []

        for file in file_names:
            with open(os.path.join(self.input_dir, file), 'rb') as data:
                # Stream items using ijson
                objects = ijson.items(data, f'{key}.item', use_float=True)
                current_chunk = []
                
                for idx, obj in enumerate(objects):
                    current_chunk.append(obj)
                    if len(current_chunk) >= chunk_size:
                        chunks.append(pd.DataFrame(current_chunk, columns=columns))
                        current_chunk = []
                    
                    if self.is_test and (idx == 500):
                        print("[INFO] Early termination for test mode")
                        break
                
                if current_chunk:
                    chunks.append(pd.DataFrame(current_chunk, columns=columns))
        
        if not chunks:
            return ddf.from_pandas(pd.DataFrame(columns=columns), npartitions=1)
        
        # Convert list of Pandas DataFrames to a Dask DataFrame
        dask_chunks = [ddf.from_pandas(chunk, npartitions=1) for chunk in chunks]
        
        if key == "images":
            return ddf.concat(dask_chunks) \
                .drop_duplicates()
        elif key == "categories":
            return ddf.concat(dask_chunks) \
                .drop_duplicates() \
                .sort_values(by=["name"], ignore_index=True) \
                .reset_index()
        
        return ddf.concat(dask_chunks)
    
    def _safe_literal(self, x: Any) -> Any:
        """
        Safely evaluate a string literal into a Python object.
        Attempts to parse a string representation of a Python literal (such as 
        lists, dictionaries, tuples, numbers, etc.) into its corresponding Python 
        object. If the input is not a string or if evaluation fails, it returns 
        the input unchanged or None respectively.
        
        Parameters
        ----------
        x : str or any
            The value to be evaluated. If a string, it will be attempted to parse 
            as a Python literal. If not a string, it is returned as-is.
        
        Returns
        -------
        any or None
            - If x is not a string: returns x unchanged
            - If x is a valid string literal: returns the evaluated Python object
            - If x is a string but cannot be parsed as a literal: returns None
        """

        if not isinstance(x, str):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return None

    def _polygonFromMask_row(self, row):
        """
        Convert RLE-encoded masks to polygon representations for a single row.
        This method processes segmentation data in a pandas DataFrame row, handling both
        RLE (Run-Length Encoded) mask format and standard polygon formats. For RLE masks,
        it decodes the mask and extracts contours to convert them into polygon coordinates.
        
        Parameters
        ----------
        row : pandas.Series
            A row from a DataFrame containing segmentation data. Expected to have keys:
            - 'segmentation' : str or dict
                RLE encoded mask (as dict with 'counts' key) or polygon coordinates (as str)
            - 'bbox' : str or list
                Bounding box coordinates in string or list format
            - 'iscrowd' : int
                Flag indicating crowd annotation (1) or individual object (0)
        
        Returns
        -------
        pandas.Series
            The modified row with 'segmentation' converted to a list of polygon coordinates.
            For crowd annotations, returns the RLE counts. For individual objects with RLE masks,
            returns a list of polygons where each polygon is a flattened list of coordinates.
            Non-RLE segmentation data is passed through unchanged.
        """

        row = row.copy()

        # Convert strings to Python objects
        row['segmentation'] = self._safe_literal(row['segmentation'])
        row['bbox'] = self._safe_literal(row['bbox'])

        # Handle RLE mask case
        if isinstance(row['segmentation'], dict) and 'counts' in row['segmentation']:
            if row['iscrowd'] == 1:
                row['segmentation'] = [row['segmentation']['counts']]
            else:
                rle_data = maskUtils.decode(row['segmentation'])
                contours, _ = cv2.findContours(
                    rle_data,
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                polygons = []
                for contour in contours:
                    if contour.size >= 6:
                        polygons.append(contour.flatten().tolist())

                row['segmentation'] = polygons

        return row

    def polygonFromMask_partition(self, pdf):
        """
        Convert binary masks to polygons for a partition of data.
        This method applies polygon extraction to each row of a pandas DataFrame,
        converting binary mask representations into polygon coordinates.
        
        Parameters
        ----------
        pdf : pandas.DataFrame
            A pandas DataFrame containing binary mask data where each row
            represents a single image or mask instance to be converted.
        
        Returns
        -------
        pandas.Series
            A Series containing polygon coordinates for each row, where each
            element is the result of applying _polygonFromMask_row to the
            corresponding row of the input DataFrame.
        
        See Also
        --------
        _polygonFromMask_row : The method applied to each row for individual
                              mask-to-polygon conversion.
        """

        return pdf.apply(self._polygonFromMask_row, axis=1)
    
    @staticmethod
    def create_parquet_data(annotations_dir: str, output_dir: str, output_folder: str, file_names: List[str], keys: List[str], columns: List[List[str]], chunk_sizes: List[int], is_test: bool):
        """
        Creates a parquet file from image, annotations and categories.
        This function loads image data, annotations, and categories from specified files,
        merges them, applies transformations, and exports the combined data to a parquet file.
        
        Parameters:
        -------
        annotations_dir : str
            The directory containing the annotation files.
        output_dir : str
            The directory where the output parquet file will be saved.
        output_folder : str
            The name of the folder within the output directory to save the parquet file.
        file_names : List[str]
            A list of file names to load data from.
        keys : List[str]
            A list of keys corresponding to the data to be loaded.
        columns : List[List[str]]
            A list of lists, where each inner list contains the column names to be loaded for each key.
        chunk_sizes : List[int]
            A list of chunk sizes for loading data in manageable pieces.
        is_test : bool
            Flag indicating whether the data preprocessor is working on test mode or not.
        
        Returns:
        -------
        None: The function exports the combined data to a parquet file and does not return any value.
        """

        data_preprocess = DataPreprocess(
            annotations_dir=annotations_dir,
            output_dir=output_dir,
            is_test=is_test
        )
        print("[INFO] Initialized data preprocessor object")

        combined_images = data_preprocess.load_annotations_file(
                            file_names=file_names,
                            key=keys[0],
                            columns=columns[0],
                            chunk_size=chunk_sizes[0]
                        )
        print("[INFO] Loaded images data")
        combined_annots = data_preprocess.load_annotations_file(
                            file_names=file_names,
                            key=keys[1],
                            columns=columns[1],
                            chunk_size=chunk_sizes[1]
                        )
        print("[INFO] Loaded annotations data")
        combined_catego = data_preprocess.load_annotations_file(
                            file_names=file_names,
                            key=keys[2],
                            columns=columns[2],
                            chunk_size=chunk_sizes[2]
                        )
        print("[INFO] Loaded categories data")
        
        ddf_combined = ddf.merge(left=combined_images, right=combined_annots, how="inner", left_on="id", right_on="image_id", suffixes=("_image", "_annots")) \
                        .merge(combined_catego, how="inner", left_on="category_id", right_on="id", suffixes=("_combined", "categos")) \
                        .rename(columns={"id": "old_category_id", "category_id": "stale_category_id", "index": "category_id", "id_image": "id"})
        print("[INFO] Merged images, annotations and categories")

        ddf_combined = ddf_combined.map_partitions(data_preprocess.polygonFromMask_partition, meta=ddf_combined._meta)
        print("[INFO] Applied transformations to segemntation and bbox columns")
        
        ddf_combined = ddf_combined.drop(columns=["image_id", "stale_category_id", "id_annots"], errors="ignore")
        print("[INFO] Dropped redundant columns")

        # Aggregation to be performed on each column in group by
        agg_func = {
            'segmentation': list,
            'area': list,
            'iscrowd': list,
            'bbox': list,
            'category_id': list,
            'supercategory': list,
            'old_category_id': list,
            'name': list,
        }
        ddf_combined = ddf_combined.groupby(by=["file_name", "height", "width", "id"]).agg(agg_func, split_out=ddf_combined.npartitions).reset_index()
        print("[INFO] Grouped multiple rows for each image")

        # PyArrow schema to export data to parquet file
        pyarrow_schema = pa.schema([
            ("file_name", pa.string()),
            ("height", pa.int64()),
            ("width", pa.int64()),
            ("id", pa.int64()),
            ("segmentation", pa.list_(pa.list_(pa.list_(pa.float64())))),
            ("area", pa.list_(pa.float64())),
            ("iscrowd", pa.list_(pa.int64())),
            ("bbox", pa.list_(pa.list_(pa.float64()))),
            ("category_id", pa.list_(pa.int64())),
            ("supercategory", pa.list_(pa.string())),
            ("old_category_id", pa.list_(pa.int64())),
            ("name", pa.list_(pa.string())),
        ])
        path = os.path.join(output_dir, output_folder)
        os.makedirs(path, exist_ok=True)
        ddf.to_parquet(df=ddf_combined, path=path, write_index=False, schema=pyarrow_schema, compression="snappy", engine="pyarrow", name_function=lambda x: f"{output_folder}-{x}.parquet")
        print("[INFO] Exported data to {}".format(path))
