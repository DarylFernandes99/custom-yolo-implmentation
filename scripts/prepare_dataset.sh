#!/bin/bash

# Root dataset path
ROOT_PATH="/scratch/$USER/dataset"

# Raw annotations and images path
RAW_PATH="$ROOT_PATH/raw"
RAW_ANNOTATIONS_PATH="$RAW_PATH/annotations"
RAW_IMAGES_PATH="$RAW_PATH/images"

# Processed Parquet path
PROCESSED_PATH="$ROOT_PATH/processed"
PROCESSED_PARQUET_PATH="$PROCESSED_PATH/parquet"

# Check for argument
if [ -z "$1" ]; then
    echo "Usage: $0 <data_type>"
    echo "Available data types: train_images, val_images, test_images, annotations, stuff_annotations, all"
    exit 1
fi

DATA_TYPE=$1

# Create all required directories
mkdir -p $RAW_ANNOTATIONS_PATH $RAW_IMAGES_PATH $PROCESSED_PARQUET_PATH

DOWNLOAD_URLS=()

case $DATA_TYPE in
    "train_images")
        DOWNLOAD_URLS+=("http://images.cocodataset.org/zips/train2017.zip")
        ;;
    "val_images")
        DOWNLOAD_URLS+=("http://images.cocodataset.org/zips/val2017.zip")
        ;;
    "test_images")
        DOWNLOAD_URLS+=("http://images.cocodataset.org/zips/test2017.zip")
        ;;
    "annotations")
        DOWNLOAD_URLS+=("http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        ;;
    "stuff_annotations")
        DOWNLOAD_URLS+=("http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip")
        ;;
    "all")
        DOWNLOAD_URLS=(
            "http://images.cocodataset.org/zips/train2017.zip"
            "http://images.cocodataset.org/zips/val2017.zip"
            "http://images.cocodataset.org/zips/test2017.zip"
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip"
        )
        ;;
    *)
        echo "Invalid data type: $DATA_TYPE"
        echo "Available data types: train_images, val_images, test_images, annotations, stuff_annotations, all"
        exit 1
        ;;
esac

for url in "${DOWNLOAD_URLS[@]}"; do
    echo "> Downloading $url..."
    base_name="$(basename $url)"
    if [ -f "$RAW_IMAGES_PATH/$base_name" ]; then
        echo " - $base_name already exists, skipping download"
    else
        echo " - Downloading $base_name..."
        wget -O "$RAW_IMAGES_PATH/$base_name" "$url"
        if [ $? -eq 0 ]; then
            echo " - Downloaded $base_name successfully"
            echo " - Extracting $base_name..."
            if [[ "$url" == *"annotations"* ]]; then
                unzip -d "$RAW_PATH" "$RAW_IMAGES_PATH/$base_name"
            else
                unzip -d "$RAW_IMAGES_PATH" "$RAW_IMAGES_PATH/$base_name"
            fi
            echo " - Extracting $base_name completed"
        else
            echo " - Downloaded $base_name failed"
        fi
    fi
done

# Rename directories if they exist
[ -d "$RAW_IMAGES_PATH/train2017" ] && mv "$RAW_IMAGES_PATH/train2017" "$RAW_IMAGES_PATH/train"
[ -d "$RAW_IMAGES_PATH/val2017" ] && mv "$RAW_IMAGES_PATH/val2017" "$RAW_IMAGES_PATH/val"
[ -d "$RAW_IMAGES_PATH/test2017" ] && mv "$RAW_IMAGES_PATH/test2017" "$RAW_IMAGES_PATH/test"
