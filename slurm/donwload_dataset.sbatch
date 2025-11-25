#!/bin/bash
#SBATCH --partition=courses
#SBATCH --job-name=train_data_download    # Job name
#SBATCH --output=download_log.txt   # Standard output and error log
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --time=05:00:00             # Time limit hrs:min:sec
#SBATCH --mem-per-cpu=24GB         # Memory limit

# Define variables
DOWNLOAD_DIR="/scratch/$USER/"
TRAIN_FILE_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_FILE_URL="http://images.cocodataset.org/zips/val2017.zip"

# Create the directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Navigate to the download directory (optional, but good practice)
cd "$DOWNLOAD_DIR" || exit

# Download the file using wget
# The -O flag can specify the output path and filename relative to the current directory
echo "Starting download of $TRAIN_FILE_URL to $DOWNLOAD_DIR"
wget -O "$DOWNLOAD_DIR/$(basename $TRAIN_FILE_URL)" "$TRAIN_FILE_URL"

echo "Starting download of $VAL_FILE_URL to $DOWNLOAD_DIR"
wget -O "$DOWNLOAD_DIR/$(basename $VAL_FILE_URL)" "$VAL_FILE_URL"

if [ $? -eq 0 ]; then
    unzip stuff_annotations_trainval2017.zip
    unzip instances_annotations_trainval2017.zipunzip stuff_annotations_trainval2017.zip
    echo "Download completed successfully."
else
    echo "Download failed."
fi
