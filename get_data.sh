#!/usr/bin/env bash
# Download “A large and rich EEG dataset for modeling human visual object recognition”
# (Gifford et al., 2022 ‑ Figshare 18470912 v4) and unzip all nested archives
# while keeping the original folder structure.

set -euo pipefail

DATA_URL="https://plus.figshare.com/ndownloader/articles/18470912/versions/4"
TOP_ZIP="things_eeg2.zip"

# # 1. Download
# echo "Downloading dataset archive ..."
# wget --content-disposition -O "$TOP_ZIP" "$DATA_URL"

# # 2. Unpack top‑level downloaded archive
# echo "Unzipping top‑level archive ..."
# unzip -q "$TOP_ZIP"
# rm "$TOP_ZIP"                      

export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE # disable zipbomb detection
# 3. Recursively find every *.zip and expand it next to itself
echo "Expanding inner zip files ..."
find . -type f -name '*.zip' -print0 | while IFS= read -r -d '' ZIPFILE; do
    DIR="${ZIPFILE%.zip}"
    if [[ -d $DIR ]]; then
        echo "Skipping already‑extracted $ZIPFILE"
        continue
    fi
    echo "  -> $ZIPFILE"
    unzip -q "$ZIPFILE"
    rm "$ZIPFILE" # remove the inner zip file
done

echo "Finished. Dataset rooted at $(pwd)"