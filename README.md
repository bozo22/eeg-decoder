# EEEG-decoder

## Dataset
To fetch the Things-EEG2 data, run `data/get_data.sh` in the desired target directory.

## Run Params

- `--use_image_projector` - if used, will use the image projector to project the CLIP embeddings into the shared bi-modal space
- `--mode` - options: ["debug", "small_run"]. If `debug`, will run in debug mode with only 100 samples per subject. If `small_run`, will use only  25% of the dataset.
- `--split_val_set_per_condition` - if used, will split train/val by conditions, so that all samples from a condition are in the same set.