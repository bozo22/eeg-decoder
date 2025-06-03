# EEEG-decoder

This is the codebase for the SuperNICE project.

## Dataset
To fetch the Things-EEG2 data, run `data/get_data.sh` in the desired target directory.

## Run Params
Quick rundown of some of the run parameters:

- `--use_image_projector` - if used, will use the image projector to project the CLIP embeddings into the shared bi-modal space
- `--mode` - options: ["debug", "small_run"]. If `debug`, will run in debug mode with only 100 samples per subject. If `small_run`, will use only  25% of the dataset.
- `--split_val_set_per_condition` - if used, will split train/val by conditions, so that all samples from a condition are in the same set.

## Training the model

To train the model, first make sure you set the correct value for `--dataset_path`.
- Training Baseline (NICE-GA): `NICE-EEG-main/nice_stand.py --config none`
- Training Final SuperNICE: `NICE-EEG-main/nice_stand.py --use_image_projector --config GASA --use_eeg_denoiser --eeg_patch_encoder multiscale_1block --mixup --mixup-alpha 0.3 `

This will save one model checkpoint for each subject (10).

## Inference

For inference, run `NICE-EEG-main/inference.py`.
Once you have the checkpoints per subject, you need to set the following runtime args based on the location of the checkpoints and their names: `--checkpoint_path`, `--checkpoint_run_name`, `--checkpoint_uuid`
