# EEEG-decoder

## Dataset
To fetch the Things-EEG2 data, run `data/get_data.sh` in the desired target directory.

## Run Params

- `use_attn` - if used, it will use bi-directional cross-attention for aligning the EEG/visual features.

    The type of image features being used **depends on this**.
	- **Without** attention, the image features are the *image embeddings* from the shared CLIP subspace where the contrastive loss is computed: 1-D (768). 
	- **With** attention, the image features are all the *hidden states* of the last layer: 2-D (257, 1024), where first token is the CLS token