"""
Obtain CLIP features of training and test images in Things-EEG.

using huggingface pretrained CLIP model

"""

import argparse
import torch.nn as nn
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Dataset – returns raw PIL images
class ImgDataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self):         return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return img, i

def collate_pil(batch):
    images, indices = zip(*batch)
    return list(images), list(indices)

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='Things-EEG2/Image_set', type=str)
parser.add_argument('--feature_type', default='final_embedding', choices=['final_embedding', 'hidden_states'], 
					help="""
                    Type of features to extract. 
					For final_embedding, the image features are the image embeddings from the shared CLIP subspace where the contrastive loss is computed - 1-D (768). 
					For hidden_states, the image features are all the hidden states of the last layer - 2-D (257, 1024).""")
args = parser.parse_args()
base_save_dir = os.path.join('NICE-EEG-main/image_features', args.feature_type)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
batch_size = 64
num_workers = 4

print('Extract feature maps CLIP <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model.eval()
model = model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("CLIP model loaded")

# Image directories
img_set_dir = args.project_dir
img_partitions = ["training_images", "test_images"]
for p in img_partitions:
	part_dir = os.path.join(img_set_dir, p)
	image_list = []
	for root, dirs, files in os.walk(part_dir):
		for file in files:
			if file.endswith(".jpg") or file.endswith(".JPEG"):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	# Create the saving directory if not existing
	save_dir = os.path.join(base_save_dir, p)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	dataset  = ImgDataset(image_list)
	loader   = DataLoader(dataset,
						batch_size=batch_size,
						shuffle=False,
						num_workers=num_workers,
						pin_memory=True,
						collate_fn=collate_pil)
	print("Extracting feature maps for ", p)
	# Extract and save the feature maps
	with torch.no_grad():  # fp16 inference
		for images, image_idxs in tqdm(loader):
			# tokenize and move to device
			vision_inputs = processor(text=["a photo of a cat", "a photo of a dog"],images=images, return_tensors="pt").to(device)
			if args.feature_type == "final_embedding":
				img_embeds    = model(**vision_inputs).image_embeds.cpu().numpy()        # (B, 768)
			else:
				img_embeds    = model(**vision_inputs).vision_model_output.last_hidden_state.cpu().numpy()        # (B, 257, 1024)
			print("Image embeddings shape: ", img_embeds.shape)
			# save each image’s features individually
			for i, image_idx in enumerate(image_idxs):
				file_name = p + '_' + format(image_idx+1, '07')
				np.save(os.path.join(save_dir, file_name), img_embeds[i])

print('Done extracting feature maps CLIP')

print("Start combining feature maps into the same numpy file...")
for p in ['training', 'test']:
	# Load the feature maps
	feats = []
	save_dir = os.path.join(base_save_dir, p + '_images')
	fmaps_list = os.listdir(save_dir)
	fmaps_list.sort()
	for f, fmaps in enumerate(fmaps_list):
		fmaps_data = np.load(os.path.join(save_dir, fmaps))
		feats.append(fmaps_data)

	# Save all the train/test feature maps into one numpy file
	file_name = 'clip_feature_maps_' + p
	np.save(os.path.join(save_dir, file_name), feats)
	del feats

	# Delete the individual numpy files after combining
	for fmaps in fmaps_list:
		os.remove(os.path.join(save_dir, fmaps))

print("Done extracting the feature maps!")