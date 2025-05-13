import argparse
import os
from pprint import pprint

import torch
from tqdm import tqdm
from models.SuperNICE import SuperNICE
from utils.utils import load_model
from utils.dataset import get_dataloaders, get_test_dataloader
parser = argparse.ArgumentParser(description='Test the model for Stimuli Recognition')
parser.add_argument('--device', default='gpu', type=str, choices=['gpu', 'cpu'], help='Device to use for testing.')
parser.add_argument('--dnn', default='clip', type=str)

parser.add_argument('--subject_id', default=1, type=int, help='Subject ID to test.')
parser.add_argument('--checkpoint_path', default='NICE-EEG-main/results/checkpoints', type=str)
parser.add_argument('--checkpoint_suffix', default=None, type=str, help='Suffix of the checkpoint to load (part after "modelName-").')
parser.add_argument('--dataset_path', default='Things-EEG2/', type=str, help='Path to the dataset. ')

# Parameters for the model to load
parser.add_argument('--use_attn', action='store_true', help='If True, will use attention.')
parser.add_argument('--att_heads', default=4, type=int, help='Number of attention heads.')
parser.add_argument('--att_blocks', default=3, type=int, help='Number of attention blocks.')
parser.add_argument('--att_dropout', default=0.5, type=float, help='Dropout rate for the attention.')
parser.add_argument('--proj_dim', default=256, type=int, help='Dimension of the projected features + attention embeddings.')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
print(f'Using device: {device}')

# Pre-run setup
args.image_features_type = 'hidden_states' if args.use_attn else 'final_embedding'
args.debug_higher_scores = None

pprint(args)

eeg_data_path = os.path.join(args.dataset_path, 'Preprocessed_data_250Hz')
img_data_path = os.path.join(args.dataset_path, 'image_features', args.image_features_type)

# Load the model
model = SuperNICE(args)
model = load_model(model, args.checkpoint_path, args.checkpoint_suffix, device)
model.to(device)

# Test the model
test_batch_size = 400
test_loader, test_centers = get_test_dataloader(
    eeg_data_path, 
    img_data_path, 
    args.dnn,
    args.subject_id
    )

# Test the model
model.eval()

total = 0
top1 = 0
top3 = 0
top5 = 0

with torch.no_grad():
    for teeg, tlabel in tqdm(test_loader):
        teeg = teeg.to(device)
        tlabel = tlabel.to(device)
        timg = test_centers.to(device)

        # Feed through the model
        tfea, timg = model(teeg, timg)

        similarity = (tfea @ timg.t()).softmax(dim=-1)
        _, indices = similarity.topk(5)

        tt_label = tlabel.view(-1, 1)
        total += tlabel.size(0)
        top1 += (tt_label == indices[:, :1]).sum().item()
        top3 += (tt_label == indices[:, :3]).sum().item()
        top5 += (tt_label == indices).sum().item()

    
    top1_acc = float(top1) / float(total)
    top3_acc = float(top3) / float(total)
    top5_acc = float(top5) / float(total)


print(f'>> Subject {args.subject_id} - The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
