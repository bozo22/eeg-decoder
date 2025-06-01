import argparse
import os
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
from models.SuperNICE import SuperNICE
from utils.utils import load_model
from utils.dataset import get_test_dataloader

parser = argparse.ArgumentParser(description='Test the model for Stimuli Recognition')
# Main parameters
parser.add_argument("--dnn", default="clip", type=str)
parser.add_argument(
    "--dataset_path", default="Things-EEG2/", type=str, help="Path to the dataset. ")
parser.add_argument("--subject_id", default=None, type=int, help="Subject ID to test. If None, will test all subjects.")
parser.add_argument("--checkpoint_path", default="NICE-EEG-main/results/SuperNICE", type=str, help="Path to the checkpoints.")
parser.add_argument("--checkpoint_run_name", type=str, help="Run name of the checkpoint to load (part after 'modelName-' and before '-subx')")
parser.add_argument("--checkpoint_uuid", default="61649fe6", type=str, help="UUID of the checkpoint to load.")
# Auxiliary parameters
parser.add_argument(
    "--seed", default=2023, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--device",
    default="gpu",
    type=str,
    choices=["gpu", "cpu"],
    help="Device to use for training.",
)
# Parameters for the model to load
parser.add_argument(
    "--proj_dim",
    default=768,
    type=int,
    help="Dimension of the projected features.",
)
parser.add_argument(
    "--use_image_projector",
    action="store_true",
    help="""
                    If true, will use the image projector, otherwise will skip it.
                    """,
)
parser.add_argument(
    "--use_eeg_denoiser",
    action="store_true",
    help="""
                    If true, will use the eeg denoiser, otherwise will skip it.
                    """,
)
parser.add_argument(
    "--config",
    default="GASA",
    type=str,
    choices=["SA", "GA", "SAGA", "GASA", "none"],
    help="Configuration for the EEG encoder.",
)
parser.add_argument(
    "--eeg_patch_encoder",
    default="tsconv",
    type=str,
    choices=["tsconv", "multiscale_1block", "multiscale_2block"],
    help="Configuration for the EEG patch encoder"
)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
print(f'Using device: {device}')
pprint(args)

eeg_data_path = os.path.join(args.dataset_path, 'Preprocessed_data_250Hz')
img_data_path = os.path.join(args.dataset_path, 'image_features', "final_embedding")

subject_ids = [args.subject_id] if args.subject_id is not None else [i for i in range(1, 11)]
print(f"Subject IDs to test on: {subject_ids}")

SAVE_PATH = os.path.join(args.checkpoint_path, 'similarity')

for subject_id in subject_ids:

    print(f">> Testing subject {subject_id}...")
    # Load the model
    model = SuperNICE(args)
    model, _ = load_model(model, args.checkpoint_path, args.checkpoint_run_name, subject_id, args.checkpoint_uuid, device)
    model.to(device)

    # Test the model
    test_batch_size = 400
    test_loader, test_centers = get_test_dataloader(
        eeg_data_path, 
        img_data_path, 
        args.dnn,
        subject_id,
        args.use_eeg_denoiser
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
            np.save(os.path.join(SAVE_PATH, f'sub{subject_id}_sim.npy'), similarity.cpu().numpy())
            _, indices = similarity.topk(5)

            tt_label = tlabel.view(-1, 1)
            total += tlabel.size(0)
            top1 += (tt_label == indices[:, :1]).sum().item()
            top3 += (tt_label == indices[:, :3]).sum().item()
            top5 += (tt_label == indices).sum().item()

        
        top1_acc = float(top1) / float(total)
        top3_acc = float(top3) / float(total)
        top5_acc = float(top5) / float(total)


    print(f'>> Subject {subject_id} - The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
