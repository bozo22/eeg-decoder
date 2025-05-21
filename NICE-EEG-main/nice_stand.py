"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
from pprint import pprint
import logging as l
import wandb

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from functools import partialmethod
from tqdm import tqdm
from models.SuperNICE import SuperNICE
from utils.utils import (
    load_model,
    save_checkpoint_wandb,
    save_model,
    seed_experiments,
    wandb_login,
)
from utils.dataset import get_dataloaders, mixup

# gpus = [0]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
NICE_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(NICE_path, "results")
model_checkpoint_path = os.path.join(result_path, "checkpoints")


parser = argparse.ArgumentParser(
    description="Experiment Stimuli Recognition test with CLIP encoder"
)
# Architectures
parser.add_argument("--dnn", default="clip", type=str)
# Training parameters
parser.add_argument("--epoch", default="200", type=int)
parser.add_argument(
    "--num_sub",
    default=10,
    type=int,
    help="number of subjects used in the experiments. ",
)
parser.add_argument(
    "--batch_size",
    default=1000,
    type=int,
    help="mini-batch size, this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--dataset_path", default="Things-EEG2/", type=str, help="Path to the dataset. "
)
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
parser.add_argument(
    "--mode",
    default=None,
    type=str,
    choices=["debug", "small_run"],
    help="If `debug`, will run in debug mode with only 100 samples per subject. If `small_run`, will use only  25% of the dataset.",
)
parser.add_argument("--split_val_set_per_condition", action="store_true", help="Get the val set by splitting by conditions, keeping all samples for each condition together.")
parser.add_argument("--mixup", action="store_true", help="Use mixup data augmentation")
parser.add_argument(
    "--mixup-alpha", type=float, default=0.2, help="Mixup alpha parameter"
)
parser.add_argument(
    "--mixup_in_class",
    action="store_true",
    help="Use mixup data augmentation within the same class",
)
# WandB parameters
parser.add_argument(
    "--disable_wandb", action="store_true", help="If True, will not use wandb."
)
parser.add_argument(
    "--run_group", default=None, type=str, help="Group name for the WandB run."
)

# Experiment parameters
parser.add_argument("--lr", default=0.0002, type=float, help="Learning rate.")
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
if args.mixup and args.mixup_in_class:
    raise ValueError(
        "Cannot use both mixup across classes and mixup within the same class. Please choose one."
    )
pprint(args)

# ===== WandB setup =====
wandb_login(args.disable_wandb)
run = wandb.init(
    entity="EEG_decoder",
    project="EEG-Decoder",
    config=vars(args),
    mode="disabled" if args.disable_wandb else "online",
    group=args.run_group,
)
for k, v in run.config.items():
    setattr(args, k, v)
pprint(args)

# ===== WandB metrics =====
wandb.define_metric("epoch")
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("val/*", step_metric="epoch")

# ===== Set device =====
device = torch.device(
    "cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
)
print(f"Using device: {device}")
# ===== Set debug logger, if debug is True =====
if args.mode == "debug":
    l.basicConfig(level=l.DEBUG, format="%(levelname)s: %(message)s")
    l.debug(">>> Running in DEBUG mode!")
tqdm.__init__ = partialmethod(tqdm.__init__, disable=False if args.mode == "debug" else True)

# ===== Seed experiments =====
seed_experiments(args.seed)

# ===== Prepare run name =====
run_name = f"lr({args.lr})-proj_dim({args.proj_dim})"
if args.mode == "debug":
    run_name = "[DEBUG]" + run_name

if args.use_image_projector:
    print(f">>> Using image projector")
    run_name = f"useIP-" + run_name
else:
    print(f">>> Skipping image projector")
    run_name = f"skipIP-" + run_name
run_name = f"{args.eeg_patch_encoder}-" + run_name
run.name = run_name

# ===== Pre-run setup =====
dataset_mode = None
if args.mode == "small_run":
    epochs = 30
    print(f">>> Training with small run (25% of the dataset and {epochs} epochs)")
    dataset_mode = "small"
    args.epoch = epochs
elif args.mode == "debug":
    print(">>> Training with debug mode (100 training EEG samples per subject only)")
    dataset_mode = "debug"

# Image2EEG
class IE:
    def __init__(self, args, n_ways, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500
        self.n_epochs = args.epoch
        self.n_ways = n_ways
        self.val_set_size = 740
        self.nSub = nsub

        self.use_mixup = args.mixup
        self.mixup_alpha = args.mixup_alpha
        self.use_mixup_in_class = args.mixup_in_class

        # Optimizer parameters
        self.lr = args.lr
        self.b1 = 0.5
        self.b2 = 0.999
        self.weight_decay = 1e-4

        self.start_epoch = 0
        self.eeg_data_path = os.path.join(args.dataset_path, "Preprocessed_data_250Hz")
        self.img_data_path = os.path.join(
            args.dataset_path, "image_features", "final_embedding"
        )

        os.makedirs(result_path, exist_ok=True)

        self.model = SuperNICE(args)
        self.model.to(device)

        # self.criterion_l1 = torch.nn.L1Loss().cuda()
        # self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(device)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        print("initial define done.")

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def train(self):

        self.model.init_weights()

        (
            train_loader,
            val_loader,
            test_loader,
            test_centers,
            test_n_way_loaders,
            test_n_way_centers,
        ) = get_dataloaders(
            self.eeg_data_path,
            self.img_data_path,
            self.args.dnn,
            self.nSub,
            self.batch_size,
            mixup_in_class=self.use_mixup_in_class, 
            use_mixup=self.use_mixup, 
            mixup_val_set_size=self.val_set_size,
            n_ways=self.n_ways,
            dataset_mode=dataset_mode,
            val_set_per_condition=self.args.split_val_set_per_condition
        )

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=self.weight_decay
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr    = self.lr,            # same base
            div_factor= 25,              # start lr = max_lr/25
            pct_start = 0.3,             # up-ramp for 30 % of training
            total_steps = len(train_loader) * epochs,
)

        num = 0
        best_loss_val = np.inf
        train_results = np.zeros(
            (2, self.n_epochs, 3)
        )  # dim1 - for train/val, dim2 - for epoch, dim3 - for loss/loss_eeg/loss_img

        for e in range(self.n_epochs):
            epoch_losses = []
            epoch_losses_eeg = []
            epoch_losses_img = []

            self.model.train()
            starttime_epoch = time.time()

            # ===== Training =====
            for eeg, img in tqdm(train_loader):

                # img = Variable(img.cuda().type(self.Tensor))
                eeg = eeg.to(device)
                img_features = img.to(device)
                

                if self.use_mixup:
                    # Apply mixup to both EEG and image features using the same permutation and lambda
                    # This maintains correspondence between mixed samples
                    mixed_eeg, mixed_img = mixup(self.mixup_alpha, eeg, img_features, device)

                    # Ensure all tensors are of the same type
                    mixed_eeg = mixed_eeg.type(torch.FloatTensor).to(device)
                    mixed_img = mixed_img.type(torch.FloatTensor).to(device)
                    
                    eeg = torch.concatenate((eeg, mixed_eeg), axis=0)
                    img_features = torch.concatenate((img_features, mixed_img), axis=0)

                labels = torch.arange(eeg.shape[0]).to(device)
                eeg_features, img_features = self.model(eeg, img_features)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss
                loss = loss_cos
                epoch_losses.append(loss.item())
                epoch_losses_eeg.append(loss_eeg.item())
                epoch_losses_img.append(loss_img.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                self.scheduler.step()
            # Log epoch metrics
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            avg_epoch_loss_eeg = sum(epoch_losses_eeg) / len(epoch_losses_eeg)
            avg_epoch_loss_img = sum(epoch_losses_img) / len(epoch_losses_img)
            wandb.log(
                {
                    "epoch": e + 1,
                    f"train/loss/subj{self.nSub}": avg_epoch_loss,
                    f"train/loss_eeg/subj{self.nSub}": avg_epoch_loss_eeg,
                    f"train/loss_img/subj{self.nSub}": avg_epoch_loss_img,
                }
            )
            train_results[0, e, 0] = avg_epoch_loss
            train_results[0, e, 1] = avg_epoch_loss_eeg
            train_results[0, e, 2] = avg_epoch_loss_img

            # ===== Validation =====
            if (e + 1) % 1 == 0:
                self.model.eval()

                with torch.no_grad():
                    # * validation part
                    val_losses = []
                    val_losses_eeg = []
                    val_losses_img = []
                    for veeg, vimg in tqdm(val_loader):

                        veeg = veeg.to(device)
                        vimg = vimg.to(device)
                        vlabels = torch.arange(veeg.shape[0]).to(device)

                        # Feed through the model
                        veeg_features, vimg_features = self.model(veeg, vimg)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = (
                            logit_scale * veeg_features @ vimg_features.t()
                        )
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2
                        val_losses.append(vloss.item())
                        val_losses_eeg.append(vloss_eeg.item())
                        val_losses_img.append(vloss_img.item())

                    avg_val_loss = sum(val_losses) / len(val_losses)
                    avg_val_loss_eeg = sum(val_losses_eeg) / len(val_losses_eeg)
                    avg_val_loss_img = sum(val_losses_img) / len(val_losses_img)
                    wandb.log(
                        {
                            "epoch": e + 1,
                            f"val/loss/subj{self.nSub}": avg_val_loss,
                            f"val/loss_eeg/subj{self.nSub}": avg_val_loss_eeg,
                            f"val/loss_img/subj{self.nSub}": avg_val_loss_img,
                        }
                    )
                    train_results[1, e, 0] = avg_val_loss
                    train_results[1, e, 1] = avg_val_loss_eeg
                    train_results[1, e, 2] = avg_val_loss_img
                    if vloss <= best_loss_val:
                        best_loss_val = vloss
                        best_epoch = e + 1
                        os.makedirs(model_checkpoint_path, exist_ok=True)
                        print(f"New best epoch - {best_epoch}")
                        # Save models, handling both DataParallel and non-DataParallel cases
                        save_model(
                            self.model, model_checkpoint_path, run_name, self.nSub
                        )

                print(
                    "Epoch:",
                    e + 1,
                    "  Cos eeg: %.4f" % loss_eeg.detach().cpu().numpy(),
                    "  Cos img: %.4f" % loss_img.detach().cpu().numpy(),
                    "  loss val: %.4f" % vloss.detach().cpu().numpy(),
                )

            endtime_epoch = time.time()
            print(f"Epoch {e + 1} took {endtime_epoch - starttime_epoch} seconds")

        # ===== Test =====
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0
        n_way_totals = [0] * len(self.n_ways)
        n_way_top1 = [0] * len(self.n_ways)

        self.model, save_path = load_model(
            self.model, model_checkpoint_path, run_name, self.nSub
        )
        save_checkpoint_wandb(save_path, self.nSub, best_loss_val)
        self.model.eval()

        with torch.no_grad():
            for teeg, tlabel in tqdm(test_loader):
                teeg = teeg.to(device)
                tlabel = tlabel.to(device)
                timg = test_centers.to(device)

                # Feed through the model
                tfea, timg = self.model(teeg, timg)

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

            for i, (test_n_way_loader, test_n_way_center) in enumerate(
                zip(test_n_way_loaders, test_n_way_centers)
            ):
                for teeg, tlabel in tqdm(test_n_way_loader):
                    teeg = teeg.to(device)
                    tlabel = tlabel.to(device)
                    timg = test_n_way_center.to(device)

                    # Feed through the model
                    tfea, timg = self.model(teeg, timg)

                    similarity = (tfea @ timg.t()).softmax(dim=-1)
                    _, indices = similarity.topk(1)

                    tt_label = tlabel.view(-1, 1)
                    n_way_totals[i] += tlabel.size(0)
                    n_way_top1[i] += (tt_label == indices[:, :1]).sum().item()

            n_way_top1_acc = [
                float(n_way_top1[i]) / float(n_way_totals[i])
                for i in range(len(n_way_top1))
            ]

        print(
            f">> Subject {self.nSub} - The test Top1-%.6f, Top3-%.6f, Top5-%.6f"
            % (top1_acc, top3_acc, top5_acc)
        )
        print(f"Subject {self.nSub} - n-way Top1 accuracies:")
        for i, n_way in enumerate(self.n_ways):
            print(f"  {n_way}-way: {n_way_top1_acc[i]:.6f}")

        return top1_acc, top3_acc, top5_acc, train_results, n_way_top1_acc
        # writer.close()


def main():
    num_sub = args.num_sub
    n_ways = [2, 5, 10]
    cal_num = 0
    avg_train_results = []
    aver = []
    aver3 = []
    aver5 = []
    avern = []

    for i in range(num_sub):

        cal_num += 1
        starttime = time.time()
        seed_n = np.random.randint(args.seed)

        print("Subject %d" % (i + 1))
        ie = IE(args, n_ways, i + 1)

        Acc, Acc3, Acc5, train_results, n_way_top1_acc = ie.train()
        print("THE BEST ACCURACY IS " + str(Acc))

        endtime = time.time()
        print("subject %d duration: %.2f minutes" % (i + 1, (endtime - starttime) / 60))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)
        avern.append(n_way_top1_acc)

        avg_train_results.append(train_results)

    # Compute and log average train/validation results
    avg_train_results = np.mean(avg_train_results, axis=0)  # size: (2, epochs, 3)
    for i in range(len(avg_train_results)):
        mode = "train" if i == 0 else "val"
        for e in range(len(avg_train_results[i])):
            epoch = e + 1
            for k in range(len(avg_train_results[i][e])):
                metric = "loss" if k == 0 else "loss_eeg" if k == 1 else "loss_img"
                wandb.log(
                    {"epoch": epoch, f"{mode}/{metric}": avg_train_results[i][e][k]}
                )
    # Compute and log test results
    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))
    avern_a = np.array(avern)
    avern.append(np.mean(avern_a, axis=0))

    for i in range(len(aver)):
        if i == len(aver) - 1:
            subj = "ave"
        else:
            subj = f"Subject {i+1}"
        wandb.run.summary[f"{subj}/top1"] = aver[i]
        wandb.run.summary[f"{subj}/top3"] = aver3[i]
        wandb.run.summary[f"{subj}/top5"] = aver5[i]
        for j in range(len(avern[i])):
            wandb.run.summary[f"{subj}/{n_ways[j]}-way"] = avern[i][j]

    column = np.arange(1, cal_num + 1).tolist()
    column.append("ave")
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(os.path.join(result_path, "train_results.csv"))

    run.finish()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
