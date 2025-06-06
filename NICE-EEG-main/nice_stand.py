"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""

import os
import argparse
import time
import uuid
import numpy as np
import pandas as pd
from pprint import pprint
import logging as l

import torch
import torch.nn as nn

from functools import partialmethod
from tqdm import tqdm
from models.SuperNICE import SuperNICE
from utils.utils import (
    load_model,
    new_best_epoch,
    save_model,
    seed_experiments,
)
from utils.dataset import SMALL_RUN_RATIO, get_dataloaders, mixup


# gpus = [0]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
NICE_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(NICE_path, "results")
model_checkpoint_path = os.path.join(result_path, "checkpoints")
checkpoint_uuid = str(uuid.uuid4())[:8]


parser = argparse.ArgumentParser(
    description="Experiment Stimuli Recognition test with CLIP encoder"
)
# Architectures
parser.add_argument("--dnn", default="clip", type=str)
# Training parameters
parser.add_argument("--epoch", default="150", type=int)
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
parser.add_argument(
    "--split_val_set_per_condition",
    action="store_true",
    help="Get the val set by splitting by conditions, keeping all samples for each condition together.",
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
    choices=["debug", "small_run", "no_patience"],
    help="If `debug`, will run in debug mode with only 100 samples per subject. If `small_run`, will use only  25% of the dataset. If `no_patience`, will not use early stopping.",
)

# Mixup parameters
parser.add_argument("--mixup", action="store_true", help="Use mixup data augmentation")
parser.add_argument(
    "--mixup-alpha", type=float, default=0.3, help="Mixup alpha parameter"
)
parser.add_argument(
    "--mixup_in_class",
    action="store_true",
    help="Use mixup data augmentation within the same class",
)

# Experiment parameters
parser.add_argument("--lr", default=0.0002, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=None, type=float, help="Weight decay for AdamW, if None, will use Adam.")
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
    "--saliency",
    action="store_true",
    help="""
                    If true, will calculate the saliency maps, otherwise will skip it.
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
run_name = f"mixup({args.mixup_alpha if args.mixup else 'none'})-{run_name}"
run_name = f"spatial({args.config})-denoiser({args.use_eeg_denoiser})-{run_name}"

# ===== Pre-run setup =====
dataset_mode = None
if args.mode == "small_run":
    epochs = 30
    print(f">>> Training with small run ({SMALL_RUN_RATIO*100}% of the dataset and {epochs} epochs)")
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
        self.saliency = args.saliency
        self.val_set_size = 740
        self.nSub = nsub

        self.use_mixup = args.mixup
        self.mixup_alpha = args.mixup_alpha
        self.use_mixup_in_class = args.mixup_in_class

        # Optimizer parameters
        self.lr = args.lr
        self.b1 = 0.5
        self.b2 = 0.999
        self.weight_decay = args.weight_decay

        self.start_epoch = 0
        self.eeg_data_path = os.path.join(args.dataset_path, "Preprocessed_data_250Hz")
        self.img_data_path = os.path.join(
            args.dataset_path, "image_features", "final_embedding"
        )

        os.makedirs(result_path, exist_ok=True)

        self.model = SuperNICE(args)
        self.model.to(device)
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
            eeg_denoiser=self.args.use_eeg_denoiser,
            dataset_mode=dataset_mode,
            val_set_per_condition=self.args.split_val_set_per_condition
        )

        # Optimizer & Scheduler
        if self.weight_decay is not None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=self.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2)
            )


        # Metrics
        best_val_top1 = 0
        best_val_loss = np.inf
        # dim1 - for train/val, dim2 - for epoch, dim3 - for loss/loss_eeg/loss_img/top1acc
        train_results = np.zeros(
            (2, self.n_epochs, 4)
        )
        epochs_no_gain   = 0
        patience         = 10             # stop if no gain for 10 epochs after epoch_patience
        epoch_patience   = self.n_epochs if args.mode in ["no_patience", "small_run"] else self.n_epochs // 3

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
                    mixed_eeg, mixed_img = mixup(
                        self.mixup_alpha, eeg, img_features, device
                    )

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
            
                # self.scheduler.step()
            # Log epoch metrics
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            avg_epoch_loss_eeg = sum(epoch_losses_eeg) / len(epoch_losses_eeg)
            avg_epoch_loss_img = sum(epoch_losses_img) / len(epoch_losses_img)

            train_results[0, e, 0] = avg_epoch_loss
            train_results[0, e, 1] = avg_epoch_loss_eeg
            train_results[0, e, 2] = avg_epoch_loss_img
            # No top1 accuracy for the train set

            # ===== Validation =====
            if (e + 1) % 1 == 0:
                self.model.eval()

                with torch.no_grad():
                    # * validation part
                    val_losses = []
                    val_losses_eeg = []
                    val_losses_img = []
                    val_top1 = 0
                    total_val_samples = 0
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

                        # top1 accuracy for the val set
                        similarity = (veeg_features @ vimg_features.t()).softmax(dim=-1)
                        _, indices = similarity.topk(1)
                        vlabels = vlabels.view(-1, 1)
                        val_top1 += (indices == vlabels).sum().item()
                        total_val_samples += vlabels.size(0)

                    avg_val_loss = sum(val_losses) / len(val_losses)
                    avg_val_loss_eeg = sum(val_losses_eeg) / len(val_losses_eeg)
                    avg_val_loss_img = sum(val_losses_img) / len(val_losses_img)
                    val_top1 = val_top1 / total_val_samples
                    train_results[1, e, 0] = avg_val_loss
                    train_results[1, e, 1] = avg_val_loss_eeg
                    train_results[1, e, 2] = avg_val_loss_img
                    train_results[1, e, 3] = val_top1

                    if new_best_epoch(args.split_val_set_per_condition, best_val_loss, best_val_top1, avg_val_loss, val_top1):
                        best_val_top1 = val_top1
                        best_val_loss = avg_val_loss
                        best_epoch = e + 1
                        epochs_no_gain = 0
                        os.makedirs(model_checkpoint_path, exist_ok=True)
                        print(f"!!! New best epoch - {best_epoch}")
                        # Save models, handling both DataParallel and non-DataParallel cases
                        save_model(
                            self.model, model_checkpoint_path, run_name, self.nSub, checkpoint_uuid
                        )
                    elif e >= epoch_patience:
                        epochs_no_gain += 1

                print(
                    "Epoch:",
                    e + 1,
                    "  Avg train loss: %.4f" % avg_epoch_loss,
                    "  Val top1: %.4f" % val_top1,
                    "  Val loss: %.4f" % avg_val_loss,
                )

            endtime_epoch = time.time()
            print(f"Epoch {e + 1} took {endtime_epoch - starttime_epoch} seconds")

            # Early stopping
            if epochs_no_gain >= patience:
                print(f">>> No gain for {patience} epochs after epoch {epoch_patience}, stop training")
                break

        # ===== Test =====
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0
        n_way_totals = [0] * len(self.n_ways)
        n_way_top1 = [0] * len(self.n_ways)

        self.model, save_path = load_model(
            self.model, model_checkpoint_path, run_name, self.nSub, checkpoint_uuid
        )
        self.model.eval()

        saliencies = []

        with torch.set_grad_enabled(self.saliency):
            for teeg, tlabel in tqdm(test_loader):
                teeg = teeg.to(device)
                tlabel = tlabel.to(device)
                timg = test_centers.to(device)

                if self.saliency:
                    teeg.requires_grad = True

                # Feed through the model
                tfea, timg = self.model(teeg, timg)

                similarity = (tfea @ timg.t()).softmax(dim=-1)
                _, indices = similarity.topk(5)
                scores, _ = similarity.max(dim=-1)

                if self.saliency:
                    # Calculate saliency maps
                    scores.sum().backward()
                    saliency_map = teeg.grad.data
                    saliencies.append(
                        saliency_map.abs().mean(dim=-1).mean(dim=-1).cpu().numpy()
                    )
                    teeg.grad.data.zero_()

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

            saliencies = np.concat(saliencies, axis=0).mean(axis=0) if self.saliency else []

        print(
            f">> Subject {self.nSub} - The test Top1-%.6f, Top3-%.6f, Top5-%.6f"
            % (top1_acc, top3_acc, top5_acc)
        )
        print(f"Subject {self.nSub} - n-way Top1 accuracies:")
        for i, n_way in enumerate(self.n_ways):
            print(f"  {n_way}-way: {n_way_top1_acc[i]:.6f}")

        return (
            top1_acc,
            top3_acc,
            top5_acc,
            train_results,
            n_way_top1_acc,
            best_val_top1,
            saliencies,
        )
        # writer.close()


def main():
    assert (
        args.use_eeg_denoiser or not args.saliency
    ), "Saliency maps can only be calculated if EEG denoiser is used."
    num_sub = args.num_sub
    n_ways = [2, 5, 10]
    cal_num = 0
    avg_train_results = []
    aver = []
    aver3 = []
    aver5 = []
    avern = []
    all_saliencies = []
    best_val_top1_avg = 0
    for i in range(num_sub):

        cal_num += 1
        starttime = time.time()

        print("Subject %d" % (i + 1))
        ie = IE(args, n_ways, i + 1)

        Acc, Acc3, Acc5, train_results, n_way_top1_acc, best_val_top1, saliencies = (
            ie.train()
        )
        print("THE BEST ACCURACY IS " + str(Acc))

        endtime = time.time()
        print("subject %d duration: %.2f minutes" % (i + 1, (endtime - starttime) / 60))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)
        avern.append(n_way_top1_acc)
        all_saliencies.append(saliencies)
        best_val_top1_avg += best_val_top1

        avg_train_results.append(train_results)

    # Compute and log average train/validation results
    avg_train_results = np.mean(avg_train_results, axis=0)  # size: (2, epochs, 3)
    for i in range(len(avg_train_results)):
        mode = "train" if i == 0 else "val"
        for e in range(len(avg_train_results[i])):
            epoch = e + 1
            for k in range(len(avg_train_results[i][e])):
                metric_name = {0: "loss", 1: "loss_eeg", 2: "loss_img", 3: "top1"}

    # Compute and log test results
    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))
    avern_a = np.array(avern)
    avern.append(np.mean(avern_a, axis=0))

    # Log saliency maps
    if args.saliency:
        all_saliencies = np.array(all_saliencies)
        saliency_mean = np.mean(all_saliencies, axis=0)
        saliency_std = np.std(all_saliencies, axis=0)

    column = np.arange(1, cal_num + 1).tolist()
    column.append("ave")
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(os.path.join(result_path, "train_results.csv"))

    # Print results table
    print("\nResults Table:")
    print("-" * 120)
    headers = [f"Subject{i+1}" for i in range(10)] + ["Average"]
    print(f"{'':12} " + " ".join(f"{h:>8}" for h in headers))
    print("-" * 120)
    print(f"{'Top-1':12} " + " ".join(f"{acc:8.2f}" for acc in aver))
    print(f"{'Top-3':12} " + " ".join(f"{acc:8.2f}" for acc in aver3))
    print(f"{'Top-5':12} " + " ".join(f"{acc:8.2f}" for acc in aver5))
    print("-" * 120)

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
