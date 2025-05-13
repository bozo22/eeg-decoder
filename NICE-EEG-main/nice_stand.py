"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""

import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange
from einops import rearrange

# from topognn.models import TopologyLayer

from utils.utils import load_model, save_model, seed_experiments

# gpus = [0]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
NICE_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(NICE_path, "results")
model_checkpoint_path = os.path.join(result_path, "checkpoints")
model_idx = "test0"


parser = argparse.ArgumentParser(
    description="Experiment Stimuli Recognition test with CLIP encoder"
)
parser.add_argument("--dnn", default="clip", type=str)
parser.add_argument("--epoch", default="200", type=int)
parser.add_argument(
    "--num_sub",
    default=10,
    type=int,
    help="number of subjects used in the experiments. ",
)
parser.add_argument(
    "-batch_size",
    "--batch-size",
    default=1000,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--seed", default=2023, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--dataset_path",
    default="Things-EEG2/Preprocessed_data_250Hz/",
    type=str,
    help="Path to the dataset. ",
)
parser.add_argument(
    "--device",
    default="gpu",
    type=str,
    choices=["gpu", "cpu"],
    help="Device to use for training.",
)
parser.add_argument(
    "--config",
    default="GA",
    type=str,
    choices=["SA", "GA", "SAGA", "GASA"],
    help="Configuration for the EEG encoder.",
)
parser.add_argument("--channels", default=63, type=int, help="Number of EEG channels. ")

args = parser.parse_args()
pprint(args)

# Set device
device = torch.device(
    "cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
)
print(f"Using device: {device}")

# Import function
seed_experiments(args.seed)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname.find("GATConv") == -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


# class Enc_eeg(nn.Sequential):
#     def __init__(self, emb_size=40, **kwargs):
#         super().__init__(PatchEmbedding(emb_size), FlattenHead())


class channel_attention(nn.Module):
    def __init__(self, sequence_num=250, inter=30, n_channels=63):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(
            self.sequence_num / self.inter
        )  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(n_channels, n_channels), nn.LayerNorm(n_channels), nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(n_channels, n_channels), nn.LayerNorm(n_channels), nn.Dropout(0.3)
        )

        self.projection = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, "b o c s->b o s c")
        temp_query = rearrange(self.query(temp), "b o s c -> b o c s")
        temp_key = rearrange(self.key(temp), "b o s c -> b o c s")

        channel_query = temp_query
        channel_key = temp_key

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = (
            torch.einsum("b o c s, b o m s -> b o c m", channel_query, channel_key)
            / scaling
        )

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum("b o c s, b o c m -> b o c s", x, channel_atten_score)

        out = rearrange(out, "b o c s -> b o s c")
        out = self.projection(out)
        out = rearrange(out, "b o s c -> b o c s")
        return out


from torch_geometric.nn import GATConv


class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250, n_channels=63):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(
            in_channels=in_channels, out_channels=out_channels, heads=1
        )
        # self.conv2 = GATConv(in_channels=out_channels, out_channels=out_channels, heads=1)

        self.num_channels = n_channels
        # Create a list of tuples representing all possible edges between channels
        self.edge_index_list = torch.Tensor(
            [
                (i, j)
                for i in range(self.num_channels)
                for j in range(self.num_channels)
                if i != j
            ]
        ).cuda()
        # Convert the list of tuples to a tensor
        self.edge_index = (
            torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().cuda()
        )

    def forward(self, x):
        batch_size, _, num_channels, num_features = x.size()
        x = x.reshape(batch_size * num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)

        return x


class Enc_eeg(nn.Sequential):
    def __init__(
        self, emb_size=40, depth=3, n_classes=4, n_channels=63, config="GA", **kwargs
    ):
        spatial = None
        if config == "SA":
            spatial = nn.Sequential(
                nn.LayerNorm(250),
                channel_attention(n_channels=n_channels),
                nn.Dropout(0.3),
            )
        elif config == "GA":
            spatial = nn.Sequential(
                EEG_GAT(),
                # TopologyLayer(
                #     features_in=250,
                #     features_out=250,
                #     num_filtrations=8,
                #     num_coord_funs=3,
                #     filtration_hidden=True,
                # ),
                nn.Dropout(0.3),
            )
        elif config == "SAGA":
            spatial = nn.Sequential(
                nn.LayerNorm(250),
                channel_attention(n_channels=n_channels),
                nn.Dropout(0.3),
                EEG_GAT(),
                # TopologyLayer(
                #     features_in=250,
                #     features_out=250,
                #     num_filtrations=8,
                #     num_coord_funs=3,
                #     filtration_hidden=True,
                # ),
                nn.Dropout(0.3),
            )
        elif config == "GASA":
            spatial = nn.Sequential(
                EEG_GAT(),
                # TopologyLayer(
                #     features_in=250,
                #     features_out=250,
                #     num_filtrations=8,
                #     num_coord_funs=3,
                #     filtration_hidden=True,
                # ),
                nn.Dropout(0.3),
                nn.LayerNorm(250),
                channel_attention(n_channels=n_channels),
                nn.Dropout(0.3),
            )
        super().__init__(
            ResidualAdd(spatial),
            PatchEmbedding(emb_size),
            FlattenHead(emb_size, n_classes),
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x


# Image2EEG
class IE:
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = args.dataset_path
        self.img_data_path = os.path.join(NICE_path, "dnn_feature/")
        self.test_center_path = os.path.join(NICE_path, "dnn_feature/")
        self.pretrain = False

        os.makedirs(result_path, exist_ok=True)
        self.log_write = open(
            os.path.join(result_path, f"log_subject{self.nSub}.txt"), "w"
        )

        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Enc_eeg = Enc_eeg(n_channels=args.channels, config=args.config).to(device)
        self.Proj_eeg = Proj_eeg().to(device)
        self.Proj_img = Proj_img().to(device)
        # self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print("initial define done.")

    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)

        train_data = np.load(
            os.path.join(
                self.eeg_data_path,
                "sub-" + format(self.nSub, "02"),
                "preprocessed_eeg_training.npy",
            ),
            allow_pickle=True,
        )
        train_data = train_data["preprocessed_eeg_data"]
        train_data = np.mean(train_data, axis=1)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(
            self.eeg_data_path
            + "/sub-"
            + format(self.nSub, "02")
            + "/preprocessed_eeg_test.npy",
            allow_pickle=True,
        )
        test_data = test_data["preprocessed_eeg_data"]
        test_data = np.mean(test_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(
            self.img_data_path + self.args.dnn + "_feature_maps_training.npy",
            allow_pickle=True,
        )
        test_img_feature = np.load(
            self.img_data_path + self.args.dnn + "_feature_maps_test.npy",
            allow_pickle=True,
        )

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)

        return train_img_feature, test_img_feature

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def train(self):

        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        train_img_feature, _ = self.get_image_data()
        test_center = np.load(
            self.test_center_path + "center_" + self.args.dnn + ".npy",
            allow_pickle=True,
        )

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:740])
        val_image = torch.from_numpy(train_img_feature[:740])

        train_eeg = torch.from_numpy(train_eeg[740:])
        train_image = torch.from_numpy(train_img_feature[740:])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=False
        )

        test_eeg = torch.from_numpy(test_eeg)
        # test_img_feature = torch.from_numpy(test_img_feature)
        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False
        )

        # Optimizers
        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.Enc_eeg.parameters(),
                self.Proj_eeg.parameters(),
                self.Proj_img.parameters(),
            ),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):

                # img = Variable(img.cuda().type(self.Tensor))
                eeg = eeg.type(self.Tensor).to(device)
                img_features = img.type(self.Tensor).to(device)
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = labels.type(self.LongTensor).to(device)

                # obtain the features
                eeg_features = self.Enc_eeg(eeg)
                # img_features = self.Enc_img(img).last_hidden_state[:,0,:]

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)
                img_features = self.Proj_img(img_features)

                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss
                loss = loss_cos

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = veeg.type(self.Tensor).to(device)
                        vimg_features = vimg.type(self.Tensor).to(device)
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = vlabels.type(self.LongTensor).to(device)

                        veeg_features = self.Enc_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(
                            dim=1, keepdim=True
                        )
                        vimg_features = vimg_features / vimg_features.norm(
                            dim=1, keepdim=True
                        )

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = (
                            logit_scale * veeg_features @ vimg_features.t()
                        )
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1
                            os.makedirs(model_checkpoint_path, exist_ok=True)
                            # Save models, handling both DataParallel and non-DataParallel cases
                            save_model(self.Enc_eeg, model_checkpoint_path, model_idx)
                            save_model(self.Proj_eeg, model_checkpoint_path, model_idx)
                            save_model(self.Proj_img, model_checkpoint_path, model_idx)

                print(
                    "Epoch:",
                    e,
                    "  Cos eeg: %.4f" % loss_eeg.detach().cpu().numpy(),
                    "  Cos img: %.4f" % loss_img.detach().cpu().numpy(),
                    "  loss val: %.4f" % vloss.detach().cpu().numpy(),
                )
                self.log_write.write(
                    "Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n"
                    % (
                        e,
                        loss_eeg.detach().cpu().numpy(),
                        loss_img.detach().cpu().numpy(),
                        vloss.detach().cpu().numpy(),
                    )
                )

        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg = load_model(self.Enc_eeg, model_checkpoint_path, model_idx)
        self.Proj_eeg = load_model(self.Proj_eeg, model_checkpoint_path, model_idx)
        self.Proj_img = load_model(self.Proj_img, model_checkpoint_path, model_idx)

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = teeg.type(self.Tensor).to(device)
                tlabel = tlabel.type(self.LongTensor).to(device)
                all_center = all_center.type(self.Tensor).to(device)

                tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(
                    dim=-1
                )  # no use 100?
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)

        print(
            "The test Top1-%.6f, Top3-%.6f, Top5-%.6f" % (top1_acc, top3_acc, top5_acc)
        )
        self.log_write.write("The best epoch is: %d\n" % best_epoch)
        self.log_write.write(
            "The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n"
            % (top1_acc, top3_acc, top5_acc)
        )

        return top1_acc, top3_acc, top5_acc
        # writer.close()


def main():
    num_sub = args.num_sub
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []

    for i in range(num_sub):

        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print("Subject %d" % (i + 1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print("THE BEST ACCURACY IS " + str(Acc))

        endtime = datetime.datetime.now()
        print("subject %d duration: " % (i + 1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num + 1).tolist()
    column.append("ave")
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(os.path.join(result_path, "train_results.csv"))


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
