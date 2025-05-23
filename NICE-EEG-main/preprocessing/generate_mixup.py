import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from torch import Tensor
import argparse
import math

CHUNK_NUMBER = 8

def mixup_data(x, lam, perm_index, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if type(x) is not Tensor:
        x = torch.from_numpy(x)

    if type(lam) is not Tensor:
        lam = torch.from_numpy(lam)

    batch_size = x.size()[0]

    if use_cuda:
        x = x.type(torch.FloatTensor).cuda()
        perm_index = perm_index.cuda()
        lam = lam.type(torch.FloatTensor).cuda()

    lam = lam.view(batch_size, 1, 1, 1)
    mixed_x = lam * x + (1 - lam) * x[perm_index]

    return mixed_x


def get_batches(x):
    """Split the data into chunks for mixup"""
    chunk_size = math.ceil(len(x) / CHUNK_NUMBER)
    slices = [chunk_size * i for i in range(1, CHUNK_NUMBER)]

    return np.split(x, slices)


def mixup_images(img_folder, img_output_folder, lams, index, perm_index):
    """
    Mixup images from the given folder and save them to the output folder.
    """
    images = []
    os.makedirs(img_output_folder, exist_ok=True)

    for sub_folder in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, sub_folder)):
            img = cv2.imread(os.path.join(img_folder, sub_folder, file))
            images.append(img)

    images = np.array(images)
    # permute the images
    images = images[index]
    num_images = len(images)
    name_width = len(str(num_images))

    # Save original images in the permuted order
    for num_img, img in zip(range(num_images), images):
        cv2.imwrite(os.path.join(img_output_folder, f'image_num_{num_img:0{name_width}d}.jpg'), img)

    print("Shape of images before mixup: ", images.shape)
    image_batches = get_batches(images)
    del images
    lam_batches = get_batches(lams)

    # mixup the images in batches so it fits in the GPU memory
    for i, batch in enumerate(image_batches):
        mixup_batch = mixup_data(batch, lam_batches[i], perm_index[i], use_cuda=True)
        mixup_batch = mixup_batch.to('cpu').numpy()
        for img in mixup_batch:
            cv2.imwrite(os.path.join(img_output_folder, f'image_num_{num_images:0{name_width}d}.jpg'), img)
            num_images += 1

    print("Number of images after mixup: ", num_images)


def mixup_eeg(eeg_folder, eeg_output_folder, lams, index, perm_index):
    os.makedirs(eeg_output_folder, exist_ok=True)

    for sub_folder in os.listdir(eeg_folder):
        for file in os.listdir(os.path.join(eeg_folder, sub_folder)):
            if "preprocessed_eeg_training.npy" in file:
                os.makedirs(os.path.join(eeg_output_folder, sub_folder), exist_ok=True)
                eeg_data = np.load(os.path.join(eeg_folder, sub_folder, file), allow_pickle=True)
                eeg = eeg_data['preprocessed_eeg_data']
                # permute the EEG data
                eeg = eeg[index]

                print("Shape of EEG data before mixup: ", eeg.shape)
                eeg_data['preprocessed_eeg_data'] = eeg
                np.save(os.path.join(eeg_output_folder, sub_folder, file), eeg_data)
                eeg_batches = get_batches(eeg)
                lam_batches = get_batches(lams)
                # make a list to store the mixup data, the first element is the permuted eeg data
                eeg = [eeg]

                # mixup the EEG data in batches so it fits in the GPU memory
                for i, batch in enumerate(eeg_batches):
                    mixup_batch = mixup_data(batch, lam_batches[i], perm_index[i], use_cuda=True)
                    mixup_batch = mixup_batch.to('cpu').numpy()
                    eeg.append(mixup_batch)
                
                eeg_data['preprocessed_eeg_data'] = np.concatenate(eeg, axis=0)
                np.save(os.path.join(eeg_output_folder, sub_folder, file), eeg_data)
                print("Shape of EEG data after mixup: ", eeg_data['preprocessed_eeg_data'].shape)
                del eeg

            if "preprocessed_eeg_test.npy" in file:
                os.makedirs(os.path.join(eeg_output_folder, sub_folder), exist_ok=True)
                eeg_data = np.load(os.path.join(eeg_folder, sub_folder, file), allow_pickle=True)
                np.save(os.path.join(eeg_output_folder, sub_folder, file), eeg_data)

            del eeg_data


def load_data_from_folder(img_folder, eeg_folder, num_cls, img_per_cls, img_output_folder, eeg_output_folder, alpha=0.4):
    """Load data from the given folder and perform mixup."""
    total_size = num_cls * img_per_cls
    chunk_size = math.ceil(total_size / CHUNK_NUMBER)

    lams = [np.random.beta(alpha, alpha) for _ in range(total_size)]
    # data_index is used to permutate the data
    # this is needed to make sure that the mixup is also happening between the image classes
    data_index = np.random.permutation(total_size)
    perm_index = [torch.randperm(chunk_size) for _ in range(CHUNK_NUMBER - 1)]

    last_chunk_size = total_size - (CHUNK_NUMBER-1) * chunk_size

    # perm_index is used to give the same permutation for mixup between images and between EEG data
    perm_index.append(torch.randperm(last_chunk_size))

    mixup_images(img_folder, img_output_folder, lams, data_index, perm_index)
    mixup_eeg(eeg_folder, eeg_output_folder, lams, data_index, perm_index)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mixup EEG and Image Features')
    parser.add_argument('--eeg_path', default='scratch-shared/datasets/Things-EEG2/Preprocessed_data_250Hz/', type=str, help='Path to the eeg dataset. ')
    parser.add_argument('--img_path', default='scratch-shared/training_images', type=str, help='Path to the image features dataset. ')
    parser.add_argument('--eeg_output_path', type=str, default='scratch-shared/datasets/Things-EEG2/mixup_preprocessed_data_250Hz/', help='Path to save mixed data')
    parser.add_argument('--img_output_path', type=str, default='scratch-shared/mixup_training_images', help='Path to save mixed data')
    parser.add_argument('--alpha', type=float, default=0.4, help='Mixup parameter alpha')
    parser.add_argument('--num_cls', default=1654, type=int, help='number of classes used in the experiments. ')
    parser.add_argument('--num_img_per_cls', default=10, type=int, help='number of images per class used in the experiments. ')
    parser.add_argument('--num_sub', default=10, type=int,
                        help='number of subjects used in the experiments. ')
    
    args = parser.parse_args()

    set_seed(2025)

    # Perform mixup
    load_data_from_folder(args.img_path, args.eeg_path, args.num_cls, args.num_img_per_cls, args.img_output_path, args.eeg_output_path, alpha=args.alpha)

    print("Mixed data saved successfully.")