# I want to create a dataset class that will load the EEG and image data from the dataset
# and return the data in a format that can be used by the model

import os
import numpy as np
import torch
import logging as l

from torch.utils.data import DataLoader


def get_eeg_data(dir_path, data_mode=None):
    train_data = []
    test_data = []
    test_label = np.arange(200)

    train_data = np.load(
        os.path.join(dir_path, "preprocessed_eeg_training.npy"), allow_pickle=True
    )
    train_data = train_data["preprocessed_eeg_data"]
    # Average across repetitions
    train_data = np.mean(
        train_data, axis=1
    )  # Shape: (total_nr_train_imgs x 1 x channels x 250)
    train_data = np.expand_dims(train_data, axis=1)

    if data_mode == "debug":
        print(">>> Using EEG features for 100 training images only")
        train_data = train_data[:100]
    elif data_mode == "small":
        print(">>> Using EEG features for 25 percent of the training images")
        train_data = train_data[:int(len(train_data) * 0.25)]

    test_data = np.load(
        os.path.join(dir_path, "preprocessed_eeg_test.npy"), allow_pickle=True
    )
    test_data = test_data["preprocessed_eeg_data"]
    # Average across repetitions
    test_data = np.mean(
        test_data, axis=1
    )  # Shape: (total_nr_test_imgs x 1 x channels x 250)
    test_data = np.expand_dims(test_data, axis=1)

    return train_data, test_data, test_label


def get_image_data(img_data_path, dnn):

    image_features_path = os.path.join(img_data_path, dnn)
    train_img_feature = np.load(
        image_features_path + "_feature_maps_training.npy", allow_pickle=True
    )
    train_img_feature = np.squeeze(train_img_feature)

    test_centers_path = os.path.join(img_data_path, "center_" + dnn + ".npy")
    test_centers = np.load(test_centers_path, allow_pickle=True)

    return train_img_feature, test_centers


def split_train_val(eeg_data, img_data, split_ratio=0.05):
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(eeg_data))
    eeg_data = eeg_data[shuffle_idx]
    img_data = img_data[shuffle_idx]

    val_size = int(len(eeg_data) * split_ratio)
    # Split into train/val
    val_eeg = torch.from_numpy(eeg_data[:val_size]).type(torch.FloatTensor)
    val_image = torch.from_numpy(img_data[:val_size]).type(torch.FloatTensor)

    train_eeg = torch.from_numpy(eeg_data[val_size:]).type(torch.FloatTensor)
    train_image = torch.from_numpy(img_data[val_size:]).type(torch.FloatTensor)

    return train_eeg, train_image, val_eeg, val_image


def get_dataloaders(
    base_eeg_data_path,
    image_data_path,
    dnn,
    subject_id,
    batch_size,
    n_ways=[2, 5, 10],
    dataset_mode=None,
):
    """
    Create and return dataloaders for training, validation, and testing.

    Args:
        base_eeg_data_path (str): Path to the base EEG data
        image_data_path (str): Path to the image data
        subject_id (int): Subject ID (1-based)
        batch_size (int): Batch size
        n_ways (list): List of n-way classification test datasets (in addition to the full 200-way test set)
        dataset_mode (str): Dataset mode (None, "small", "debug") - how much data to use
    Returns:
        tuple: (train_loader, val_loader, test_loader, test_centers, test_n_way_loaders, test_n_way_centers)
    """
    print("Start loading data...")
    # Get the data
    eeg_data_path = os.path.join(base_eeg_data_path, "sub-" + format(subject_id, "02"))
    train_eeg, test_eeg, test_label = get_eeg_data(eeg_data_path, data_mode=dataset_mode)
    train_img_feature, test_centers = get_image_data(image_data_path, dnn)
    test_n_way_eeg, test_n_way_label = [], []

    # Add n_way test data
    for n_way in n_ways:
        assert n_way < 200, "each n_way should be less than 200"
        test_n_way_eeg.append(test_eeg[:n_way])
        test_n_way_label.append(test_label[:n_way])

    # Convert n_way test data to tensors
    test_n_way_eeg = [
        torch.from_numpy(n_way_eeg).type(torch.FloatTensor)
        for n_way_eeg in test_n_way_eeg
    ]
    test_n_way_label = [
        torch.from_numpy(n_way_label).type(torch.LongTensor)
        for n_way_label in test_n_way_label
    ]
    test_n_way_centers = [
        torch.from_numpy(test_centers[:n_way]).type(torch.FloatTensor)
        for n_way in n_ways
    ]

    # Convert test data to tensors
    test_eeg = torch.from_numpy(test_eeg).type(torch.FloatTensor)
    test_centers = torch.from_numpy(test_centers).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)

    train_eeg, train_image, val_eeg, val_image = split_train_val(
        train_eeg, train_img_feature
    )

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
    val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
    test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)

    # Create n-way test datasets
    test_n_way_datasets = []
    for n_way_eeg, n_way_label in zip(test_n_way_eeg, test_n_way_label):
        test_n_way_datasets.append(
            torch.utils.data.TensorDataset(n_way_eeg, n_way_label)
        )

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=400,  # Fixed test batch size as in original code
        shuffle=False,
    )

    # Create n-way test dataloaders
    test_n_way_loaders = []
    for n_way_dataset in test_n_way_datasets:
        n_way_loader = DataLoader(
            dataset=n_way_dataset,
            batch_size=400,  # Fixed test batch size as in original code
            shuffle=False,
        )
        test_n_way_loaders.append(n_way_loader)

    print("Data loaded successfully")
    return (
        train_loader,
        val_loader,
        test_loader,
        test_centers,
        test_n_way_loaders,
        test_n_way_centers,
    )


def get_test_dataloader(eeg_data_path, img_data_path, dnn, subject_id):

    print("Start loading test data...")

    # EEG test data
    eeg_data_path = os.path.join(eeg_data_path, "sub-" + format(subject_id, "02"))
    eeg_test_data = np.load(
        os.path.join(eeg_data_path, "preprocessed_eeg_test.npy"), allow_pickle=True
    )
    eeg_test_data = eeg_test_data["preprocessed_eeg_data"]
    # Average across repetitions
    eeg_test_data = np.mean(
        eeg_test_data, axis=1
    )  # Shape: (total_nr_test_imgs x 1 x channels x 250)
    eeg_test_data = np.expand_dims(eeg_test_data, axis=1)
    test_eeg = torch.from_numpy(eeg_test_data).type(torch.FloatTensor)

    # Image test data
    test_centers_path = os.path.join(img_data_path, "center_" + dnn + ".npy")
    test_centers = np.load(test_centers_path, allow_pickle=True)
    test_centers = torch.from_numpy(test_centers).type(torch.FloatTensor)

    test_label = np.arange(200)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)

    test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=400,  # Fixed test batch size as in original code
        shuffle=False,
    )

    print("Test data loaded successfully")

    return test_loader, test_centers
