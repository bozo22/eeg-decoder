# I want to create a dataset class that will load the EEG and image data from the dataset
# and return the data in a format that can be used by the model

import os
import numpy as np
import torch
import logging as l

from torch.utils.data import DataLoader


VALIDATION_NR_CONDITIONS = 200  # Same as for test set
SAMPLES_PER_CONDITION = 10
TEST_VAL_BATCH_SIZE = 200


def calculate_aggregations(data, eeg_denoiser=False):
    """
    Calculate various aggregations for the EEG data.
    Args:
        data (numpy.ndarray): EEG data of shape (num_samples, num_channels, num_timepoints)
        eeg_denoiser (bool): Whether EEG denoiser is used - if False, only mean is calculated. Default is False.
    Returns:
        numpy.ndarray: Aggregated EEG data
    """

    batch, _, channels, timepoints = data.shape
    n_aggr = 9 if eeg_denoiser else 1
    aggregations = np.empty((batch, n_aggr, channels, timepoints), dtype=data.dtype)
    aggregations[:, 0] = np.mean(data, axis=1)
    if eeg_denoiser:
        aggregations[:, 1] = np.std(data, axis=1)
        aggregations[:, 2] = np.min(data, axis=1)
        aggregations[:, 3] = np.max(data, axis=1)
        aggregations[:, 4] = np.median(data, axis=1)
        aggregations[:, 5] = np.percentile(data, 5, axis=1)
        aggregations[:, 6] = np.percentile(data, 95, axis=1)
        aggregations[:, 7] = np.percentile(data, 10, axis=1)
        aggregations[:, 8] = np.percentile(data, 90, axis=1)

    return aggregations


def get_eeg_data(dir_path, data_mode=None, eeg_denoiser=False):
    train_data = []
    test_data = []
    test_label = np.arange(200)

    train_data = np.load(
        os.path.join(dir_path, "preprocessed_eeg_training.npy"), allow_pickle=True
    )
    train_data = train_data["preprocessed_eeg_data"]

    if data_mode == "debug":
        print(">>> Using EEG features for 100 training samples only")
        train_data = train_data[:100]
    elif data_mode == "small":
        print(">>> Using EEG features for 25 percent of the training samples")
        train_data = train_data[: int(len(train_data) * 0.25)]

    # Calculate the aggregations
    train_data = calculate_aggregations(train_data, eeg_denoiser)

    test_data = np.load(
        os.path.join(dir_path, "preprocessed_eeg_test.npy"), allow_pickle=True
    )
    test_data = test_data["preprocessed_eeg_data"]
    test_data = calculate_aggregations(test_data, eeg_denoiser)

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


def split_train_val(eeg_data, img_data, per_condition=False, split_ratio=0.05):
    """
    Split data into train/val sets.
    Args:
        per_condition: bool -
            If True, split by conditions, keeping all samples of each condition together.
            If False, split by indices, mixing samples from all conditions.
        split_ratio: float - ratio of data to use for validation if per_condition is False
    """
    # Align in case of runs with smaller nr of EEG samples
    img_data = img_data[: len(eeg_data)]
    total_samples = len(eeg_data)
    # Split by conditions
    if per_condition:
        # Randomly select conditions for validation
        nr_conditions = total_samples // SAMPLES_PER_CONDITION
        val_nr_conditions = (
            VALIDATION_NR_CONDITIONS
            if VALIDATION_NR_CONDITIONS < nr_conditions
            else nr_conditions // 4
        )
        val_conditions = np.random.choice(
            nr_conditions, val_nr_conditions, replace=False
        )
        # Create mask for validation
        val_eeg, val_proto = [], []
        val_mask = np.zeros(total_samples, dtype=bool)

        for condId in val_conditions:
            # all 10 indices belonging to this condition
            idx_start = condId * SAMPLES_PER_CONDITION
            cond_idx = np.arange(idx_start, idx_start + SAMPLES_PER_CONDITION)

            # pick ONE held-out exemplar for the EEG query
            held_idx = np.random.choice(cond_idx)
            val_eeg.append(eeg_data[held_idx])

            # prototype = mean of the remaining 9 image embeddings
            other_idx = cond_idx[cond_idx != held_idx]
            proto_emb = img_data[other_idx].mean(axis=0)
            val_proto.append(proto_emb)
            val_mask[cond_idx] = True  # exclude all 10 from training

        val_eeg = np.stack(val_eeg)
        val_img = np.stack(val_proto)
        print(
            f"""Split train/val per condition: {nr_conditions - val_nr_conditions} training conditions 
              and {val_nr_conditions} validation conditions"""
        )

    # Split by indices
    else:
        # Shuffle the data
        shuffle_idx = np.random.permutation(total_samples)
        eeg_data = eeg_data[shuffle_idx]
        img_data = img_data[shuffle_idx]

        # Create mask for validation
        val_size = int(total_samples * split_ratio)
        val_mask = np.zeros(total_samples, dtype=bool)
        val_mask[:val_size] = True
        val_eeg = eeg_data[val_mask]
        val_img = img_data[val_mask]
        print(
            f"Split train/val by split ratio: {total_samples - val_size} training samples and {val_size} validation samples"
        )

    val_eeg = torch.from_numpy(val_eeg).type(torch.FloatTensor)
    val_image = torch.from_numpy(val_img).type(torch.FloatTensor)

    train_eeg = torch.from_numpy(eeg_data[~val_mask]).type(torch.FloatTensor)
    train_image = torch.from_numpy(img_data[~val_mask]).type(torch.FloatTensor)
    return train_eeg, train_image, val_eeg, val_image


def mixup(mixup_alpha, eeg, img_features, device):
    if type(eeg) is not torch.Tensor:
        eeg = torch.from_numpy(eeg)
        img_features = torch.from_numpy(img_features)
        eeg = eeg.type(torch.FloatTensor).to(device)
        img_features = img_features.type(torch.FloatTensor).to(device)

    batch_size = eeg.shape[0]
    index = torch.randperm(batch_size).to(device)
    lam = np.random.beta(mixup_alpha, mixup_alpha)

    # Mix both modalities consistently
    mixed_eeg = lam * eeg + (1 - lam) * eeg[index]
    mixed_img_features = lam * img_features + (1 - lam) * img_features[index]

    return mixed_eeg, mixed_img_features


def get_dataloaders(
    base_eeg_data_path,
    image_data_path,
    dnn,
    subject_id,
    batch_size,
    mixup_in_class=False,
    use_mixup=False,
    mixup_val_set_size=740,
    n_ways=[2, 5, 10],
    eeg_denoiser=False,
    dataset_mode=None,
    val_set_per_condition=False,
):
    """
    Create and return dataloaders for training, validation, and testing.

    Args:
        base_eeg_data_path (str): Path to the base EEG data
        image_data_path (str): Path to the image data
        subject_id (int): Subject ID (1-based)
        batch_size (int): Batch size
        n_ways (list): List of n-way classification test datasets (in addition to the full 200-way test set)
        eeg_denoiser (bool): Whether to use EEG denoiser
        dataset_mode (str): Dataset mode (None, "small", "debug") - how much data to use
        val_set_per_condition (bool): If True, split by conditions, keeping all samples of each condition together.
            If False, split by indices, mixing samples from all conditions.
    Returns:
        tuple: (train_loader, val_loader, test_loader, test_centers, test_n_way_loaders, test_n_way_centers)
    """
    print("Start loading data...")

    # Get the data
    eeg_data_path = os.path.join(base_eeg_data_path, "sub-" + format(subject_id, "02"))
    train_eeg, test_eeg, test_label = get_eeg_data(
        eeg_data_path, data_mode=dataset_mode, eeg_denoiser=eeg_denoiser
    )
    train_img_feature, test_centers = get_image_data(image_data_path, dnn)
    # Convert test data to tensors
    test_eeg = torch.from_numpy(test_eeg).type(torch.FloatTensor)
    test_centers = torch.from_numpy(test_centers).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)
    test_n_way_eeg, test_n_way_label, test_n_way_centers = [], [], []
    # Add n_way test data
    for n_way in n_ways:
        assert n_way < 200, "each n_way should be less than 200"
        test_n_way_eeg.append(test_eeg[:n_way])
        test_n_way_label.append(test_label[:n_way])
        test_n_way_centers.append(test_centers[:n_way])

    # Split train/val
    if mixup_in_class or use_mixup:
        val_eeg = torch.from_numpy(train_eeg[:mixup_val_set_size]).type(
            torch.FloatTensor
        )
        val_image = torch.from_numpy(train_img_feature[:mixup_val_set_size]).type(
            torch.FloatTensor
        )
        train_eeg = torch.from_numpy(train_eeg[mixup_val_set_size:]).type(
            torch.FloatTensor
        )
        train_image = torch.from_numpy(train_img_feature[mixup_val_set_size:]).type(
            torch.FloatTensor
        )
    else:
        train_eeg, train_image, val_eeg, val_image = split_train_val(
            train_eeg, train_img_feature, per_condition=val_set_per_condition
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
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=TEST_VAL_BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=TEST_VAL_BATCH_SIZE,  # Fixed test batch size as in original code
        shuffle=False,
    )
    # Create n-way test dataloaders
    test_n_way_loaders = []
    for n_way_dataset in test_n_way_datasets:
        n_way_loader = DataLoader(
            dataset=n_way_dataset,
            batch_size=TEST_VAL_BATCH_SIZE,  # Fixed test batch size as in original code
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
        batch_size=TEST_VAL_BATCH_SIZE,  # Fixed test batch size as in original code
        shuffle=False,
    )

    print("Test data loaded successfully")
    return test_loader, test_centers
