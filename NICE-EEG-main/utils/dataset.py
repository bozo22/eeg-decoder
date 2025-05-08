# I want to create a dataset class that will load the EEG and image data from the dataset
# and return the data in a format that can be used by the model

import os
import numpy as np
import torch
import logging as l

from torch.utils.data import Dataset, DataLoader, Subset

def get_eeg_data(dir_path, use_debug_eeg=False):
    train_data = []
    test_data = []
    test_label = np.arange(200)

    train_data = np.load(os.path.join(dir_path, 'preprocessed_eeg_training.npy'), allow_pickle=True)
    train_data = train_data['preprocessed_eeg_data']
    # Average across repetitions
    train_data = np.mean(train_data, axis=1) # Shape: (total_nr_train_imgs x 1 x channels x 250)
    train_data = np.expand_dims(train_data, axis=1)

    if use_debug_eeg:
        l.debug("Using EEG features for 100 training images only")
        train_data = train_data[:100]

    test_data = np.load(os.path.join(dir_path, 'preprocessed_eeg_test.npy'), allow_pickle=True)
    test_data = test_data['preprocessed_eeg_data']
    # Average across repetitions
    test_data = np.mean(test_data, axis=1) # Shape: (total_nr_test_imgs x 1 x channels x 250)
    test_data = np.expand_dims(test_data, axis=1)

    return train_data, test_data, test_label

def get_image_data(img_data_path, use_debug_images=False):
    if use_debug_images:
        l.debug("Using image features randomly generated for 100 training images only")
        train_img_feature = np.random.randn(100, 257, 1024)
    else:
        train_img_feature = np.load(img_data_path + '_feature_maps_training.npy', allow_pickle=True)
    test_img_feature = np.load(img_data_path + '_feature_maps_test.npy', allow_pickle=True)

    train_img_feature = np.squeeze(train_img_feature)
    test_img_feature = np.squeeze(test_img_feature)

    return train_img_feature, test_img_feature

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

def get_dataloaders(base_eeg_data_path, image_data_path, subject_id, batch_size, num_workers, debug=False, large_image_features=False):
    """
    Create and return dataloaders for training, validation, and testing.
    
    Args:
        base_eeg_data_path (str): Path to the base EEG data
        image_data_path (str): Path to the image data
        subject_id (int): Subject ID (1-based)
        batch_size (int): Batch size
        num_workers (int): Number of workers
        debug (bool): Whether to use only a subset of the data in training for debugging
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, all_test_img_feature)
    """
    print("Start loading data...")
    # Get the data
    eeg_data_path = os.path.join(base_eeg_data_path, 'sub-' + format(subject_id, '02'))
    train_eeg, test_eeg, test_label = get_eeg_data(eeg_data_path, use_debug_eeg=debug)
    train_img_feature, test_img_feature = get_image_data(image_data_path, use_debug_images=debug and large_image_features)

    # Convert test data to tensors
    test_eeg = torch.from_numpy(test_eeg).type(torch.FloatTensor)
    all_test_img_feature = torch.from_numpy(test_img_feature).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)

    train_eeg, train_image, val_eeg, val_image = split_train_val(train_eeg, train_img_feature)

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
    val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
    test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=400,  # Fixed test batch size as in original code
        shuffle=False,
        num_workers=num_workers
    )
    print("Data loaded successfully")
    return train_loader, val_loader, test_loader, all_test_img_feature