import os
import random
import numpy as np
import torch
import torch.nn as nn

def seed_experiments(seed):
    print(f'Seeding experiments with seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, chekpoint_path, model_idx):
    model_name = model.module.__class__.__name__ if isinstance(model, nn.DataParallel) else model.__class__.__name__
    model_state_dict = model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict()
    save_path = os.path.join(chekpoint_path, f"{model_name}-{model_idx}.pth")
    torch.save(model_state_dict, save_path)
    print(f"Model {model_name} saved to {save_path}")

def load_model(model, chekpoint_path, model_idx):
    model_name = model.module.__class__.__name__ if isinstance(model, nn.DataParallel) else model.__class__.__name__
    save_path = os.path.join(chekpoint_path, f"{model_name}-{model_idx}.pth")
    model_state_dict = torch.load(save_path)
    model.load_state_dict(model_state_dict)
    print(f"Model {model_name} loaded from {save_path}")
    return model
