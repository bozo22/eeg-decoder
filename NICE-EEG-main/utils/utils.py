import os
import random
import numpy as np
import torch
import torch.nn as nn
import wandb

def seed_experiments(seed):
    print(f'Seeding experiments with seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, chekpoint_path, run_name):
    model_name = model.module.__class__.__name__ if isinstance(model, nn.DataParallel) else model.__class__.__name__
    model_state_dict = model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict()
    save_path = os.path.join(chekpoint_path, f"{model_name}-{run_name}.pth")
    torch.save(model_state_dict, save_path)
    print(f"Model {model_name} saved to {save_path}")

def load_model(model, chekpoint_path, model_idx, device=None): 
    model_name = model.module.__class__.__name__ if isinstance(model, nn.DataParallel) else model.__class__.__name__
    save_path = os.path.join(chekpoint_path, f"{model_name}-{model_idx}.pth")
    model_state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(model_state_dict)
    print(f"Model {model_name} loaded from {save_path}")
    return model


def wandb_login(disable_wandb: bool):
    if disable_wandb:
        print("!! WandB disabled !!")
        return
    else:
        try:
            with open("wandb.password", "rt") as f:
                pw = f.readline().strip()
                os.environ["WANDB_API_KEY"] = pw
                wandb.login()
        except FileNotFoundError:
            raise FileNotFoundError("File wandb.password was not found in the project root. Either add it or disable wandb by running --disable_wandb")