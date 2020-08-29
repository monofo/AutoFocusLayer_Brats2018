import argparse
import math
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

from config import TrainGlobalConfig
from dataset import BraTSDataset
from model import ModelBuilder
from utils import AverageMeter, PytorchTrainer

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

if not os.path.exists(TrainGlobalConfig.base_dir):
    os.makedirs("result/" + TrainGlobalConfig.base_dir, exist_ok=True)


def main():
    # import network architecture
    builder = ModelBuilder()
    model = builder.build_net(
            arch=TrainGlobalConfig.id, 
            num_input=TrainGlobalConfig.num_input, 
            num_classes=TrainGlobalConfig.num_classes, 
            num_branches=TrainGlobalConfig.num_branches,
            padding_list=TrainGlobalConfig.padding_list, 
            dilation_list=TrainGlobalConfig.dilation_list
            )

    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), TrainGlobalConfig.lr, alpha=0.9, eps=10**(-4), weight_decay=1e-4, momentum=0.6)
    criterion = nn.CrossEntropyLoss()

    train_list = os.path.join(TrainGlobalConfig.root_path, "train_0.txt")
    valid_list = os.path.join(TrainGlobalConfig.root_path, "valid_0.txt")

    train_dataset = BraTSDataset(
        list_file = train_list,
        root = TrainGlobalConfig.root_path,
        crop_size=TrainGlobalConfig.crop_size,
        num_input=TrainGlobalConfig.num_input
    )

    valid_dataset = BraTSDataset(
        list_file = valid_list,
        root = TrainGlobalConfig.root_path,
        crop_size=TrainGlobalConfig.crop_size,
        num_input=TrainGlobalConfig.num_input
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=TrainGlobalConfig.batch_size, 
        shuffle=True, 
        num_workers=TrainGlobalConfig.num_workers, 
        pin_memory=True
    )    

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=TrainGlobalConfig.batch_size, 
        shuffle=False, 
        num_workers=TrainGlobalConfig.num_workers, 
        pin_memory=True
    )
    
    trainer = PytorchTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=TrainGlobalConfig,
    )

    trainer.fit(train_loader, valid_loader)

if __name__ == '__main__':
    main()
