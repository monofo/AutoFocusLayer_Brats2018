import argparse
import math
import os

import nibabel as nib
import numpy as np
import torch


from config import TrainGlobalConfig
from dataset import BraTSDataset
from model import ModelBuilder
import SimpleITK as sitk 
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

# save prediction results in the format of online submission
def visualize_result(name, pred, args):
    # _, name1, _, name2 = name.split("/")
    # _, _, _, _, _, name3, _ = name2.split(".")
    print(name)
    pred = sitk.GetImageFromArray(pred)
    sitk.WriteImage(pred, f"./result/{TrainGlobalConfig.base_dir}/" + "/VSD"+"."+ name[0] +'.mha')
    
                
def accuracy(pred, mask, label):
    # columns in score is (# pred, # label, pred and label)
    score = np.zeros([3,3])

    # compute Enhance score (label==4) in the first line
    score[0,0] = np.count_nonzero(pred * mask ==3)
    score[0,1] = np.count_nonzero(label == 3)
    score[0,2] = np.count_nonzero(pred * mask * label == 9)
    
    pred[pred > 2] = 1
    label[label > 2] = 1
    score[1,0] = np.count_nonzero(pred * mask == 1)
    score[1,1] = np.count_nonzero(label == 1)
    score[1,2] = np.count_nonzero(pred * mask * label == 1)
    
    # compute Whole score (all labels) in the third line
    pred[pred > 1] = 1
    label[label > 1] = 1
    score[2,0] = np.count_nonzero(pred * mask == 1)
    score[2,1] = np.count_nonzero(label == 1)
    score[2,2] = np.count_nonzero(pred * mask * label == 1)
    return score

    
def test(test_loader, model, args):   
    # switch to evaluate mode
    model.eval()
    test_file = open(os.path.join(TrainGlobalConfig.root_path, "valid_0.txt"), 'r')
    test_dir = test_file.readlines()
    # initialization      
    num_ignore = 0
    # num_images = int(len(test_dir)/args.num_input)
    num_images = len(test_dir)
    # columns in score is (# pred, # label, pred and label)
    # lines in score is (Enhance, Core, Whole)
    dice_score = np.zeros([num_images, 3]).astype(float)
    
    for i, sample in enumerate(test_loader):
        image = sample['images'].float().cuda()
        label = sample['labels'].long().cuda()
        mask = sample['mask'].long().cuda()
        name = sample['name']
        
        with torch.no_grad():      
            image = image.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            # The dimension of out should be in the dimension of B,C,H,W,D
            out = model(image)
            out = out.permute(0,2,3,4,1)
            out = torch.max(out, 4)[1]

            # start index corresponding to the model
            start = [14, 14, 14]
            center = out.size()[1:]
            prediction = torch.zeros(label.size())
            prediction[:, start[0]:start[0]+center[0], start[1]: start[1]+center[1], start[2]: start[2]+center[2]] = out
                          
            # make the prediction corresponding to the center part of the image
            prediction = prediction.contiguous().view(-1).cuda()            
            label = label.contiguous().view(-1)                  
            mask = mask.contiguous().view(-1)
        
        # compute the dice score
        score_per_image = accuracy(prediction.data.cpu().numpy(), mask.data.cpu().numpy(), label.data.cpu().numpy()) 

        if np.sum(score_per_image[0,:]) == 0 or np.sum(score_per_image[1,:]) == 0 or np.sum(score_per_image[2,:]) == 0:
            num_ignore += 1
            continue 
        # compute the Enhance, Core and Whole dice score
        dice_score_per = [2 * np.sum(score_per_image[k,2]) / (np.sum(score_per_image[k,0]) + np.sum(score_per_image[k,1])) for k in range(3)]   
        print('Image: %d, Enhance score: %.4f, Core score: %.4f, Whole score: %.4f' % (i, dice_score_per[0], dice_score_per[1], dice_score_per[2]))           
        
        dice_score[i, :] = dice_score_per

        if True:
            vis = out.data.cpu().numpy()[0]
            vis = np.swapaxes(vis, 0, 2).astype(dtype=np.uint8)
            visualize_result(name, vis, args)
        
    count_image = num_images - num_ignore
    dice_score = dice_score[:count_image,:]
    mean_dice = np.mean(dice_score, axis=0)
    std_dice = np.std(dice_score, axis=0)
    print('Evalution Done!')
    print('Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (mean_dice[0], mean_dice[1], mean_dice[2], np.mean(mean_dice)))
    print('Enhance std: %.4f, Core std: %.4f, Whole std: %.4f, Mean Std: %.4f' % (std_dice[0], std_dice[1], std_dice[2], np.mean(std_dice)))                         


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

    ch = torch.load(f"./result/{TrainGlobalConfig.base_dir}/last-checkpoint.bin")
    model.load_state_dict(fix_model_state_dict(ch["model_state_dict"]))

    valid_list = os.path.join(TrainGlobalConfig.root_path, "valid_0.txt")
    valid_dataset = BraTSDataset(
        list_file = valid_list,
        root = TrainGlobalConfig.root_path,
        phase="val",
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=TrainGlobalConfig.num_workers, 
        pin_memory=True
        )
    test(valid_loader, model, TrainGlobalConfig)      
    
if __name__ == '__main__':
    main()