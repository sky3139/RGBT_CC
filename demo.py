import torch
import os
import cv2
import argparse
from datasets.crowd import Crowd
from models.fusion import fusion_model
from utils.evaluation import eval_game, eval_relative
import numpy as np
from torchvision import transforms
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default='/home/u20/gitee/RGBTCrowdCounting/DroneRGBT/save',
                        help='training data directory')
parser.add_argument('--save-dir', default='./',
                        help='model directory')
parser.add_argument('--model', default='datasets/cc/best_model.pth'
                    , help='model name')

parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

if __name__ == '__main__':


    RGB_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.407, 0.389, 0.396],
            std=[0.241, 0.246, 0.242]),
    ])
    T_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.492, 0.168, 0.430],
            std=[0.317, 0.174, 0.191]),
    ])

    datasets = Crowd(os.path.join(args.data_dir, 'demo'), method='demo')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model = fusion_model()
    model.to(device)
    model_path = os.path.join(args.save_dir, args.model)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0

    for inputs, target, rgb_path,t_path in dataloader:
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            # print(outputs[0][0][0].cpu().numpy())
            output_img = outputs[0][0].cpu().numpy()
            t_img=target[0].cpu().numpy()
            target_num = target.sum().float()
            # print(t_img.shape,output_img.shape)
            cv2.imshow("output_img",output_img*255)
            cv2.imshow("t_img",t_img)
            output_num = outputs.cpu().data.sum()
            src_img=cv2.imread(rgb_path[0])
            src_timg=cv2.imread(t_path[0])
            print(target_num,output_num)
            cv2.imshow("a",src_img)
            cv2.imshow("t",src_timg)
            cv2.waitKey(0)

        # break
