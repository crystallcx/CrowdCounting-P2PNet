import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
import glob

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='./vis/test',  #<--------------------------
                        help='path where to save')
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth',
                        help='path where the trained weights saved')
    parser.add_argument('--input_dir', default='../database/test',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #saves images with predicted points in output dir, and all predicted counts in text file.
    dir_folder = os.path.basename(os.path.normpath(args.input_dir))
    with open(os.path.join(args.output_dir, 'p2p_'+dir_folder +'_pr.txt'),"w") as listfile:
        for imgname in glob.glob(args.input_dir + "/*.png"): 
            # set your image path here
            img_path = imgname
            # load the images
            img_raw = Image.open(img_path).convert('RGB')
            # round the size
            width, height = img_raw.size
            new_width = width // 128 * 128
            new_height = height // 128 * 128
            img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
            # pre-proccessing
            img = transform(img_raw)

            samples = torch.Tensor(img).unsqueeze(0)
            samples = samples.to(device)
            # run inference
            outputs = model(samples)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

            outputs_points = outputs['pred_points'][0]

            threshold = 0.5
            # filter the predictions
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > threshold).sum())

            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

            outputs_points = outputs['pred_points'][0]
            # draw the predictions
            size = 2
            img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
            for p in points:
                img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

            filename = os.path.basename(imgname).split(".")[0]        

            # save the visualized image
            cv2.imwrite(os.path.join(args.output_dir, '{}pred{}.jpg'.format(filename,predict_cnt)), img_to_draw)

            # save p2p_coords in text file
            # txt_out = os.path.join(args.output_dir, '{}pred{}.txt'.format(filename,predict_cnt))
            txt_out = os.path.join(args.output_dir, '{}.txt'.format(filename))

            with open(txt_out, 'w') as f:
                for p in points:
                    # f.write(str(p[0]) +" "+ str(p[1])) #(f"{line}\n")
                    f.write("{} {}\n".format(str((p[0])/width), str((p[1])/ height)))

            listfile.write(filename + ".png" + " " + "{}".format(predict_cnt) + '\n')  
            #ground_truths.append([filename, predict_cnt])
            #val_outputs.append([filename,img_to_draw,predict_cnt])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)