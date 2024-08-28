'''
    Scripts to compute latency and fps of a model
'''

import os
import time
import argparse

import torch 
from lib.MSCSFNet import Network
import torchvision.transforms as transformers 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FLOPs of a model.')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    parser.add_argument('--num-frames', type=int, default=32, help='temporal clip length.')
    parser.add_argument('--input-size', type=int, default=352,
                        help='size of the input image size. default is 224')
    parser.add_argument('--num-runs', type=int, default=1000,
                        help='number of runs to compute average forward timing. default is 105')
    parser.add_argument('--num-warmup-runs', type=int, default=5,
                        help='number of warmup runs to avoid initial slow speed. default is 5')

    args = parser.parse_args()

    model = Network()
    model.eval()
    model.cuda()
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, args.input_size, args.input_size)).cuda()
    print('Model is loaded, start forwarding.')


#Measure performance
    with torch.no_grad():
        for i in range(args.num_runs):
            if i == args.num_warmup_runs:
                start_time = time.time()
            pred = model(input_tensor)

    end_time = time.time()
    total_forward = end_time - start_time
    print('Total forward time is %4.2f seconds' % total_forward)

    actual_num_runs = args.num_runs - args.num_warmup_runs
    latency = total_forward / actual_num_runs
    # fps = (cfg.CONFIG.DATA.CLIP_LEN * cfg.CONFIG.DATA.FRAME_RATE) * actual_num_runs / total_forward

    print("FPS: " "; Latency: ", latency)
