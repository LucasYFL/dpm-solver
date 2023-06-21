import torch
import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nodes', type=int, default=21)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--root', type=str, default="/scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/interval_distance_exp/")
parser.add_argument('--distance_func', type=str, default="l1_distance")
parser.add_argument('--interval', nargs=2, type=float)
args = parser.parse_args()


eps = args.eps
root = args.root
l = torch.cat((torch.range(eps, 1, 0.01), torch.tensor([1])))

def calculated_distance_files(t, distancefunc, root):
    pkgs = os.listdir(root)
    distance = 0
    num = 0
    for pkg in pkgs:
        data_path = os.path.join(root, pkg, f"0_{t:.4f}.pth")
        if os.path.exists(data_path):
            datas = torch.load(data_path)
            distance += distancefunc(datas['xs'], datas['optimal_solutions'])
            num += datas['optimal_solutions'].shape[0]
    print(f"process {t:.4f}, total number {num}")
    return distance/num

# def pixel_distance(f1, f2):
#     num = f1.shape[0]
#     similar = torch.abs(f1 - f2) < (2 * args.pixel_threshold / 256)
#     similar = similar.reshape((num, -1))
#     pixel_num = similar.shape[1]
#     return (similar.to(torch.float32).sum(dim=1)/pixel_num).sum()

def l2_distance(f1, f2):
    num = f1.shape[0]
    similar = torch.pow(f1 - f2, 2)
    similar = similar.reshape((num, -1)).to(torch.float32)
    pixel_num = similar.shape[1]
    return (similar[torch.logical_not(similar.isnan())].mean()) * num

def l1_distance(f1, f2):
    num = f1.shape[0]
    similar = torch.abs(f1 - f2)
    similar = similar.reshape((num, -1))
    pixel_num = similar.shape[1]
    return (similar.to(torch.float32).mean(dim=1)).sum()

dis_sum = 0
for t in l:
    if t >= args.interval[0] and t < args.interval[1]:
        dis= calculated_distance_files(t, eval(args.distance_func), root)
        dis_sum += dis
    
print(f"The total distance for interval [{args.interval[0]:.4f}, {args.interval[1]:.4f}) is {dis_sum:.4f}")
