import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import torch


similarity = np.load("./exp/pixel_distance_p1.npy")
nodes = similarity.shape[0]
l = torch.cat((torch.range(eps, 1, 0.05), torch.tensor([1])))
solve(nodes, similarity, interval_list = l, interval_num = 4)