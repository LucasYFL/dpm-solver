import torch
import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nodes', type=int, default=201)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--root', type=str, default="/scratch/qingqu_root/qingqu1/shared_data/multistage/interval_split_graph_exp_v2")
parser.add_argument('--distance_func', type=str, default="l2_distance")
parser.add_argument('--pixel_threshold', type=int, default=1)
parser.add_argument('--interval_num', type=int, default=3)
args = parser.parse_args()

nodes= args.nodes
eps = args.eps
root = args.root
l = torch.cat((torch.range(eps, 1, 0.005), torch.tensor([1])))

assert args.distance_func in ["l2_distance"]
exp_file_path = os.path.join(root, f"{args.distance_func}.npy")



def solve_tryall(nodes, similarity, interval_list = l, interval_num = 3, type = "mean"):
    def generate_graph(idx_ts, nodes):
        graph = np.zeros((nodes, nodes, len(idx_ts) - 1))
        for idx in range(len(idx_ts) - 1):
            graph[idx_ts[idx]:idx_ts[idx + 1], idx_ts[idx]:idx_ts[idx + 1], idx] = 1
        return graph
    
    def calculate_objective(similarity, connected_graph, type = type):
        object = 0
        for idx in range(connected_graph.shape[2]):
            if type == "mean":
                object += (similarity * connected_graph[:, :, idx]).sum() / connected_graph[:, :, idx].sum()
            elif type == "sum":
                object += (similarity * connected_graph[:, :, idx]).sum()
        return object
    
    objective_max = -torch.inf
    t_optimal_idx = None
    if interval_num == 2:
        idx_t1 = 0
        idx_t3 = nodes
        for idx_t2, t2 in enumerate(interval_list):
            if idx_t2 > idx_t1 and idx_t2 < idx_t3:
                connected_graph = generate_graph((idx_t1, idx_t2, idx_t3), nodes)
                objective = calculate_objective(similarity, connected_graph)
                if objective > objective_max:
                    t_optimal_idx = (idx_t1, idx_t2, idx_t3)
                    objective_max = objective
    
    elif interval_num == 3:
        idx_t1 = 0
        idx_t4 = nodes
        for idx_t2, t2 in enumerate(interval_list):
            for idx_t3, t3 in enumerate(interval_list):
                if idx_t2 > idx_t1 and idx_t3 > idx_t2 and idx_t4 > idx_t3:
                    connected_graph = generate_graph((idx_t1, idx_t2, idx_t3, idx_t4), nodes)
                    objective = calculate_objective(similarity, connected_graph)
                    if objective > objective_max:
                        t_optimal_idx = (idx_t1, idx_t2, idx_t3, idx_t4)
                        objective_max = objective        

    elif interval_num == 4:
        idx_t1 = 0
        idx_t5 = nodes
        for idx_t2, t2 in enumerate(interval_list):
            for idx_t3, t3 in enumerate(interval_list):
                for idx_t4, t4 in enumerate(interval_list):
                    if idx_t2 > idx_t1 and idx_t3 > idx_t2 and idx_t4 > idx_t3 and idx_t5 > idx_t4:
                        connected_graph = generate_graph((idx_t1, idx_t2, idx_t3, idx_t4, idx_t5), nodes)
                        objective = calculate_objective(similarity, connected_graph)
                        if objective > objective_max:
                            t_optimal_idx = (idx_t1, idx_t2, idx_t3, idx_t4, idx_t5)
                            objective_max = objective     
    elif interval_num == 5:
        idx_t1 = 0
        idx_t6 = nodes
        for idx_t2, t2 in enumerate(interval_list):
            for idx_t3, t3 in enumerate(interval_list):
                for idx_t4, t4 in enumerate(interval_list):
                    for idx_t5, t5 in enumerate(interval_list):
                        if idx_t2 > idx_t1 and idx_t3 > idx_t2 and idx_t4 > idx_t3 and idx_t5 > idx_t4 and idx_t6 > idx_t5:
                            connected_graph = generate_graph((idx_t1, idx_t2, idx_t3, idx_t4, idx_t5, idx_t6), nodes)
                            objective = calculate_objective(similarity, connected_graph)
                            if objective > objective_max:
                                t_optimal_idx = (idx_t1, idx_t2, idx_t3, idx_t4, idx_t5, idx_t6)
                                objective_max = objective    
                                   
    idx = 1
    print(f"The {idx}th is from [0, {interval_list[t_optimal_idx [idx]- 1] + 0.025})")
    for idx in range(2, interval_num):
        print(f"The {idx}th is from [{interval_list[t_optimal_idx [idx-1] - 1] + 0.025}, {interval_list[t_optimal_idx [idx] - 1] + 0.025})")
    idx = interval_num
    print(f"The {idx}th is from [{interval_list[t_optimal_idx [idx - 1]- 1] + 0.025}, 1)")                
    return t_optimal_idx

def calculated_distance_files(distancefunc, root):
    pkgs = os.listdir(root)
    sample = []
    for pkg in pkgs:
        data_path = os.path.join(root, pkg, f"0.pth")
        if os.path.exists(data_path):
            datas = torch.load(data_path)
            sample.append(datas['optimal_solutions'])
    sample = torch.cat(sample, dim=0)
    total_num = sample.shape[0]
    print(f"There are {total_num} sample in total")
    similarity = torch.zeros(l.shape[0], l.shape[0])
    for idx_ti, ti in enumerate(l):
        for idx_tj, tj in enumerate(l):
            print(ti, tj)
            if idx_ti != idx_tj:
                similarity[idx_ti, idx_tj] = distancefunc(sample[:, idx_ti], sample[:, idx_tj])
            else:
                similarity[idx_ti, idx_tj] = -0.0
    return similarity

def l2_distance(f1, f2):
    num = f1.shape[0]
    f1_copy = f1.reshape(num, -1)
    f2_copy = f2.reshape(num, -1)
    similar = torch.pow(f1_copy - f2_copy, 2)
    similar = similar.reshape((num, -1)).to(torch.float32)
    pixel_num = similar.shape[1]
    return -(similar[torch.logical_not(similar.isnan())].mean())

def vis_similarity(similarity):
    string = ''
    for idx1, t1 in enumerate(l):
        for idx2, t2 in enumerate(l):    
            string += f"{similarity[idx1, idx2]:.2f} "
        string += "\n"
    print(string)



if os.path.exists(exp_file_path):
    similarity = np.load(exp_file_path)
else:
    similarity = calculated_distance_files(eval(args.distance_func), root)
    np.save(exp_file_path, similarity)
    
# vis_similarity(similarity)
# solve(nodes, similarity, interval_list = l, interval_num = 3)
solve_tryall(nodes, similarity, interval_list = l, interval_num = args.interval_num, type = "sum")
            
            
        