from dataset import get_dataset
import sde_lib 

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import torch
import logging
import os

import time
import torch.multiprocessing as mp

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def normal_distribution(x, y_batch, s, std, bias = torch.tensor([0])):
  bs = y_batch.shape[0]
  prob = torch.exp(-((torch.pow((x - s * y_batch), 2)).view(bs, -1).sum(dim=1)/std**2)/2 - bias)
  prob_y = prob.clone().view(-1, 1, 1, 1) * y_batch
  return prob.sum(dim=0, keepdim=True), prob_y.sum(dim=0, keepdim=True)

normal_distribution_compile = torch.compile(normal_distribution)

def get_exp_bias(x, y_batch, s, std):
  ## because exp() might return a very small number, we need a bias
  bs = y_batch.shape[0]
  return (-(((x - s * y_batch)**2).view(bs, -1).sum(dim=1)/std**2)/2).max()

def get_optimal_sol(batch, z, sde, t, dataloader, config):
    s, sigma = sde.transform_prob(torch.tensor([t]))
    std = s * sigma
    x = s * batch + s * sigma[:, None, None, None] * z
    prob_sum = 0.
    prob_y_sum = torch.zeros_like(x)
    y_sum = torch.zeros_like(x)
    exp_bias = -(torch.inf)
    # for y_batch, _ in dataloader:
    #   exp_bias = max(exp_bias, get_exp_bias(x, y_batch, s, std))
    exp_bias = get_exp_bias(x, batch[None, :], s, std)
    for y_batch, _ in dataloader:
      prob, prob_y = normal_distribution_compile(x, y_batch, s, std, exp_bias)
      prob_sum += prob
      prob_y_sum += prob_y
      y_sum += y_batch.sum(dim=0, keepdim=True)
    assert config.exp.loss_func in ["epsilon", "x0"]
    if config.exp.loss_func == "x0":
      optimal_solution = prob_y_sum/prob_sum
    elif config.exp.loss_func == "epsilon":
      optimal_solution = prob_y_sum/prob_sum
    return optimal_solution

def generate_sample(dataset, config, t_list, dataloader, sde):
    optimal_solution_total = []
    t_total = []
    for i in range(config.exp.sampling_num):
      optimal_solutions = []
      ts = []
      randn_idx = torch.randint(len(dataset), (1, ))
      batch = dataset[randn_idx][0]
      z = torch.randn_like(batch)
      for t_sample in t_list:
        t = time.time()
        optimal_solution = get_optimal_sol(batch, z, sde, t_sample, dataloader, config)
        if (optimal_solution.isnan().sum()) == 0:
          optimal_solutions.append(optimal_solution)
        else:
          print("Generate a nan sample")
          break
        t_end = time.time() - t
        logging.info(f"The {config.host_id} host finished {i + 1}th sampling at t = {t_sample} generation, with {t_end}s") 
      optimal_solutions = torch.concat(tuple(optimal_solutions))
      if optimal_solutions.shape[0] == t_list.shape[0]:
        optimal_solution_total.append(optimal_solutions[None, :])
      if not os.path.isdir(config.exp.save_dir):
        os.mkdir(config.exp.save_dir)
      host_pkg = os.path.join(config.exp.save_dir, str(config.host_id))
      if not os.path.isdir(host_pkg):
        os.mkdir(host_pkg)
      torch.save({
            'optimal_solutions': torch.concat(tuple(optimal_solution_total)),
            't_list': t_list,
            }, os.path.join(host_pkg, f"{i//config.exp.num_save}.pth"))


def main(argv):
  config = FLAGS.config
  dataset, _ = get_dataset(config)
  if config.sde.type.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.sde.beta_min, 
                        beta_max=config.sde.beta_max, 
                        N=config.sde.num_scales)
    eps = 1e-3
  elif config.sde.type.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.sde.beta_min, 
                           beta_max=config.sde.beta_max, 
                           N=config.sde.num_scales)
    eps = 1e-3
  elif config.sde.type.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.sde.sigma_min, 
                        sigma_max=config.sde.sigma_max, 
                        N=config.sde.num_scales)
    eps = 1e-5
  dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=config.data.batch_size, 
                                           shuffle=True, 
                                           num_workers=4)
  t_list = torch.cat((torch.range(eps, 1, 0.005), torch.tensor([1])))
  generate_sample(dataset, config, t_list, dataloader, sde)


if __name__ == "__main__":
  app.run(main)