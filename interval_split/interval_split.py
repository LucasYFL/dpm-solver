from dataset import get_dataset
import sde_lib 

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import torch
import tqdm
import time

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def normal_distribution(x, y_batch, std, bias = 0):
  bs = y_batch.shape[0]
  prob = torch.exp(-(((x - y_batch)**2).view(bs, -1).sum(dim=1).to(torch.float64)/std**2)/2 - bias)
  prob_y = prob.clone().view(-1, 1, 1, 1) * y_batch
  return prob.sum(dim=0, keepdim=True), prob_y.sum(dim=0, keepdim=True)

def get_exp_bias(x, y_batch, std):
  ## because exp() might return a very small number, we need a bias
  bs = y_batch.shape[0]
  return (-(((x - y_batch)**2).view(bs, -1).sum(dim=1)/std**2)/2).max()

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
                                           num_workers=8)
  
  for i in tqdm.tqdm(range(config.exp.sampling_num)):
    tt = time.time()
    randn_idx = torch.randint(len(dataset), (1, ))
    batch = dataset[randn_idx][0]
    t = torch.rand((1, )) * (sde.T - eps) + eps
    mean, std = sde.marginal_prob(batch, t)
    z = torch.randn_like(batch)
    x = mean + std[:, None, None, None]* z
    prob_sum = 0.
    prob_y_sum = torch.zeros_like(x)
    y_sum = torch.zeros_like(x)
    exp_bias = -(torch.inf)
    for y_batch, _ in dataloader:
      exp_bias = max(exp_bias, get_exp_bias(x, y_batch, std))
    for y_batch, _ in dataloader:
      y_batch = y_batch
      prob, prob_y = normal_distribution(x, y_batch, std, exp_bias)
      prob_sum += prob
      prob_y_sum += prob_y
      y_sum += y_batch.sum(dim=0, keepdim=True)
    optimal_solution = prob_y_sum/prob_sum
    y_mean = y_sum/len(dataset)
    print("time t: ", time.time() - tt)


if __name__ == "__main__":
  app.run(main)