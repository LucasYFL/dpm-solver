import ml_collections
import torch


def get_config():
  config = ml_collections.ConfigDict()

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.batch_size = 4096
  data.random_flip = False
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 3
  data.dataset_root = "/scratch/qingqu_root/qingqu1/huijiezh/dpm-solver/example_v2/score_sde_pytorch/dataset/"
  
  # sde
  config.sde = sde = ml_collections.ConfigDict()
  sde.type = "EDM"
  sde.sigma_min = 0.002
  sde.sigma_max = 80
  sde.sigma_data = 0.5

  # exp
  config.exp = exp = ml_collections.ConfigDict()
  exp.sampling_num = 200
  exp.loss_func = "epsilon"
  exp.num_save = 200
  exp.save_dir = "/scratch/qingqu_root/qingqu1/shared_data/multistage/interval_split_edm/"
  
  #host
  config.host_id = 1

  return config