# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on CIFAR-10."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.n_iters = 950001
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  """
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  """
  sampling.eps = 1e-3
  sampling.method = 'dpm_solver'
  sampling.dpm_solver_method = 'singlestep'
  sampling.dpm_solver_order = 3
  sampling.algorithm_type = 'dpmsolver'
  sampling.thresholding = False
  sampling.noise_removal = False
  sampling.steps = 10
  sampling.skip_type = 'logSNR'
  sampling.rtol = 0.05
  
  # data
  data = config.data
  data.centered = True
  evaluate = config.eval
  evaluate.t_tuples = (0.376, 0.476,0.626,0.776)
  evaluate.t_converge = (0,0,0,0,0)
  # model
  model = config.model
  model.name = 'ncsnpp_multistage'
  model.fourier_scale = 16
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.embedding_type = 'positional'
  model.init_scale = 0.0
  model.conv_size = 3
  model.en_nf = 64
  model.de_nfs = [208, 96, 144, 48, 64]
  model.stage_num = 5
  model.stage_interval = [
    [[0, 0.376]], [[0.376, 0.476]], [[0.476, 0.626]], [[0.626, 0.776]], [[0.776, 1]]
  ]
  model.group = 16
  return config
