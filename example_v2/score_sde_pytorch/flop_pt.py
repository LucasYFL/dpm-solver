import gc
import io
import os
import time
import sys
sys.path.append("../")
# Keep the import below for registering all model definitions
from models import iUNet,ncsnpp,ddpm
import losses
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import torch

from configs.vp import cifar10_MEME as configs
# from configs.vp import cifar10_ncsnpp_multistage_deep_continuous_v7 as configs
from ptflops import get_model_complexity_info

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config=configs.get_config()
config.eval.batch_size = 1
config.device = torch.device('cpu')
config.model.stage_num=3
config.model.nf=192

sampling_eps = 1e-3
# Build data pipeline
train_ds, eval_ds = datasets.get_dataset(config,
                                            uniform_dequantization=config.data.uniform_dequantization,
                                            evaluation=True)

# Create data normalizer and its inverse
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)
optimizer = losses.get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

sampling_eps = 1e-3
sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)

batch = torch.rand(sampling_shape).to(config.device)
batch = scaler(batch)
lst_steps=torch.tensor([0.1])*1000
t = lst_steps.to('cpu')#.repeat(batch.shape[0])
score_model = score_model.to('cpu')
print(score_model(batch,t))
class mfn(torch.nn.Module):
    def __init__(self,m):
        super().__init__()
        self.m = m
    def forward(self,x):
        return self.m(x,t)
macs, params = get_model_complexity_info(mfn(score_model), sampling_shape[1:], as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))