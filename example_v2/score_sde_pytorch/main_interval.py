# coding=utf-8

"""Training and evaluation"""
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
import tensorflow as tf
import gc
import io
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)

config_flags.DEFINE_config_file(
  "config1", None, "Training configuration.", lock_config=True)
config_flags.DEFINE_config_file(
  "config2", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("m1", None, "Model 1 directory.")
flags.DEFINE_string("m2", None, "Model 2  directory.")
flags.DEFINE_string("m3", None, "Model 3 directory.")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", 'm1','m2'])

tf.config.experimental.set_visible_devices([], "GPU")
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
local_rank = int(os.environ["LOCAL_RANK"])
total_rank = int(os.environ['LOCAL_WORLD_SIZE'])
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl')
def evaluate(config,
             workdir,m1,m2,m3=None,config1=None,config2=None,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for models
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  train_ds,  _ = datasets.get_dataset(config)
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_models = []
  optimizers = []
  emas = []
  states = []
  mdir = (m1,m2,m3)
  checkpoint_dirs = []
  objectives = []
  logging.info(config.eval.t_tuples)
  configs = (config,)*3 if config1 is None else (config,config1,config2)
  
  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  for i in range(len(config.eval.t_tuples)+1):
    s = mutils.create_model(configs[i],local_rank)
    score_models.append(s)
    opt = losses.get_optimizer(config, s.parameters())
    optimizers.append(opt)
    ema = ExponentialMovingAverage(s.parameters(), decay=config.model.ema_rate)
    emas.append(ema)
    states.append(dict(optimizer=opt, model=s, ema=ema, step=0))
    checkpoint_dirs.append( os.path.join(workdir,mdir[i], "checkpoints"))
    objectives.append(losses.get_objective_schedule(sde,config.eval.objectives[i],config.training.dt))
  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting,t=float(config.eval.t))


  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds_bpd,  _ = datasets.get_dataset(config)
  """ 
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")
  """
  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, local_rank)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1, config.eval.ckpt_interval):
    for i in range(len(checkpoint_dirs)):
      if config.eval.t_converge[i]:
        logging.info("{} is converged model".format(i))
        ckpt_path = os.path.join(checkpoint_dirs[i], "checkpoint_{}.pth".format(config.eval.converge_epoch))
      else:
        ckpt_path = os.path.join(checkpoint_dirs[i], "checkpoint_{}.pth".format(ckpt))
      logging.info(ckpt_path)
      states[i] =restore_checkpoint(ckpt_path, states[i], device=f"{config.device}:{local_rank}")
      emas[i].copy_to(score_models[i].parameters())
    
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    #!!!!!NO loss and bpd for interval exp!!!!!
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      logging.info(eval_dir)
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        if local_rank == 0:
          logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}_host_{local_rank}")
        os.makedirs(this_sample_dir, exist_ok=True)

        samples_raw, n = sampling_fn(score_models, config.eval.t_tuples,objectives)

        samples = torch.clip(samples_raw.permute(0, 2, 3, 1) * 255., 0, 255).to(torch.uint8)
        ## center the sample when calculating fid
        #samples_fid = (torch.clone(samples).permute(0, 3, 1, 2).to(torch.float32) / 255) * 2 - 1
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels)).cpu().numpy()
        # Write samples to disk or Google Cloud Storage
        with open(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        if r == 0:
          nrow = int(np.sqrt(samples_raw.shape[0]))
          image_grid = make_grid(samples_raw, nrow, padding=2)
          with open(
              os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
            save_image(image_grid, fout)

def main(argv):
  global config_fewer
  config_fewer = FLAGS.config
  config1 = FLAGS.config1
  # Run the evaluation pipeline
  evaluate(FLAGS.config, FLAGS.workdir,FLAGS.m1,FLAGS.m2,FLAGS.m3,config1,FLAGS.config2, FLAGS.eval_folder)
 
if __name__ == "__main__":
  app.run(main)