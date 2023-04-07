import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 256
  training.n_iters = 1300001
  training.snapshot_freq = 10000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.t0 = 0.0
  training.t1 = 0.0
  training.objective_weight = "eps"
  training.dt = 0.1
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 8
  evaluate.end_ckpt = 8
  evaluate.ckpt_interval = 5
  evaluate.batch_size = 2048
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  #added
  evaluate.t = 0.0
  evaluate.t_tuples = (0.259,0.676) #intervals
  evaluate.t_converge = (0,0,0) #num models 1 more than intervals, if the model is converged.
  evaluate.objectives = ("eps","eps","eps")
  evaluate.converge_epoch = 40 # epoch to be the baseline
  evaluate.compare_step = 0 # timestep to be compared
  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  return config