#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH --time=04-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:2
#SBATCH --output=newinterval0.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=4
#SBATCH --cpus-per-task=8
echo $EXP_FEWER_STEPS
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm2
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
#python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm_I1 --mode train --config.training.t0=0.259 --config.training.t1=0.6760
#python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm_I2 --mode train --config.training.t0=0.6760 #--config.training.t1=0.6760
#python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm_I0_x0_weighted --mode train --config.training.t1=0.259 --config.training.objective_weight="x0_weightedloss"
#python main.py --config "configs/vp/cifar10_ddpm_half.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm_I2_half --mode train --config.training.t0=0.6760
#python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm_I2_x0_weighted --mode train --config.training.t0=0.6760 --config.training.objective_weight="x0_weightedloss"
torchrun --nproc_per_node=2 main.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/new_interval/dpm_deep_I0 --mode train --config.training.t1=0.4420 --config.training.batch_size=128  --noconfig.training.snapshot_sampling
#torchrun --nproc_per_node=2 main.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm_deep_I1 --mode train --config.training.t0=0.259 --config.training.t1=0.6760 --config.training.batch_size=128 --noconfig.training.snapshot_sampling