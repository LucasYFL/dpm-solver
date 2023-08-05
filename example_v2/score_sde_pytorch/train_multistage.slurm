#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=new_int
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=05-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:2
#SBATCH --output=i1_double.out
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL,EXP_FEWER_STEPS=4
echo $EXP_FEWER_STEPS
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm2
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
#torchrun --nproc_per_node=2 --master_port=29600 main.py --config ./configs/vp/uniform.py --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/uniform_interval --mode train --config.training.batch_size=128 
#torchrun --nproc_per_node=2 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous_v2.py --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/multistage_gn16_v2 --mode train --config.training.batch_size=128 
#torchrun --nproc_per_node=2 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous_compare.py --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/one_stage --mode train --config.training.batch_size=128 
#torchrun --nproc_per_node=2 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous_3stage_uniform.py --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/uniform_intervalv2 --mode train --config.training.batch_size=128 
#torchrun --nproc_per_node=2 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_serial_deep_continuous.py --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/serial --mode train --config.training.batch_size=128 
torchrun --nproc_per_node=2 --master_port=29602 main.py --config ./configs/vp/cifar10_ddpmpp_deep_continuous.py --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/new_interval/dpm_deep_I1_b --mode train --config.training.t0=0.442 --config.training.t1=0.6308 --config.model.nf=160 --config.training.batch_size=128 --config.training.snapshot_freq=10000