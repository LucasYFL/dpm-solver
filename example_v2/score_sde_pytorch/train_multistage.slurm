#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=biased
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=05-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:2
#SBATCH --output=gn16_v2_biased.out
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL,BIASED_SAMPLE=1

echo $BIASED_SAMPLE
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
torchrun --nproc_per_node=2 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous_v2_new_interval_biased.py --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/multistage_gn16_v2_new_interval_biased --mode train --config.training.batch_size=128 