#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=00-08:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:2
#SBATCH --output=multgn16v2_2.out
#SBATCH --cpus-per-task=8
echo $EXP_FEWER_STEPS
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm2
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
torchrun --nproc_per_node=2 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous_v2.py --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/multistage_gn16_v2 --mode train --config.training.batch_size=128 

torchrun --nproc_per_node=1 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_serial_deep_continuous.py --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/test --mode train --config.training.batch_size=128 
