#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=01-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:4
#SBATCH --output=multistage2.out
#SBATCH --cpus-per-task=8
echo $EXP_FEWER_STEPS
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm2
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
torchrun --nproc_per_node=4 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous.py --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/multistage --mode train --config.training.batch_size=128 
