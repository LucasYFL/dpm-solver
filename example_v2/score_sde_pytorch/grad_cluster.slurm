#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=grad_cluster
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --time=01-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=g_cluster.out
#SBATCH --cpus-per-task=8
echo $EXP_FEWER_STEPS
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm2
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch

python gradient_cluster.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/multistage/dpm_deep  --config.eval.begin_ckpt=1 --config.eval.end_ckpt=9 --config.training.batch_size=128 