#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH --time=00-04:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=evaldeep.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=0
echo  $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch

python main.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --mode eval --workdir shared_exp/dpm_deep --config.eval.begin_ckpt=8 --config.eval.end_ckpt=8
