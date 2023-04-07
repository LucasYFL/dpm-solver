#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=01-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=evalF_%a.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=1
echo  $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir experiments/dpm_fewer --config.eval.begin_ckpt=$SLURM_ARRAY_TASK_ID --config.eval.end_ckpt=$SLURM_ARRAY_TASK_ID
