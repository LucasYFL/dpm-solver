#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=06-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:2
#SBATCH --output=testD.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=3

echo $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir experiments/dpm_data --mode train #--workdir experiments/dpm_unconditional #--noconfig.model.conditional