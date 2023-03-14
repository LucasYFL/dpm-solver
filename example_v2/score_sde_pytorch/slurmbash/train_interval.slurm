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
#SBATCH --output=testI1.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=4

echo $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate score_sde_pytorch
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm_I1 --mode train --config.training.t0=0.2590 --config.training.t1=0.6760