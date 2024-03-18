#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=meme3
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --time=03-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=gpu_mig40
#SBATCH --gpus=nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --output=meme3.out
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL,EXP_FEWER_STEPS=4
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/multistage/dpm-solver/example_v2/score_sde_pytorch
# torchrun main.py --config configs/vp/cifar10_MEME.py --mode train --workdir meme/s0 --config.model.stage_num=0 --config.training.t0=0.0 --config.training.t1=0.25  --config.training.batch_size=128
# torchrun --master_port=29700 main.py  --config configs/vp/cifar10_MEME.py --mode train --workdir meme/s1 --config.model.stage_num=1 --config.training.t0=0.25 --config.training.t1=0.5  --config.training.batch_size=128
# torchrun --master_port=29800 main.py --config configs/vp/cifar10_MEME.py --mode train --workdir meme/s2 --config.model.stage_num=2 --config.training.t0=0.5 --config.training.t1=0.75  --config.training.batch_size=128
torchrun --master_port=29600 main.py  --config configs/vp/cifar10_MEME.py --mode train --workdir meme/s3 --config.model.stage_num=3 --config.training.t0=0.75 --config.training.t1=1.0  --config.training.batch_size=128