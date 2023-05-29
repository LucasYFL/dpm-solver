#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --time=00-10:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=multEval.out
#SBATCH --cpus-per-task=8
echo $EXP_FEWER_STEPS
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm2
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
torchrun --nproc_per_node=1 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous.py --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/multistage --mode eval --config.eval.begin_ckpt=1 --config.eval.end_ckpt=9 --config.eval.batch_size=1024 --config.sampling.steps=20  --config.sampling.eps=1e-4
python evaluation_fromsample.py --config "configs/vp/cifar10_ncsnpp_multistage_deep_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/multistage  --config.eval.begin_ckpt=1 --config.eval.end_ckpt=9 --config.eval.batch_size=1024