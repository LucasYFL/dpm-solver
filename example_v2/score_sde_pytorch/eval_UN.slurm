#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=01-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:2
#SBATCH --output=evalF3.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=2
echo  $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir experiments/dpm_one_first --eval_folder noise_loss --config.eval.begin_ckpt=1 --config.eval.end_ckpt=1 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.050950001925230026
python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir experiments/dpm_last --eval_folder noise_loss --config.eval.begin_ckpt=1 --config.eval.end_ckpt=1 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.9000999927520752
