#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=01-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=v100:2
#SBATCH --output=eval1.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=2
echo  $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch

#python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir experiments/dpm --eval_folder on_first_t --config.eval.begin_ckpt=40 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.30069997906684875
python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir experiments/dpm_one_step --eval_folder on_first_t --config.eval.begin_ckpt=9 --config.eval.end_ckpt=9 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.30069997906684875