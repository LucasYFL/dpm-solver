#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval_stage_effect
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=03-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=eval_stage_effect.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=2
echo  $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate score_sde_pytorch
cd /scratch/qingqu_root/qingqu1/huijiezh/dpm-solver/example_v2/score_sde_pytorch

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder eval_stage_effect_t090 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40  config.eval.compare_step=2

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder eval_stage_effect_t030 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40  config.eval.compare_step=14

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder eval_stage_effect_t005 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40  config.eval.compare_step=19

