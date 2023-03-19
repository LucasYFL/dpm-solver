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
#SBATCH --output=evalIC1.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=4
echo  $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch

#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm --m2 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval_control_0 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.259,)" --config.eval.t_converge="(0,1)"
python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm --m2 dpm --m3 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval_control_1 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_converge="(1,0,1)"