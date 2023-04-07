#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH --time=00-10:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=evalI2q.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=4
echo  $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm_I0 --m2 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval0 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.259,)" --config.eval.t_converge="(0,1)"
#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm --m2 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval_control_0 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.259,)" --config.eval.t_converge="(0,1)"
#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm --m2 dpm --m3 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval_control_1 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_converge="(1,0,1)"
#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm --m2 dpm --m3 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval_control_1 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_converge="(1,0,1)"
#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm --m2 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval_control_2 --config.eval.begin_ckpt=6 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.6760,)" --config.eval.t_converge="(1,0)"
#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm --m2 dpm_I1 --m3 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval1 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_converge="(1,0,1)"
#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm --m2 dpm_I2 --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval2 --config.eval.begin_ckpt=2 --config.eval.end_ckpt=5 --config.eval.ckpt_interval=1 --config.eval.t_tuples="(0.6760,)" --config.eval.t_converge="(1,0)"
#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm_I0 --m2 dpm_I1 --m3 dpm_I2 --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder intervalA --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.259,0.6760)" --config.eval.t_converge="(0,0,1)" --config.eval.converge_epoch=2
#python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm_I0_x0_weighted --m2 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval0_x0_weighted --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.259,)" --config.eval.t_converge="(0,1)" --config.eval.objectives='("x0_weightedloss","eps")'
python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --config1 "configs/vp/cifar10_ddpm_quart.py" --m1 dpm --m2 dpm_I2_quart --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval2_quart --config.eval.begin_ckpt=4 --config.eval.end_ckpt=5 --config.eval.ckpt_interval=1 --config.eval.t_tuples="(0.6760,)" --config.eval.t_converge="(1,0)"
python main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --config1 "configs/vp/cifar10_ddpm_quart.py" --m1 dpm --m2 dpm_I2_quart --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval2_quart --config.eval.begin_ckpt=7 --config.eval.end_ckpt=8 --config.eval.ckpt_interval=1 --config.eval.t_tuples="(0.6760,)" --config.eval.t_converge="(1,0)"