#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval2
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=00-04:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=evali2x0w.out
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL,EXP_FEWER_STEPS=4
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
torchrun --nproc_per_node=1 main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --m1 dpm --m2 dpm_I2_x0_weighted --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval2x0w --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.6760,)" --config.eval.t_converge="(1,0)" --config.eval.batch_size=1024 --config.sampling.eps=1e-4 --config.sampling.steps=20
python evaluation_fromsample.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval2x0w --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.batch_size=1024