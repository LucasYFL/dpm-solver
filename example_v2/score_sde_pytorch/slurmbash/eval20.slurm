#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval2
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --time=00-10:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=evali0ue3.out
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL,EXP_FEWER_STEPS=4
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm2
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
#torchrun --nproc_per_node=1 main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py" --config1 "configs/vp/cifar10_ddpmpp_deep_continuous.py"  --m1 dpm --m2 dpm_deep --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval0u_control --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.259,)" --config.eval.t_converge="(0,1)" --config.eval.batch_size=1024 --config.sampling.eps=1e-4 --config.sampling.steps=20 --config.eval.converge_epoch=8 
torchrun --nproc_per_node=1 main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py"  --config1 "configs/vp/cifar10_ddpmpp_deep_continuous.py" --m1 dpm_I0 --m2 dpm_deep  --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder time_uniform_e-3/interval0 --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.259,)" --config.eval.t_converge="(0,1)" --config.eval.batch_size=1024 --config.sampling.steps=20 --config.eval.converge_epoch=8 --config.sampling.skip_type="time_uniform"
torchrun --nproc_per_node=1 main_interval.py --config "configs/vp/cifar10_ddpm_continuous.py"  --config1 "configs/vp/cifar10_ddpmpp_deep_continuous.py" --m1 dpm --m2 dpm_deep --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder time_uniform_e-3/interval0_control --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.259,)" --config.eval.t_converge="(0,1)" --config.eval.batch_size=1024 --config.sampling.steps=20 --config.eval.converge_epoch=8 --config.sampling.skip_type="time_uniform"
python evaluation_fromsample.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder time_uniform_e-3/interval0 --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.batch_size=1024
python evaluation_fromsample.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder time_uniform_e-3/interval0_control --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.batch_size=1024
#python evaluation_fromsample.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval1_control --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.batch_size=1024
#python evaluation_fromsample.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval2 --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.batch_size=1024
#torchrun --nproc_per_node=1 main_interval.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --m1 dpm_deep_I0 --m2 dpm_deep_I1 --m3 dpm_deep_I2 --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval_deep --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.t_tuples="(0.259,0.676)" --config.eval.t_converge="(0,0,0)" --config.eval.batch_size=1024 --config.sampling.eps=1e-4 --config.sampling.steps=20
#torchrun --nproc_per_node=1 main_interval.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --m1 dpm_deep --m2 dpm_deep --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder dpm_deep/eval --config.eval.begin_ckpt=1 --config.eval.end_ckpt=9 --config.eval.ckpt_interval=1 --config.eval.t_tuples="()" --config.eval.t_converge="(0,)" --config.eval.batch_size=1024 --config.sampling.eps=1e-4 --config.sampling.steps=20 --config.eval.converge_epoch=9
#python evaluation_fromsample.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder dpm_deep/eval --config.eval.begin_ckpt=1 --config.eval.end_ckpt=9 --config.eval.ckpt_interval=1 --config.eval.batch_size=1024
#python evaluation_fromsample.py --config "configs/vp/cifar10_ddpm_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval_deep --config.eval.begin_ckpt=5 --config.eval.end_ckpt=40 --config.eval.ckpt_interval=5 --config.eval.batch_size=1024
#torchrun --nproc_per_node=1 main_interval.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --config1 "configs/vp/cifar10_ddpm_continuous.py"  --m1 dpm_deep --m2 dpm --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder interval0_dpo --config.eval.begin_ckpt=8 --config.eval.end_ckpt=8 --config.eval.t_tuples="(0.259,)" --config.eval.t_converge="(0,1)" --config.eval.batch_size=1024 --config.sampling.eps=1e-4 --config.sampling.steps=20 --config.eval.converge_epoch=40