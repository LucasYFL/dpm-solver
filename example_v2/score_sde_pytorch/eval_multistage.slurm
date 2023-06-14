#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --time=01-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=multc2.out
#SBATCH --cpus-per-task=8
echo $EXP_FEWER_STEPS
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm2
cd /home/yifulu/work/dpm-solver/example_v2/score_sde_pytorch
#torchrun --nproc_per_node=1 --master_port=29600 main_interval.py --config "configs/vp/cifar10_ncsnpp_multistage_deep_continuous.py" --config1 "configs/vp/cifar10_ddpmpp_deep_continuous.py" --m1 multistage --m2 dpm_deep --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder multistage_converge/i0 --config.eval.t_tuples="(0.4420,)" --config.eval.t_converge="(0,1)" --config.eval.begin_ckpt=1 --config.eval.end_ckpt=12 --config.eval.batch_size=1024 --config.sampling.steps=20  --config.sampling.eps=1e-4 --config.eval.converge_epoch=8
#torchrun --nproc_per_node=1 --master_port=29600 main_interval.py --config1 ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous.py --config ./configs/vp/cifar10_ddpmpp_deep_continuous.py --config2 ./configs/vp/cifar10_ddpmpp_deep_continuous.py --m1 dpm_deep --m2 multistage --m3 dpm_deep --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder multistage_converge/i1 --config.eval.t_tuples="(0.4420,0.6308)" --config.eval.t_converge="(1,0,1)" --config.eval.begin_ckpt=1 --config.eval.end_ckpt=12 --config.eval.batch_size=1024 --config.sampling.steps=20  --config.sampling.eps=1e-4 --config.eval.converge_epoch=8
torchrun --nproc_per_node=1 --master_port=29600 main_interval.py --config1 ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous.py --config ./configs/vp/cifar10_ddpmpp_deep_continuous.py  --m1 dpm_deep --m2 multistage --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments --eval_folder multistage_converge/i2 --config.eval.t_tuples="(0.6308,)" --config.eval.t_converge="(1,0)" --config.eval.begin_ckpt=1 --config.eval.end_ckpt=12 --config.eval.batch_size=1024 --config.sampling.steps=20  --config.sampling.eps=1e-4 --config.eval.converge_epoch=8
python evaluation_fromsample.py --config "configs/vp/cifar10_ncsnpp_multistage_deep_continuous.py" --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/multistage_converge --eval_folder i2 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=12 --config.eval.batch_size=1024