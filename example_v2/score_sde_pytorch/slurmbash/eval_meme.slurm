#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=eval_meme
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --time=01-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=gpu_mig40
#SBATCH --gpus=nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --output=eval_meme.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=4
echo  $EXP_FEWER_STEPS
module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate dpm
cd /home/yifulu/work/multistage/dpm-solver/example_v2/score_sde_pytorch
torchrun main_interval.py --config configs/vp/cifar10_MEME.py --config1 configs/vp/cifar10_MEME.py --config2 configs/vp/cifar10_MEME.py --config3 configs/vp/cifar10_MEME.py --workdir meme --eval_folder test --m1 s0 --m2 s1 --m3 s2 --m4 s3 --config.eval.begin_ckpt=5 --config.eval.end_ckpt=30 --config.eval.ckpt_interval=5 --config1.model.stage_num=1 --config2.model.stage_num=2 --config3.model.stage_num=3 
python evaluation_fromsample.py --config configs/vp/cifar10_MEME.py --workdir meme --eval_folder test --config.eval.begin_ckpt=5 --config.eval.end_ckpt=30   --config.eval.ckpt_interval=5
# torchrun main_interval.py --config configs/vp/cifar10_MEME.py --config1 configs/vp/cifar10_MEME.py --config2 configs/vp/cifar10_MEME.py --config3 configs/vp/cifar10_MEME.py --workdir meme --eval_folder test --m1 s0 --m2 s1 --m3 s2 --m4 s3 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=3 --config.eval.ckpt_interval=1 --config1.model.stage_num=1 --config2.model.stage_num=2 --config3.model.stage_num=3 
# python evaluation_fromsample.py --config configs/vp/cifar10_MEME.py --workdir meme --eval_folder test --config.eval.begin_ckpt=1 --config.eval.end_ckpt=3 --config.eval.ckpt_interval=1
# torchrun main_interval.py --config configs/vp/cifar10_MEME.py --config1 configs/vp/cifar10_MEME.py --config2 configs/vp/cifar10_MEME.py --config3 configs/vp/cifar10_MEME.py --workdir meme --eval_folder test --m1 s0 --m2 s1 --m3 s2 --m4 s3 --config.eval.begin_ckpt=5 --config.eval.end_ckpt=5 --config.eval.ckpt_interval=1 --config1.model.stage_num=1 --config2.model.stage_num=2 --config3.model.stage_num=3 
# python evaluation_fromsample.py --config configs/vp/cifar10_MEME.py --workdir meme --eval_folder test --config.eval.begin_ckpt=5 --config.eval.end_ckpt=5   --config.eval.ckpt_interval=1