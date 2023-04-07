#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=DPM_eval_continuous
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --time=04-00:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=/scratch/qingqu_root/qingqu1/huijiezh/out/evalcontinupous.out
#SBATCH --export=ALL,EXP_FEWER_STEPS=2
echo  $EXP_FEWER_STEPS
module purge
module load cuda/11.3.0 cudnn/11.3-v8.2.1
eval "$(conda shell.bash hook)"
conda activate score_sde_pytorch
cd /scratch/qingqu_root/qingqu1/huijiezh/dpm-solver/example_v2/score_sde_pytorch

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t000 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.0010000000474974513

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t010 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.10090000182390213

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t015 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.15084998309612274

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t020 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.20079998672008514

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t025 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.25074997544288635

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t035 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.35064995288848877

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t040 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.40059995651245117

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t045 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.4505499303340912

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t050 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.5004999041557312

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t055 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.5504499673843384

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t060 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.6003999710083008

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t065 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.6504499673843384

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t070 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.7002999782562256

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t075 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.750249981880188

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t080 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.8001999258995056

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t085 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.8501499891281128

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t095 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=0.950049877166748

python main.py --config "configs/vp/cifar10_ddpm_continuous.py" --mode eval --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/dpm --eval_folder noise_loss_t1 --config.eval.begin_ckpt=1 --config.eval.end_ckpt=40 --noconfig.eval.enable_sampling --config.eval.enable_loss --config.eval.t=1.0


