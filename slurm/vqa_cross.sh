#!/usr/bin/bash
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL  # Email when job ends or fails
#SBATCH --mail-user=gdhanuka@andrew.cmu.edu # Replace with your email address
#SBATCH --output=logs/slurm-%A-videotf.out  # Standard output log file (per task)
#SBATCH --error=logs/slurm-%A-videotf.err   # Standard error log file (per task)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion

mkdir -p logs  # Directory for log files

python /home/gdhanuka/STAR_Benchmark/baselines/video_embed_transformer.py