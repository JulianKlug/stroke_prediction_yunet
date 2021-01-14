#!/bin/bash
#SBATCH --job-name=run_main_yunet    # Job name
#SBATCH -G volta:2                    # Number of GPUs
#SBATCH --ntasks=2                    # Run on a single CPU
#SBATCH --mem=30G                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=1                     # Job memory request
#SBATCH --output=julian/logs/yunet_log_%j.log   # Standard output and error log

module load anaconda

cd /home/gridsan/gleclerc/julian/yunet
export PYTHONPATH=$(pwd)

source /home/gridsan/gleclerc/.bashrc

conda activate yunet

ulimit -S -n 131072
ulimit -S -u 1546461

srun /home/gridsan/gleclerc/.conda/envs/yunet/bin/python /home/gridsan/gleclerc/julian/yunet/main.py