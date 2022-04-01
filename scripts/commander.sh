#!/bin/bash

#SBATCH --job-name=my_test
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH --qos=an_all_short
#SBATCH --partition=an
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --wckey="P11MM:SALOME"
#SBATCH --exclusive
#SBATCH -o out/commander.out
#SBATCH -e out/commander.err

set -x
srun hostname
. /home/i37181/anaconda3/etc/profile.d/conda.sh

conda activate env_gat
mkdir out
srun python commander.py
