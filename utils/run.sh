#!/bin/bash -l
#SBATCH --job-name="efel"
#SBATCH --partition=prod
#SBATCH --nodes=2
#SBATCH --cpus-per-task=2
#SBATCH --time=18:00:00
#SBATCH --account=proj34
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --error=ngv-stderr%j.log

module purge
module load archive/2020-11 py-bluepy py-efel

srun bash iter_efelAll2s1.sh 
