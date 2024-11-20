#!/bin/bash -l
#SBATCH --job-name="freq"
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --account=proj34
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --error=ngv-stderr%j.log

module purge
module load archive/2020-11 py-bluepy py-efel

srun python getMeanFreq.py /gpfs/bbp.cscs.ch/project/proj34/scratch/polina/setups_sim/thal1p2_noextras_emodels_dend_modif_nfs/BlueConfig /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/out_data/freq_thal1p2_noextras_emodels_dend_modif_nfs.pkl 1000 2000 
