#SBATCH --account=proj34
#SBATCH --partition=prod
#SBATCH --nodes=300
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --constraint=cpu
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --job-name=HippocampusNrdmsPySim
#SBATCH --output=/gpfs/bbp.cscs.ch/project/proj113/scratch/hippocodes/notebooks/ach_modulation/BioM/57a65a02-1dda-4a80-8d6e-237a25417c67/%a/.out.dat.job_log
#SBATCH --array=0-3%10
#SBATCH --wait

cd /gpfs/bbp.cscs.ch/project/proj113/scratch/hippocodes/notebooks/ach_modulation/BioM/57a65a02-1dda-4a80-8d6e-237a25417c67/$SLURM_ARRAY_TASK_ID
srun bash -c "test -f out.dat || dplace special -mpi -python $NEURODAMUS_PYTHON/init.py "

