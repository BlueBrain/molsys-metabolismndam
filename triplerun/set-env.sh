module purge
 module load unstable
# module load archive/2020-11
#module load archive/2021-05
module load python-dev py-mpi4py
module load neurodamus-neocortex
module load py-neurodamus
module load intel hpe-mpi
module load steps

module load julia/1.6.0

python -mvirtualenv triplerun-env
. ./triplerun-env/bin/activate

#pip install /gpfs/bbp.cscs.ch/home/gcastigl/project34/neurodamus-py/


julia -e 'using Pkg; Pkg.add("IJulia")'
julia -e 'using Pkg; Pkg.add("DifferentialEquations")'
julia -e 'using Pkg; Pkg.add("ParameterizedFunctions")'
julia -e 'using Pkg; Pkg.add("StaticArrays")'
julia -e 'using Pkg; Pkg.add("RecursiveArrayTools")'
julia -e 'using Pkg; Pkg.add("PyCall");Pkg.build("PyCall")'

python3 -m pip install julia

python3 -c 'import julia as jl; jl.install()'

python3 -m pip install diffeqpy
python3 -m pip install psutil
python3 -m pip install pympler
python3 -m pip install h5py



#PYTHONPATH=/gpfs/bbp.cscs.ch/home/gcastigl/software/install/linux-rhel7-x86_64/gcc-8.3.0/steps-develop-jngudx:$PYTHONPATH
# PYTHONPATH=/gpfs/bbp.cscs.ch/home/gcastigl/software/install/linux-rhel7-x86_64/gcc-8.3.0/steps-develop-bwafym:$PYTHONPATH
# PYTHONPATH=/gpfs/bbp.cscs.ch/home/gcastigl/spack/opt/spack/linux-rhel7-x86_64/gcc-9.3.0/steps-develop-kupiyq:$PYTHONPATH
export HOC_LIBRARY_PATH=/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/triplerun/custom_ndam_special/hoc/
export PATH="$PATH:~/gpfs/bbp.cscs.ch/home/shichkov/.local/bin"

#export PYTHONPATH=$PYTHONPATH:$HOC_LIBRARY_PATH

echo "completed"

