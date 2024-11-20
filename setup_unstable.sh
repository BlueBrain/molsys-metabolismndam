#!/bin/bash

module load unstable
module load neurodamus-neocortex
module load intel hpe-mpi

module load julia/1.6.0

python3 -m venv py_venv_unstable

source py_venv_unstable/bin/activate

julia -e 'using Pkg; Pkg.add("IJulia")'
julia -e 'using Pkg; Pkg.add("DifferentialEquations")'
julia -e 'using Pkg; Pkg.add("ParameterizedFunctions")'
julia -e 'using Pkg; Pkg.add("StaticArrays")'
julia -e 'using Pkg; Pkg.add("JSON")'
julia -e 'using Pkg; Pkg.add("RecursiveArrayTools")'
julia -e 'using Pkg; Pkg.add("Catalyst")'
julia -e 'using Pkg; Pkg.add("Latexify")'
julia -e 'using Pkg; Pkg.add("PyCall");Pkg.build("PyCall")'

python3 -m pip install julia

python3 -c 'import julia as jl; jl.install()'

python3 -m pip install diffeqpy
python3 -m pip install psutil
python3 -m pip install pympler
python3 -m pip install h5py

export PATH="$PATH:~/gpfs/bbp.cscs.ch/home/shichkov/.local/bin"

echo "completed"
