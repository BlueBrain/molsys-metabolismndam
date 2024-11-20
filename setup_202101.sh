#!/bin/bash

module load archive/2021-01
module load neurodamus-neocortex/1.2-3.1.0
module load neuron/7.9.0b

camps_hoc="/gpfs/bbp.cscs.ch/project/proj34/camps/custom_neurodamus/neocortex/hoc"
camps_mod="/gpfs/bbp.cscs.ch/project/proj34/camps/custom_neurodamus/neocortex/mod/ionic_v6_unclamped_dan_ionic"
merged202011mod="/gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/custom_ndam/mod"

custom_ndam_dir="custom_ndam_$(date +"%Y_%m_%d")"

if [[ ! -e $custom_ndam_dir ]]; then
    mkdir $custom_ndam_dir
elif [[ ! -d $custom_ndam_dir ]]; then
    echo "$custom_ndam_dir exists but is not a directory. Rename your $custom_ndam_dir file and run again."
fi

CLASSICSPECIAL=`which special`
echo $CLASSICSPECIAL

classichoc="${CLASSICSPECIAL%"bin/special"}lib/hoc"
classicmod="${CLASSICSPECIAL%"bin/special"}lib/mod"

echo $classichoc
echo $classicmod

cd $custom_ndam_dir

echo `diff <(ls -1a $classichoc) <(ls -1a $camps_hoc)`

cp -r $classichoc .

echo `diff <(ls -1a $classicmod) <(ls -1a $camps_mod)`

echo `diff <(ls -1a /gpfs/bbp.cscs.ch/ssd/apps/hpc/jenkins/deploy/applications/2020-02-01/linux-rhel7-x86_64/intel-19.0.4/neurodamus-neocortex-1.1-3.0.2-t7gybsjwj4/lib/mod) <(ls -1a /gpfs/bbp.cscs.ch/ssd/apps/hpc/jenkins/deploy/applications/2021-01-06/linux-rhel7-x86_64/intel-19.1.2.254/neurodamus-neocortex-1.2-3.1.0-s7vj2h/lib/mod)`

cp -r $merged202011mod mod
cp "$classicmod/neuron_only_mods.txt" mod/.

echo "internal_ions.mod" >> mod/neuron_only_mods.txt

export HOC_LIBRARY_PATH=`pwd`/hoc/

build_neurodamus.sh mod

cd -


module load julia/1.5.2

python3 -m venv py_venv_dir_2021

source py_venv_dir_2021/bin/activate

julia -e 'using Pkg; Pkg.add("IJulia")'
julia -e 'using Pkg; Pkg.add("DifferentialEquations")'
julia -e 'using Pkg; Pkg.add("ParameterizedFunctions")'
julia -e 'using Pkg; Pkg.add("StaticArrays")'
julia -e 'using Pkg; Pkg.add("PyCall");Pkg.build("PyCall")'

python3 -m pip install julia

python3 -c 'import julia as jl; jl.install()'

python3 -m pip install diffeqpy
python3 -m pip install psutil
python3 -m pip install pympler
python3 -m pip install h5py

export PATH="$PATH:~/gpfs/bbp.cscs.ch/home/shichkov/.local/bin"

echo "completed"
