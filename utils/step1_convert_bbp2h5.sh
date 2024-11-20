#Convert reports from bbp format to h5
# to run it: bash step1_convert_bbp2h5.sh OutputRoot #where OutputRoot is path to ndam sim output folder same as in BlueConfig
# example: bash step1_convert_bbp2h5.sh /gpfs/bbp.cscs.ch/project/proj34/scratch/polina/invivolike/met_gen_onepulse 

#module load archive/2020-11 brion

module load archive/2020-06 brion
#module load unstable

#cd OutputRoot # specify OutputRoot same as in BlueConfig
cd $1 #/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/invivolike/met_gen_onepulse # modify it to OutputRoot of given sim to be analysed

for i in *.bbp ; do compartmentConverter $i "${i%.bbp}.h5" ; done

#compartmentConverter NaCurrSumCol.bbp NaCurrSumCol.h5

cd - 

