module load archive/2020-11 py-bluepy py-efel

for i in  {1..3}
do
    python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/setup_sim/sim4fig/synchrony/newProj_ca_1p25/BlueConfig ../out_data/efel${i}_newProj_ca_1p25_13mar2021.pickle $i 1000 2000
    python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/scratch/polina/setups_sim/nostim500_ca_1p25/BlueConfig ../out_data/efel${i}_nostim500_ca_1p25_13mar2021.pickle $i 1000 2000
    python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/scratch/polina/setups_sim/nostim500_withGenMet_ca_1p25/BlueConfig ../out_data/efel${i}_nostim500_withGenMet_ca_1p25_13mar2021.pickle $i 1000 2000
    python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/setup_sim/sim4fig/synchrony/newProj_ca_1p25_met/BlueConfig ../out_data/efel${i}_newProj_ca_1p25_met_13mar2021.pickle $i 1000 2000
    python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/scratch/polina/setups_sim/lin1/BlueConfig ../out_data/efel${i}_lin1_13mar2021.pickle $i 1100 1600
    python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/scratch/polina/setups_sim/lin1_nm/BlueConfig ../out_data/efel${i}_lin1_nm_13mar2021.pickle $i 1100 1600

done
