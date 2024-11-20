for i in  {1..3}
do
    python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/setup_sim/sim4fig/synchrony/newProj_ca_1p25/BlueConfig ../out_data/efel${i}_newProj_ca_1p25_13mar2021.pickle $i 1000 2000
done
