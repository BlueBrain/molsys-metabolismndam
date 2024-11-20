module load archive/2020-11 py-bluepy py-efel

for i in  {1..10}
do
    python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/setup_sim/sim4fig/gen_diff_pulses_25feb2021/linamp${i}exc/BlueConfig /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/results/gen_diff_pulses/linamp${i}/efel_amp_f1.pickle 1
    python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/setup_sim/sim4fig/gen_diff_pulses_25feb2021/linamp${i}exc/BlueConfig /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/results/gen_diff_pulses/linamp${i}/efel_amp_f2.pickle 2
done
