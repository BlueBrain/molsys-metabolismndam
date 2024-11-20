import sys
from bluepy.v2 import Simulation
import efel
import numpy as np
import pickle

# to run it:
# module load unstable py-bluepy py-efel
# python getMeanFreq.py Path2BlueConfig outfilename
# example: python getMeanFreq.py /gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_general/BlueConfig ../out_data/soma_vmv.pickle 1000 2000

#############################
Path2BlueConfig = sys.argv[1] # example Path2BlueConfig: '/gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_general/BlueConfig'
outfilename = sys.argv[2] # example: '../out_data/soma_vmv.pickle'

sim = Simulation(Path2BlueConfig)

stim_start = float(sys.argv[3]) #0 #modify it  # stim_start and stim_end are used to specify the time range you want to analyse
stim_end = float(sys.argv[4]) #np.max(data.time) #modify it # stim_start and stim_end are used to specify the time range you want to analyse

report = sim.report('soma', source='h5')
data = report.get()
data = data.reset_index() 
###########################

feature_values = {}

for idx,cell_gid in enumerate(data.columns[1:len(data.columns)]):
    trace = {'T': data['time'], 'V': data[cell_gid], 'stim_start': [stim_start], 'stim_end': [stim_end]}
    if (list(efel.getFeatureValues([trace], ['mean_frequency'])[0].values())[0] is not None):
        feature_values[cell_gid] = list(efel.getFeatureValues([trace], ['mean_frequency'])[0].values())[0].tolist()[0]

with open(outfilename, 'wb') as handle:
    pickle.dump(feature_values, handle)

print("Finished: getMeanFreq.py")
