import sys
from bluepy.v2 import Simulation
import efel
import numpy as np
import pickle

# to run it:
# module load unstable py-bluepy py-efel
# python getAllEfel_10percentGids.py Path2BlueConfig outfilename 1
# example: python getAllEfel_10percentGids.py /gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_general/BlueConfig ../out_data/efel_met_general_thal.pickle 1

#############################
Path2BlueConfig = sys.argv[1] # example Path2BlueConfig: '/gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_general/BlueConfig'
outfilename = sys.argv[2] # example: '../out_data/soma_vmv.pickle'

features_set = sys.argv[3] # 1 or 2 for f1 or f2 of efel_feats correspondingly

tenpercent_gids_f = '/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/in_data/mc2_tenpercent_allLayers_EXCandINH_gids.txt'


sim = Simulation(Path2BlueConfig)

stim_start = 0.1 #0 #modify it  # stim_start and stim_end are used to specify the time range you want to analyse

report = sim.report('soma', source='h5')
data = report.get()
data = data.reset_index() 

stim_end = np.max(data.time) - 0.1  #modify it # stim_start and stim_end are used to specify the time range you want to analyse

############################

#efel_feats = efel.getFeatureNames() # all features


if features_set == '1':
    #f1:
    efel_feats = ['AP_amplitude','AP_height','AP_duration_half_width','mean_frequency', 'AHP_depth','AHP_time_from_peak','voltage_base', 'time_to_first_spike','time_to_last_spike','ISI_CV','ISI_log_slope', 'inv_first_ISI','inv_second_ISI', 'inv_third_ISI', 'inv_fourth_ISI', 'inv_fifth_ISI', 'inv_last_ISI']
elif features_set == '2':
    #f2:
    efel_feats = ['AHP_depth_abs', 'Spikecount','amp_drop_first_last','depolarized_base','irregularity_index', 'max_amp_difference','maximum_voltage','mean_AP_amplitude','peak_voltage','spike_half_width']
else:
    print("use 1 or 2 for f1 or f2 of efel_feats")

#f1 + f2
#efel_feats = ['AP_amplitude','AP_height','AP_duration_half_width','mean_frequency', 'AHP_depth','AHP_time_from_peak','voltage_base', 'time_to_first_spike','time_to_last_spike','ISI_CV','ISI_log_slope', 'inv_first_ISI','inv_second_ISI', 'inv_third_ISI', 'inv_fourth_ISI', 'inv_fifth_ISI', 'inv_last_ISI','AHP_depth_abs', 'Spikecount','amp_drop_first_last','depolarized_base','irregularity_index', 'max_amp_difference','maximum_voltage','mean_AP_amplitude','peak_voltage','spike_half_width']


subset_percent_gids = np.loadtxt(tenpercent_gids_f)
data = data.loc[:,['time']+subset_percent_gids.tolist()] 

feature_values = {}

for idx,cell_gid in enumerate(data.columns[1:len(data.columns)]):
    trace = {'T': data['time'], 'V': data[cell_gid], 'stim_start': [stim_start], 'stim_end': [stim_end]}
    if (list(efel.getFeatureValues([trace], ['mean_frequency'])[0].values())[0] is not None): 
        fv = efel.getFeatureValues([trace], efel_feats)[0]
        fv = {feature_name: np.mean(values) for feature_name, values in fv.items() if values is not None}
        feature_values[cell_gid] = fv

with open(outfilename, 'wb') as handle:
    pickle.dump(feature_values, handle)

print("Finished")

