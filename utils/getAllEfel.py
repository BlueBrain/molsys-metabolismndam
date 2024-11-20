import sys
from bluepy.v2 import Simulation
import efel
import numpy as np
import pickle

# to run it:
# module load unstable py-bluepy py-efel
# python getAllEfel.py Path2BlueConfig outfilename 1 # 1 or 2 is to choose which features_set to calculate
# example: python getAllEfel.py /gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_general/BlueConfig ../out_data/efel_met_general_thal.pickle 1 1000 2000

#############################
Path2BlueConfig = sys.argv[1] # example Path2BlueConfig: '/gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_general/BlueConfig'
outfilename = sys.argv[2] # example: '../out_data/soma_vmv.pickle'
features_set = sys.argv[3] # 1 or 2 for f1 or f2 of efel_feats correspondingly

sim = Simulation(Path2BlueConfig)

stim_start = float(sys.argv[4]) #0.1 #0 #modify it  # stim_start and stim_end are used to specify the time range you want to analyse

report = sim.report('soma', source='h5')
data = report.get()
data = data.reset_index() 

stim_end = float(sys.argv[5]) #np.max(data.time) - 0.1  #modify it # stim_start and stim_end are used to specify the time range you want to analyse

############################

#efel_feats = efel.getFeatureNames() # all features

if (features_set == '1'):
    print("chosen feature set 1")
    #f1:
    efel_feats = ['AP_width','AP_amplitude','AP_height','AP_duration','AP_duration_half_width','mean_frequency', 'AHP_depth','AHP_time_from_peak','voltage_base', 'time_to_first_spike','time_to_last_spike','ISI_CV','ISI_log_slope', 'inv_first_ISI','inv_second_ISI', 'inv_third_ISI', 'inv_fourth_ISI', 'inv_fifth_ISI', 'inv_last_ISI']
elif (features_set == '2'):
    print("chosen feature set 2")
    #f2:
    efel_feats = ['AHP_depth_abs','AHP_depth_diff', 'Spikecount','amp_drop_first_last','depolarized_base','irregularity_index', 'max_amp_difference','maximum_voltage','mean_AP_amplitude','peak_voltage','spike_half_width']
elif (features_set == '3'):
    print("chosen feature set 3")
    efel_feats = ['AP_fall_rate','AP_phaseslope','AP_rise_rate','adaptation_index','adaptation_index2','burst_mean_freq','burst_number','fast_AHP','fast_AHP_change','interburst_voltage','maximum_voltage_from_voltagebase','minimum_voltage','ohmic_input_resistance','sag_amplitude','single_burst_ratio','steady_state_hyper', 'voltage_deflection','depol_block' ]
else:
    print("choose features set")
#f1 + f2
#efel_feats = ['AP_amplitude','AP_height','AP_duration_half_width','mean_frequency', 'AHP_depth','AHP_time_from_peak','voltage_base', 'time_to_first_spike','time_to_last_spike','ISI_CV','ISI_log_slope', 'inv_first_ISI','inv_second_ISI', 'inv_third_ISI', 'inv_fourth_ISI', 'inv_fifth_ISI', 'inv_last_ISI','AHP_depth_abs', 'Spikecount','amp_drop_first_last','depolarized_base','irregularity_index', 'max_amp_difference','maximum_voltage','mean_AP_amplitude','peak_voltage','spike_half_width']


feature_values = {}

for idx,cell_gid in enumerate(data.columns[1:len(data.columns)]):
    trace = {'T': data['time'], 'V': data[cell_gid], 'stim_start': [stim_start], 'stim_end': [stim_end]}
    if (list(efel.getFeatureValues([trace], ['mean_frequency'])[0].values())[0] is not None): 
        fv = efel.getFeatureValues([trace], efel_feats)[0]
        fv = {feature_name: np.nanmedian(values) for feature_name, values in fv.items() if values is not None}
        feature_values[cell_gid] = fv

with open(outfilename, 'wb') as handle:
    pickle.dump(feature_values, handle)

print("Finished")
