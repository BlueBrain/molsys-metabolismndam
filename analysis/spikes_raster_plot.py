from bluepy.v2 import Simulation
from bluepy.v2 import Circuit
from bluepy.v2 import Cell

import efel

from os import path

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import pickle as pkl

import random

sim1 = Simulation('/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/setup_sim/sim4fig/synchrony/atp_1_2/BlueConfig')
sim2 = Simulation('/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/sim/setup_sim/sim4fig/synchrony/atp_2_8/BlueConfig')



report1 = sim1.report('soma', source='h5')
report2 = sim2.report('soma', source='h5')

data1 = report1.get()
data2 = report2.get()

data1 = data1.reset_index()
data2 = data2.reset_index()


c = Circuit('/gpfs/bbp.cscs.ch/project/proj64/circuits/O1.v6a/20181207/CircuitConfig')
mc2_cells = c.cells.get({'$target': 'mc2_Column'}, properties=[Cell.X, Cell.Y, Cell.Z,Cell.SYNAPSE_CLASS,Cell.LAYER])
mc2gidsids = {k: v for k, v in enumerate(mc2_cells.index)}


print("spike_counts1")

spike_counts1 = pd.DataFrame(columns = ['time','gid'])

#for idx in data1.columns[0:len(data1.columns)-1]:   # when reading data using h5py

for idx in data1.columns[1:len(data1.columns)]:    # when reading data using bluepy
    Index_label1 = np.asarray(data1.index.tolist()) 
    
    candidates = []
    
    for item in Index_label1:
        if data1.loc[item,idx] >0:
            candidates.append(item)
    
    if len(candidates) > 0: 
        Index_label = np.asarray(candidates) 

        sidx = Index_label.argsort()
        ys = Index_label[sidx]


        cut_idx = np.flatnonzero(np.concatenate(([True], np.diff(ys)!=1 )))

        y_new = Index_label[np.minimum.reduceat(sidx, cut_idx)]

        #print(y_new)
        
        for y_new_t in y_new:
            timepoint = data1.loc[y_new_t,'time']
            spike_counts1.loc[len(spike_counts1.index)] = [timepoint, idx]  





spikes_df1 = pd.merge(spike_counts1, mc2_cells, left_on='gid',right_index=True, how='inner')

print("plot2")

spikes_df1['colors'] = None
spikes_df1.loc[spikes_df1['layer']==1,'colors'] = '#FFF200'
spikes_df1.loc[spikes_df1['layer']==2,'colors'] = '#F7941D'
spikes_df1.loc[spikes_df1['layer']==3,'colors'] = '#E02F61'
spikes_df1.loc[spikes_df1['layer']==4,'colors'] = '#FC9BFD'
spikes_df1.loc[spikes_df1['layer']==5,'colors'] = '#68A8E0'
spikes_df1.loc[spikes_df1['layer']==6,'colors'] = '#6CE662'

fig, axs = plt.subplots(2, figsize=(15, 6))
ax = axs[0]
t_window = 50

ax.vlines(spikes_df1['time'], 
          spikes_df1['y'], 
          spikes_df1['y'] + 20, 
          rasterized=True, lw=0.1,
         colors=spikes_df1['colors'])

ax2 = ax.twinx()
ax2.hist(spikes_df1['time'], 
         bins=np.linspace(0, 3000, 101), histtype='step', 
         weights=np.zeros(spikes_df1['time'].size) + (3000.0/50.0)/spikes_df1['gid'].size, 
         color='orange')

ax2.set_ylabel('FR (Hz)')

ax.set_xlim([0, 3000])


fig.savefig('/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/analysis/sim_analyze/plots/raster1_atp1p2_5mar2021.png',dpi=300, bbox_inches='tight', transparent=False)
fig.savefig('/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/analysis/sim_analyze/plots/raster1_atp1p2_5mar2021.pdf',dpi=300, bbox_inches='tight', transparent=False)





print("spike_counts2")

spike_counts2 = pd.DataFrame(columns = ['time','gid'])

#for idx in data2.columns[0:len(data2.columns)-1]:   # when reading data using h5py

for idx in data2.columns[1:len(data2.columns)]:    # when reading data using bluepy
    Index_label1 = np.asarray(data2.index.tolist()) 
    
    candidates = []
    
    for item in Index_label1:
        if data2.loc[item,idx] >0:
            candidates.append(item)
    
    if len(candidates) > 0: 
        Index_label = np.asarray(candidates) 

        sidx = Index_label.argsort()
        ys = Index_label[sidx]


        cut_idx = np.flatnonzero(np.concatenate(([True], np.diff(ys)!=1 )))

        y_new = Index_label[np.minimum.reduceat(sidx, cut_idx)]

        #print(y_new)
        
        for y_new_t in y_new:
            timepoint = data2.loc[y_new_t,'time']
            spike_counts2.loc[len(spike_counts2.index)] = [timepoint, idx]  



spikes_df2 = pd.merge(spike_counts2, mc2_cells, left_on='gid',right_index=True, how='inner')

print("plot2")

spikes_df2['colors'] = None
spikes_df2.loc[spikes_df2['layer']==1,'colors'] = '#FFF200'
spikes_df2.loc[spikes_df2['layer']==2,'colors'] = '#F7941D'
spikes_df2.loc[spikes_df2['layer']==3,'colors'] = '#E02F61'
spikes_df2.loc[spikes_df2['layer']==4,'colors'] = '#FC9BFD'
spikes_df2.loc[spikes_df2['layer']==5,'colors'] = '#68A8E0'
spikes_df2.loc[spikes_df2['layer']==6,'colors'] = '#6CE662'


fig, axs = plt.subplots(2, figsize=(15, 6))
ax = axs[0]
t_window = 50

ax.vlines(spikes_df2['time'], 
          spikes_df2['y'], 
          spikes_df2['y'] + 20, 
          rasterized=True, lw=0.1,
         colors=spikes_df2['colors'])

ax2 = ax.twinx()
ax2.hist(spikes_df2['time'], 
         bins=np.linspace(0, 3000, 101), histtype='step', 
         weights=np.zeros(spikes_df2['time'].size) + (3000.0/50.0)/spikes_df2['gid'].size, 
         color='orange')

ax2.set_ylabel('FR (Hz)')
ax.set_xlim([0, 3000])


fig.savefig('/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/analysis/sim_analyze/plots/raster2_atp2p8_5mar2021.png',dpi=300, bbox_inches='tight', transparent=False)
fig.savefig('/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/analysis/sim_analyze/plots/raster2_atp2p8_5mar2021.pdf',dpi=300, bbox_inches='tight', transparent=False)



