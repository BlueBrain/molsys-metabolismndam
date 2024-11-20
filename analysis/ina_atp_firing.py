# Analyse INa, firing rate, ATP consumption
# module load py-bluepy py-efel #neurodamus-neocortex py-neurodamus

# to run:
# python ina_atp_firing.py percent_gids sim_f mean_freq_f out_f_name
# example:
# python ina_atp_firing.py 10 /gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_gen_inasum_nostim/BlueConfig /gpfs/bbp.cscs.ch/project/proj34/metabolismndam/out_data/soma_vmv_met_gen_inasum_nostim.pickle /gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_gen_inasum_nostim/atpFromINaVsMeanFreq_10percentCells.tsv

from bluepy.v2 import Simulation

import efel

from bluepy.v2 import Circuit
from bluepy.v2 import Cell
from bluepy.v2.enums import Synapse

from os import path

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import random

import pickle
import h5py

import sys

##### INPUT #####

percent_gids = sys.argv[1] # 1 or 10
duration_in_seconds = 3 #from INa report in BlueConfig

circuit_f = '/gpfs/bbp.cscs.ch/project/proj64/circuits/O1.v6a/20181207/CircuitConfig'
sim_f = sys.argv[2] #'/gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_gen_inasum/BlueConfig'

onepercent_gids_f = '/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/in_data/mc2_onepercent_allLayers_EXCandINH_gids.txt'
tenpercent_gids_f = '/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/in_data/mc2_tenpercent_allLayers_EXCandINH_gids.txt'

vols_areas_f = '/gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/cells_volume_calc/volumes_2020113012.txt'
mean_freq_f = sys.argv[3] #'/gpfs/bbp.cscs.ch/project/proj34/metabolismndam/out_data/soma_vmv_met_gen_inasum.pickle'

#ina_f_name = '/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/invivolike/met_gen_inasum_2dec2020/NaCurrentAllCmps.h5'
#soma_v_f_name = '/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/invivolike/met_gen_inasum_2dec2020/soma.h5'

out_f_name = sys.argv[4] #'/gpfs/bbp.cscs.ch/project/proj34/sh_ngvm/column/met_gen_inasum/atpFromINaVsMeanFreq_10percentCells.tsv'

#################
# const

Avogadro = 6.02e23 # 1/mol
Faraday = 96485.0 #96.485 # 96485.0  C/mol  # FARADAY = 96485.309 (coul) in Somjen2008
#ina = [mA/cm2] (distributed)  #coulomb in terms of the ampere and second: 1 C = 1 A × 1 s.

#################

def get_cells(circuit_f,percent_gids,onepercent_gids_f,tenpercent_gids_f):
    c = Circuit(circuit_f)
    mc2_cells = c.cells.get({'$target': 'mc2_Column'}, properties=[Cell.X, Cell.Y, Cell.Z,Cell.SYNAPSE_CLASS,Cell.LAYER,Cell.ETYPE,Cell.MTYPE])
    mc2gidsids = {k: v for k, v in enumerate(mc2_cells.index)}

    if percent_gids == '1':
        print("work with 1% of cells")
        subset_percent_gids = np.loadtxt(onepercent_gids_f)
    elif percent_gids == '10':
        print("work with 10% of cells")
        subset_percent_gids = np.loadtxt(tenpercent_gids_f)
    else:
        print("check percent_gids")

    subset_cells = mc2_cells.loc[subset_percent_gids,:]
    subset_cells['etype_mtype'] = subset_cells['etype'].astype(str) + "_" + subset_cells['mtype'].astype(str)
    subset_cells.index = subset_cells.index.astype(int)
    subset_cells = subset_cells.reset_index()
    subset_cells = subset_cells.rename(columns={'index':'c_gid'})

    return subset_cells

def get_volumes_areas(vols_areas_f,subset_cells):
    vols_areas = pd.read_csv(vols_areas_f, sep='\,\s',header=None)
    vols_areas.columns = ['gid','rank','cells_volumes_um3','cells_areas']
    vols_areas['gid'] = vols_areas['gid'].astype(int)
    vols_areas = vols_areas.loc[vols_areas['gid'].isin(subset_cells['c_gid'].tolist()),:]
    vols_areas = vols_areas.drop_duplicates(keep='first')
    vols_areas = vols_areas.reset_index(drop=True)
    return vols_areas

def get_mean_freq(mean_freq_f,subset_cells):
    with open(mean_freq_f, 'rb') as handle:
        feature_values = pickle.load(handle)
    fr = pd.DataFrame(feature_values.items(),columns = ['gid','mean_frequency'])
    fr['gid']=fr['gid'].astype(int)
    fr = fr.loc[fr['gid'].isin(subset_cells['c_gid'].tolist()),:]
    fr = fr.reset_index(drop=True)
    return fr

def get_ina(sim_f,subset_cells):
    sim = Simulation(sim_f)
    report = sim.report('NaCurrentAllCmps', source='h5')
    ina_data = report.get()
    ina_data = ina_data.reset_index()
    ina_data = ina_data.loc[:,['time'] + subset_cells['c_gid'].tolist()]
    ina_t = ina_data.transpose()
    ina_t.columns = ina_t.loc['time',:]
    ina_t.drop(ina_t.index[0], inplace=True)

    ina_t = ina_t.reset_index()

    return ina_t

def get_atp(sim_f,subset_cells,duration_in_seconds):
    sim = Simulation(sim_f)
    report = sim.report('ATPConcAllCmps', source='h5')
    atp_data = report.get()
    atp_data = atp_data.reset_index()
    atp_data = atp_data.loc[:,['time'] + subset_cells['c_gid'].tolist()]
    atp_t = atp_data.transpose()
    atp_t.columns = atp_t.loc['time',:]
    atp_t.drop(atp_t.index[0], inplace=True)

    atp_t = atp_t.reset_index()
    
    atp_mean = atp_t.loc[:,['gid']].copy()
    
    for i in range(duration_in_seconds):
        atp_mean['atp_meanconc_'+str(i+1)]  = atp_t.iloc[:,(1000*i+1):(1000+1000*i+1)].mean(axis=1)

    return atp_mean

def calc_atp_scirep(ina_t, vols_areas,Avogadro,Faraday,duration_in_seconds):
    combo = pd.merge(ina_t, vols_areas, left_on='gid',right_on='gid',how="inner")    
    atp_calc = combo.loc[:,['gid','cells_volumes_um3','cells_areas']].copy()

    for i in range(duration_in_seconds):
        atp_calc['atp_'+str(i+1)] = 1e-8*combo['cells_areas']*(Avogadro/Faraday)*1e-3*((combo.iloc[:,(10000*i+1):(10000+10000*i+1)].sum(axis=1))/3)
        atp_calc['inaSum_'+str(i+1)] = combo.iloc[:,(10000*i+1):(10000+10000*i+1)].sum(axis=1)

    return atp_calc

print("get cells subset")
subset_cells = get_cells(circuit_f,percent_gids,onepercent_gids_f,tenpercent_gids_f)
print("get volumes and areas")
vols_areas = get_volumes_areas(vols_areas_f,subset_cells)
print("get mean freq")
fr = get_mean_freq(mean_freq_f,subset_cells)
print("get ina")
ina_t = get_ina(sim_f,subset_cells)
print("get_atp")
atp_mean = get_atp(sim_f,subset_cells,duration_in_seconds)
print("calc_atp_scirep")
atp_calc = calc_atp_scirep(ina_t, vols_areas,Avogadro,Faraday,duration_in_seconds)
print("merges2out")
def merges2out(atp_calc,fr,subset_cells,atp_mean):
    combo1 = pd.merge(atp_calc, fr, left_on='gid',right_on='gid',how="inner")
    combo1 = combo1.reset_index(drop=True)
    
    subset_cells = subset_cells.rename(columns={'c_gid':'gid'})
    combo2 = pd.merge(combo1, subset_cells, left_on='gid',right_on='gid',how="inner")
    combo2 = combo2.reset_index(drop=True)

    combo3 = pd.merge(combo2, atp_mean, left_on='gid',right_on='gid',how="inner")
    combo3 = combo3.reset_index(drop=True)

    combo3.to_csv(out_f_name, sep='\t')

    return

merges2out(atp_calc,fr,subset_cells,atp_mean)
print("finished!")

 










