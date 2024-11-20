# metabolism + ndam + steps
#!/bin/env python

import logging
import numpy as np
import time
import textwrap
from collections import defaultdict
from contextlib import contextmanager
import steps 
import steps.model as smodel
import steps.geom as stetmesh
import steps.rng as srng
import steps.utilities.meshio as meshio
import steps.utilities.geom_decompose as gd
import os as os
import pickle
import math

from neurodamus import Neurodamus
from neurodamus.core import ProgressBarRank0 as ProgressBar
from neurodamus.utils.logging import log_stage
from mpi4py import MPI

comm = MPI.COMM_WORLD

np.set_printoptions(threshold=10000, linewidth=200)

ELEM_CHARGE = 1.602176634e-19
STEPS_USE_MPI = True
REPORT_FLAG = True

#################################################

#### for Julia
import julia
from julia.api import Julia
#jl = Julia() ##jl = Julia(compiled_modules=False)
import diffeqpy
from diffeqpy import de

#We can directly define the functions in Julia. 
#This will allow for more specialization and could be helpful to increase the efficiency over the Numba version for repeat or long calls. 
#This is done via julia.Main.eval:
from julia import Main ##https://stackoverflow.com/questions/57441384/trouble-getting-differential-equation-to-solve-via-diffeqpy

_timings = defaultdict(float)

@contextmanager
def timer(var=None):
    start = time.time()
    yield
    elapsed = time.time() - start
    if var is None:
        print("The function took %g secs." % (elapsed,))
    else:
        _timings[var] += elapsed

#################################################
# Simulate one molecule each 10e9
######## QUESTION ########
# T in ms
DT = 0.025  #ms i.e. = 25 usec which is timstep of ndam
SIM_END = 1000.0 #1000.0 #ms
DT_s = DT / 1000 #because steps_sim.run(t / 1000)  [DT_s] = sec

# In steps use M/L and apply the SIM_REAL ratio
CONC_FACTOR = 1e-9 ######## QUESTION ######## 

AVOGADRO = 6.02e23
COULOMB = 6.24e18
CA = COULOMB/AVOGADRO*CONC_FACTOR*DT_s

#########################################################################
## to calculate mean of processes: https://github.com/openai/baselines/blob/master/baselines/common/mpi_moments.py 
#def mpi_mean(x, axis=0, comm=None, keepdims=False):
#    x = np.asarray(x)
#    assert x.ndim > 0
#    if comm is None: comm = MPI.COMM_WORLD # check indents !!!!
#    xsum = x.sum(axis=axis, keepdims=keepdims)
#    n = xsum.size
#    localsum = np.zeros(n+1, x.dtype)
#    localsum[:n] = xsum.ravel()
#    localsum[n] = x.shape[axis]
#    # globalsum = np.zeros_like(localsum)
#    # comm.Allreduce(localsum, globalsum, op=MPI.SUM)
#    globalsum = comm.allreduce(localsum, op=MPI.SUM)   
#    return(globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n])
    
# alternative from Nicolas: tet_currents_mean = tet_currents_tot / steps.mpi.nhosts
#########################################################################

class Geom:
    meshfile = 'box.msh'
    compname = 'extra'
    
#class Na:
#    name = 'Na'
#    conc_0 = 140  # (mM/L) i.e. mM to be correct in definitions because mM is mmol/L
#    diffname = 'diff_Na'
#    diffcst = 2e-9
#    current_var = 'ina' # 'ina' - current
#    charge = 1 * ELEM_CHARGE
#    e_var = 'ena' #added by Polina, 28nov2019
#    nai_var = 'nai' #added by Polina, 28nov2019 ('nai' - internal concentration)
    
class Na:
    name = 'Na'
    conc_0 = 5  # (mM/L)
    base_conc=140 #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = 'diff_Na'
    diffcst = 2e-9
    current_var = 'ina'
    charge = 1 * ELEM_CHARGE
    e_var = 'ena' #added by Polina, 28nov2019
    nai_var = 'nai' #added by Polina, 28nov2019 ('nai' - internal concentration)
    

class K:
    name = 'K'
    conc_0 = 2 #3 it was 3 in Dan's example  # (mM/L)
    base_conc= 2 #3 it was 3 in Dan's example #sum here is 6 which is probably too high according to Magistretti #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = 'diff_K'
    diffcst = 2e-9
    current_var = 'ik'
    charge = 1 * ELEM_CHARGE


class Ca: # need to check conc numbers for it
    name = 'Ca'
    conc_0 = 2  # (mM/L)
    base_conc=2 #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = 'diff_Ca'
    diffcst = 2e-9
    current_var = 'ica'
    charge = 2 * ELEM_CHARGE

class Cl: # need to check conc numbers for it
    name = 'Cl'
    conc_0 = 5  #3 (mM/L)
    base_conc=120 #137 mM # Novel determinants of the neuronal Cl− concentration Eric Delpire 2014 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4215762/ #560 #Dan had it as 560 #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = 'diff_Cl'
    diffcst = 2e-9
    current_var = 'icl'
    charge = -1 * ELEM_CHARGE

    
    
class ATP:
    name = 'ATP'
    conc_0 = 0.1  
    base_conc= 2.2 #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = 'diff_ATP'
    diffcst = 2e-9
    charge = -3 * ELEM_CHARGE
    atpi_var = 'atpi'
class ADP:
    name = 'ADP'
    conc_0 = 0.0001  
    base_conc= 0.0063 #base_conc+conc_0 is the real concentration, conc_0 is the extracellular amount simulated
    diffname = 'diff_ADP'
    diffcst = 2e-9
    charge = -2 * ELEM_CHARGE
    adpi_var = 'adpi'


class Volsys0:
    name = 'extraIons'
    specs = (Na,Ca,K,Cl) ### !!!!!!!!!

#################################################
# METABOLISM Model Build
#################################################

def gen_metabolism_model():

    
    ##### ATTENTION!!!! The code below is NOT commented!!!!! It is executable by Julia language which is called from this python interface code below #####
    
    metabolism = Main.eval("""  # !!! remove # from here, it was temporary measure for readability 
    function metabolism(du,u,p,t)
        #metabolism + blood flow model
            
    R = 8.31 # J/(K*mol)
    T = 310.0 # Kelvin, temperature 310K is 37 C 
    F = 96.485 # 96485.0  C/mol  # FARADAY = 96485.309 (coul) in Somjen2008
    #Fglut = 96485.0  # C/mol  is for V but use 96.485  for mV
    # 8.31*310.0 /96.485 # 310K is 37 C # like this because of mV and mM


    #Calvetti2018, for O2
    #V_oxphos_n = 8.18 # mM/s # 0.00818 #mM/ms # 
    #V_oxphos_a = 2.55 # mM/s # 0.00255 #mM/ms # 
    #K_oxphos_n = 1.0 #mM
    #K_oxphos_a = 1.0 #mM
    #mu_oxphos_n = 0.01
    #mu_oxphos_a = 0.01
    #nu_oxphos_n = 0.10
    #nu_oxphos_a = 0.10


    #LacTr Calvetti2018
    TaLac = 66.67 # mM/s # 0.06667 #mM/ms#
    KaLac = 15.0 #0.4 # mM  ##################### KmMCT4 = 15-30 mM (astrocytes: MCT4, MCT1. KmMCT1=3-5 mM) - Heidelberger ... Byrne book From Molecules to Networks.... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # volume fractions and blood related parameters (table 1) Calvetti2018
    eto_n =  1.0 #0.4 # volume fraction neuron
    eto_a =  1.0 #0.3 # volume fraction astrocyte
    eto_ecs =  1.0 #0.3 # volume fraction ecs
    eto_b = 1.0 #0.04 # volume fraction blood

    Hct = 0.45 # is the volume percentage (vol%) of red blood cells in blood.
    Hb = 5.18
    KH = 0.0364 #36.4*10^(-3) mM

    C_Glc_a = 5.0 # mM arterial concentration glucose  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    C_Lac_a = 1.1 # mM arterial concentration lactate
    C_O_a = 9.14 # mM arterial concentration oxygen
    Fr_blood = 2/3 # "The numerical values used for the blood flow and the mixing ratio F are 0.9 l/min and 2/3, respectively" from Bayesian flux balance analysis applied to a skeletal muscle metabolic model 2007 Heino .. Calvetti ..

    # table 2 Calvetti2018
    TbGlc = 0.02 # mM/s #0.00002 #mM/ms #
    TbLac = 0.17 # mM/s # 0.00017 #mM/ms# 
    lambda_b_O2 = 0.04 # mM^(1-k)/s # 0.00004 #mM^(1-k)/ms #
    TnGlc = 83.33 # mM/s # 0.08333  #mM/ms# 
    TnLac = 66.67 # mM/s # 0.06667  #mM/ms# 
    lambda_n_O2 = 0.94 # 1/s # 0.00094 #1/ms #
    TaGlc = 83.33 # mM/s # 0.08333 #mM/ms#
    lambda_a_O2 = 0.68 # 1/s # 0.00068 #1/ms #

    tau = 1000.0 # Calvetti2018  #tau = 1.0 # if all other things are in ms, actually, it's more complicated, so even in ms we need it 1000 # Calvetti2018

    KbGlc = 4.6 #0.6- worked #5.0 #4.60 # mM
    KbLac = 5.00 # mM
    KnGlc = 5.00 # mM
    KnLac = 0.4 # mM
    KaGlc = 12500.0 # mM

    V_Cr_n = 16666.67 # mM/s # 16.66667 #mM/ms #
    V_Cr_a = 16666.67 # mM/s # 16.66667 #mM/ms #
    K_Cr_n = 495.00 # mM
    K_Cr_a = 495.00 # mM
    mu_Cr_n = 0.01
    mu_Cr_a = 0.01

    V_PCr_n = 16666.67 # mM/s # 16.66667 #mM/ms #
    V_PCr_a = 16666.67 # mM/s # 16.66667 #mM/ms #
    K_PCr_n = 528.00 # mM
    K_PCr_a = 528.00 # mM
    mu_PCr_n = 100.00
    mu_PCr_a = 100.00

    #Calvetti2018
    Cm = 1.0 # uF/cm2 Calvetti2018 capacitance
    phi = 3.0 # 1/ms Calvetti2018 time constant
    # Conductance param from Calvetti2018 and Cressman2011
    gNa = 100.0 # mS/cm2
    gK = 40.0 # mS/cm2
    gNa0leak = 0.0175 # mS/cm2
    gK0leak = 0.05 # mS/cm2
    gCl = 0.05 # mS/cm2

    # NKA Calvetti2018; in Cressman2011
    beta = 1.33 # Calvetti2018; in Cressman2011 it was set to 7.0
    #ksi_rest = 0.06 #0.15 #0.06 # 0.06 is for resting system # 2.5 #  Calvetti2018; external stimuli, 2.5 should give 90 Hz # this can be decreased for smaller Hz
    #ksi_act =  2.5 #  Calvetti2018; external stimuli, 2.5 should give 90 Hz # this can be decreased for smaller Hz
    gamma = 0.0445 # Calvetti2018 mM*cm2/uC
    #rho=1.25 # Cressman2011
    #epsilon=1.333333333 # Cressman2011
    #kbath=4.0 # Cressman2011
    #glia=66.666666666 # Cressman2011
    rho=13.83 # mM/s # Calvetti2018
    epsilon=9.33 # 1/s # Calvetti2018
    kbath=6.3 # mM # Calvetti2018
    glia=20.75 # mM/s # Calvetti2018

    s_metabo_atpase = 0.15 # Calvetti2018 tuned this param from OGI
    H1_metabo_atpase = 0.071667 # mM/s # 4.3 #mM/min
    sigma_atpase = 103.0 # assumed by Calvetti2018
        
    mu_pump_ephys = 0.1 # Calvetti2018
    mu_glia_ephys = 0.1 # Calvetti2018


    #########################################################################################
    ############################# Glycolysis astrocyte ######################################
    #########################################################################################


    #GLCtr - see below
    # based on Berndt2015
    #VmaxGlcTr_n = 0.72
    #KmGlcGlcTr_n = 1.2 #2.87 #-Berndt2015 #1.0 #
    #KmGlcExt_GlcTr_n = 1.2 #2.87 #-Berndt2015

    #HK1
    #mostly based on Mulukutla2015
    # Mulukutla 2015 HK1 (brain is HK1)
    #partial rapid equilibrium random bi bi mechanism with the assumption that all the steps in the mechanism, 
    #except for the reactive-ternary complexes, are fast reactions. The inhibitions by g6p, glucose-1,6-phosphate (g16bp), 
    #2,3-bisphosphoglycerate (2,3bpg) and glutathione (gsh) were modeled as mixed type of inhibition affecting both the activity (Vmax) 
    #as well as the affinity (KM) of the enzyme for glucose.

    # HK astrocyte
    VmfHK =  180.0*0.0005 #180.0*0.0014 #-worked  #959.0/3600.0 #mM/s # 959mM/h ### Mulukutla 2014 HK2: 6380.0/3600.0 # mM/s  #6380.0 # mM/h !!!!!!!!!!!!!
    VmrHK = 1.16*0.0005 # 1.16*0.0014 # -worked #6.18/3600.0 #mM/s #6.18 mM/h ### Mulukutla 2014 HK2: 41.0/3600.0 # mM/s # 41 mM/h !!!!!!!!!!!!!!!!!

    KATPHK = 1.0 #0.68 #-worked #1.0 mM -Mulukutla2015  # 0.4 mM -Garfinkler1987   !!!!!!!!!!!!!!!!!
    KGLCHK = 1.0 #0.1 #0.053 #-worked  #0.1 mM -Mulukutla2015 # 0.049 mM -Garfinkler1987 # 0.04 mM -Gerber1974    !!!!!!!!!!!!!!!!!

    KiADPHK = 1.45 #1.0#-worked  # mM  -Mulukutla2015 # 1.45 mM -Garfinkler1987  !!!!!!!!!!!!!!!!!

    KG6PHK =  0.47 #0.334 #0.47#-worked  # mM  -Mulukutla2015   ######################################################### if GLC grows with high G6P -> set it to 4.7 #mM
    KiG6PHK = 0.02 #1.0#-worked  #0.074 #0.074 # mM -Garfinkler1987 #0.02 # mM -Mulukutla2015 ######################################################### if GLC grows with high G6P (more fine grained regulation than KG6PHK) -> increase it 

    KiG16BP_HK = 0.03 #mM -Mulukutla2015 
    KiGSH_HK = 3.0 #1.0 #3.0 #mM Mulukutla2015

    KiATPHK = 2.06 #2.55 # mM -Garfinkler1987 #2.06 mM -Gerber1974 # 1.0 # 2.06 #20.0 #1.0  # 1.0 #mM  # not given in Mulukutla2014! set to KATPHK try also 2.06 # mM  from their ref Gerber1974  # can be also 1.0 based on their ref1 from supp
    KiBPG23 = 4.0 #2.0 #4.0 #mM Mulukutla2014 HK2
    ######

    #PGI a
    #mostly based on Mulukutla2015
    VmfPGI = 1.2*2400.0/3600.0 #1.5*2400.0/3600.0 #48000.0/3600.0 # 2400.0/3600.0 #mM/s # 2400 mM/h Mulukutla2015  ###48000.0/3600.0 # mM/s # 4.8*(10^4) mM/h -Mulukutla2014
    VmrPGI = 1.2*2000.0/3600 #1.5*2000.0/3600 #2000.0/3600.0 #40000.0/3600.0 # 2000.0/3600.0 #mM/s 2000.0 mM/h Mulukutla2015 ### 40000.0/3600.0 # mM/s # 4.0*(10^4) mM/h -Mulukutla2014
    KfPGI = 0.5 #0.6 #1.2 #0.96 #0.7 #0.96 #mM ###0.3 #mM-Mulukutla2014 # 0.7 
    KrPGI = 0.1 #0.123 #0.09 #0.13 #0.14 #0.3 #0.123 #mM-Mulukutla201  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #  PFK 1 
    #from Mulukutla2014,2015
    VmfPFK_a = 0.00016*822.0  #0.00018*822.0 #-Mulquiney kcat # 1.8*263.0/3600.0  #2.0*263.0/3600.0 # mM/s # 263.0 # mM/h -Mulukutla2015 # 1550.0/3600.0  -Mulukutla2014
    VmrPFK_a = 0.00016*36.0 #0.00018*36.0 #-Mulquiney kcat #1.8*11.53/3600.0  #2.0*11.53/3600.0 # mM/s # 11.53 # mM/h -Mulukutla2015 # 67.8/3600.0  -Mulukutla2014
    #### activ ####
    ############################################################################
    KF16BPPFK = 0.008 #0.0055 #0.01 #0.5 #-Mulquiney #1.0 #0.05 #-worked #0.1 #0.3 # mM # 0.65  -Mulukutla2014
    #FBP_a = 1.52 #-worked #0.0723 # Lambeth # Jay Glia expand  # 1.52 #mM -Park2016 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KAMPPFK =  0.001 #0.005 #0.03#-worked #0.3 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #AMP_a = 0.01 #2e-5 #0.03 #-Mulukutla2015 #2e-5 # Lambeth # 0.01 # 2e-5 # 1e-5 # from 1e-5 to 0.05
    KADPPFK = 0.03 #0.03 #0.54 # mM
    #ADP_a = 0.03 #0.1 # 0.03 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KG16BPPFK = 0.005 #0.1 # mM
    #GBP = 0.01 #0.6 #0.3 #0.1 #mM  #for now fixed; check value # G16bp #Quick1974: 10 μm and 600 μm #C_g16bp_a = 0.1 # mM #GBP
    KF26BPPFK = 0.0042 #-Berndt #0.0055 #0.05 #0.0055 #worked #5.5e-3 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #f26bp_a0 = 0.015  #0.005-0.025 #mM #Mulukutla supp fig plot 6
    ##############################################################################
    #### inhib ####
    #####################################################################  
    KATPPFK = 1.5 #1.45 #1.4 #0.25 #0.1 #0.068 #6.8e-2 #mM
    #ATP_a = 1.4 #2.17  # or 1.4 ?
    KATP_minor_PFK = 0.2 #0.01#-worked #0.1 #mM
    #ATPminor=0.105*ATP
    KMgPFK = 0.4 #0.06#-worked #0.1 #0.2 #mM # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #C_Mg_a = 0.369  #0.7 # mM  # 0.369 mM 0.4 mM -Mulquiney1999  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KLACPFK = 30.0 # mM
    #Lac_a = 1.4 #1.3 # mM Calvetti2018; Lambeth   ## 0.602393 # Jay 181130 
    #####################################################################
    KF6PPFK = 0.1 #0.2 #0.25 #0.075 #-Mulquiney 0.06 #0.1 #-worked#0.6 #0.06 #6.0e-2 # mM
    #F6P_a = 0.0969 #mM -Park2016 # 0.228 mM -Lambeth2002 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KPiPFK = 30.0 # mM
    #Pi_a = 4.1 # Lambeth # 40.0 # 4.1 # 31.3 # 4.1 # 4.0 Anderson&Wright 1979 # wide range from 1 to 40 mM
    L_pfk = 2e-3 
    K23BPGPFK = 5.0 #1.44 #0.5 # mM
    #BPG23 = 0.237 #mM -Park2016 #3.1 # mM #for now fixed; check value  #C_23BPG_a = 3.1 # mM #BPG23

    # PFK2 FB
    # only astrocytes !!!
    VfPFK2 = 41.6/3600.0 #300.0/3600.0 #-Mulukutla2015  #41.6/3600.0 #Mulukutla2014 ### 300.0/3600.0 #mM/s # 300 mM/h #Mulukutla2015
    KeqPFK2 = 16.0
    KAKTPFK2 = 0.5
    KmATPPFK2 = 0.15 # mM
    KmF6PPFK2 = 0.032 # mM
    KmF26BPPFK2 = 0.008 # mM
    KmADPPFK2 = 0.062 # mM
    KiATPPFK2 = 0.15 # mM
    KiF6PPFK2 = 0.001 # mM
    KiF26BPPFK2 = 0.02 # mM
    KiADPPFK2 = 0.23 # mM
    KiPEPPFK2 = 0.013 # mM
    Vf26BPase = 13.86/3600.0 #11.78/3600.0 # mM/s #-Mulukutla2014 #13.86/3600.0 # mM/s 13.86 # mM/h #-Mulukutla2015
    KiF6PF26BPase = 25.0e-3 # mM #-Mulukutla2015
    KmF26BPF26BPase = 0.001 #mM #-Mulukutla2015 # 0.001 # mM
    #!!!!!!!!!!!!!!!!!!!!!!!!! REGULATOR OF GLYCOLISYS FLUX !!!!!!!!!!!!!!
    #see in metabolism function: pAKTPFK2 = p[8] #0.1 #0.17 # -Mulukutla2015 p.7,p.12 main text; p.11 fig5 main text: pAKT = 0.1mM;0.25mM;0.35mM;1.0mM <- high pAKT corresponds to increased glycolysis and cell growth;  #0.5 set to be consistent with Mulukutla 2014 Bistability in Glycolysis....  !!!!!!! REGULATOR OF GLYCOLISYS FLUX !!!!!!!


    # ALD             !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #Mulukutla2015
    VmfALD = 0.4*68.0*0.01 #68.0*0.001 # 133.0/3600.0 #1.5*133.0/3600.0 #68.0*0.001 #68.0*0.001  #133.0/3600.0  #675.0/3600.0 # -Mulukutla2014  #133.0/3600.0 #mM/s # 133.0 mM/h #-Mulukutla2015  ###12.0/3600.0 #optimiz due to Vmax = kcat*ConcEnz = 68*ConcEnz # Mulquiney
    VmrALD = 0.4*234.0*0.01 #234.0*0.001 # 457.0/3600.0 #1.5*457.0/3600.0 #234.0*0.001 #234.0*0.001 #457.0/3600.0  #2320.0/3600.0 # -Mulukutla2014  #457.0/3600.0 #mM/s # 457.0 mM/h #-Mulukutla2015 ### 2320.0/3600.0 #optimiz due to Vmax = kcat*ConcEnz = 234*ConcEnz# # Mulquiney
    KmFBPALD = 1.0 #0.01 #0.5 #1.0 #0.05 #mM
    KiFBPALD = 1.0 #0.01 #0.198 #1.0 #0.05  #0.0198 #mM
    KmDHAPALD = 2.0 #0.048 #0.03 #1.7 #2.0 #1.7 #2.0 #0.35 #0.035 #mM
    KiDHAPALD =  2.0 #0.048 #0.03 #1.7  #2.0  #1.7 #2.0 #0.11 #0.011 #mM
    KmGAPALD = 0.3 #0.15 #0.015 #0.08 #0.2 #1.0 #0.189 #mM
    KiBPG23ALD = 0.5 #1.0 #4.0  #1.5 #0.15 #4.0 #1.5 #mM - Mulukutla2015  ### 4.0 possible !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #FBP_a = 1.52 #-worked #0.0723 # Lambeth # Jay Glia expand  # 1.52 #mM -Park2016 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #GAP_a = 0.141 #mM -Park2016  #0.0355 # Lambeth # 0.0574386 # 0.0574386 is from latest Jay's data; it was 0.0046 in Jay Glia expand  !!!!!!!!!!!!!!!!!!!!!!!!!!
    #DHAP_a = 1.63 #mM  -Park2016 # 0.0764 # Lambeth # Jay Glia expand   !!!!!!!!!!!!!!!!!!!!!!!!!!

    # TPI  # fast; DHAP -> GAP (reversible)
    #Mulukutla2015; Mulquiney
    # DHAP -> Gap (reversibe)
    VmfTPI = 100.0*510.0/3600.0 #1280.0*0.01  #510.0/3600.0  #14560.0*0.01 # -Mulquiney #510.0/3600.0 #?   #  Vm depends of enzyme conc
    VmrTPI = 100.0*2760.0/3600.0  #14560.0*0.01 #2760.0/3600.0  #1280.0*0.01 # -Mulquiney #2760.0/3600.0 #?   #  Vm depends of enzyme conc
    KfTPI = 0.2 #0.2 #0.16 #0.1#-worked!!! #0.162 #0.43 #0.162 #mM -Mulukutla # #-Mulquiney is reversed directionality compared to Mulukutla2015
    KrTPI = 0.3 #0.4 #0.3 #0.43 #1.0#-worked!!! #0.43 #0.162 #0.43 #mM #-Mulquiney is reversed directionality compared to Mulukutla2015
    #GAP_a = 0.141 #mM -Park2016  #0.0355 # Lambeth # 0.0574386 # 0.0574386 is from latest Jay's data; it was 0.0046 in Jay Glia expand  !!!!!!!!!!!!!!!!!!!!!!!!!!
    #DHAP_a = 1.63 #mM  -Park2016 # 0.0764 # Lambeth # Jay Glia expand   !!!!!!!!!!!!!!!!!!!!!!!!!!

    # GAPDH a
    #Mulukutla2015
    VmfGAPD = 1.1*5317.0/3600.0 # mM/s
    VmrGAPD = 1.1*3919.0/3600.0 # mM/s

    KNADgapd = 0.045 # mM
    KiNADgapd = 0.045 # mM

    KPIgapd =  4.0 #3.5 #2.5 # mM
    KiPIgapd = 4.0 #3.5 #2.5 # mM

    KGAPgapd = 0.1 #0.08 #0.095 #mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KiGAPgapd = 1.59e-16 # mM
    Ki1GAPgapd = 0.031 # mM

    KNADHgapd = 0.0033 # mM
    KiNADHgapd = 0.01 # mM

    KBPG13gapd = 0.00671 #0.000671 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KiBPG13gapd = 1.52e-18 # mM
    Ki1BPG13gapd = 0.001 # mM
    KeqGAPD = 1.9e-8 
    #NADH_a = 0.0007 #0.075  # Nad/NADH = 670-715 
    #NAD_a = 0.5

    # PGK 
    #Mulukutla2015

    VmfPGK =  0.08*59600.0/3600.0 # mM/s  -Mulukutla2015
    VmrPGK = 0.08*23900.0/3600.0 # mM/s  -Mulukutla2015
    KiADPPGK =  0.08 #mM
    KmBPG13PGK = 0.002 #0.01 #0.05 #0.1 #0.05 #0.04 #0.01 #0.002 # mM-Mulukutla2014
    KiBPG13PGK = 1.0 #0.07 #0.16 #1.6 # mM # 
    KiATPPGK = 0.186 #0.36  #0.186 # mM
    KmPG3PGK = 0.6 #1.1 #0.4 #0.6 #1.1 # mM
    KiPG3PGK = 0.205 #0.4 #0.205 # mM

    #BPG13_a = 0.065 #mM Lambeth
    #PG3_a,PG2_a worked: 0.52, 0.05
    #PG3_a = 0.375 #0.52 #0.0168 ####0.375 #-Park # 0.052 #mM Lambeth  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #PG2_a = 0.00949 #0.02 #0.05 #0.00256 ####0.00949 #mM #-Park #0.02 #mM -Berndt #0.005  #mM Lambeth # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #KmPG3pgm = 0.168 # mM
    #KmPG2pgm = 0.0256 # mM

    # PGM a
    #Mulukutla2015
    VmfPGM = 0.017*795*0.1 #489400.0/3600.0 # mM/s  ### kf=795 s-1
    VmrPGM = 0.017*714*0.1 #439500.0/3600.0 # mM/s  ### kr=714 s-1
    KmPG3pgm = 0.168 # mM
    KmPG2pgm = 0.008 #0.0256 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # ENOL
    #Mulukutla2015
    VmfENOL = 0.08*21060.0/3600.0 # mM/s
    VmrENOL = 0.08*5542.0/3600.0 # mM/s
    KiMgENOL = 0.14 #mM ==KMgENOL
    KmPEPENOL = 0.11 # mM
    KiPEPENOL = 0.11 # mM not given
    KmPG2ENOL = 0.046 # mM 
    KiPG2ENOL = 0.046 # mM not given

    #PK
    #Mulukutla2015
    VmfPK = 3.0*2020.0/3600.0 # mM/s
    VmrPK = 3.0*4.75/3600.0 # mM/s
    KpepPK = 0.02 #0.225 # mM
    KadpPK = 0.474 # mM
    KMgatpPK = 3.0 # mM
    KatpPK = 3.39 # mM
    KpyrPK = 0.4 #4.0 # mM
    KfbpPK = 0.005 #mM
    KgbpPK = 0.1 #mM
    L_pk = 0.389 # mM
    KalaPK = 0.02 #mM
    #PEP_a = 0.0194 # Lambeth # 0.014203938186866 # Jay 181130 this value taken from Jolivet PEPg # 0.0279754 # was working with 0.0170014 # was working with 0.0279754 - Glia_170726(1).mod # was working 0.015 # glia expand in between n and g ### check it
    #PYR_a = 0.15 #0.1–0.2 mM -Lajtha 2007  #0.35 # mM Calvetti2018  # 0.0994 # Lambeth # 0.202024 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #LDH
    #Mulukutla2015
    VmfLDH = 8.0*866.0/3600.0 #mM/s
    VmrLDH = 8.0*217.0/3600.0 #mM/s
    KmPYRldh = 0.3 #0.2 #mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KiPYRldh = 0.3 #0.228 #mM
    KiPYRPRIMEldh = 0.101 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KmNADldh = 0.107 #mM
    KiNADldh = 0.503 #mM
    KmLACldh = 4.2 #10.1 #mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KiLACldh = 4.2 #12.4 #30.0 #mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KmNADHldh = 0.00844 #mM
    KiNADHldh = 0.00245 #mM

    ######################### Mulukutla2015
    #VmfGlcTr = 7.67/3600.0 #mM/s
    #VmrGlcTr = 0.767/3600.0 #mM/s
    #KmfGLCtr = 1.0 #5.0 #1.0 #1.5 #mM #1.0-2.0mM-WayneAlberts2012
    #KmrGLCtr = 25.0 #mM #5.0 #1.0 #1.5 #mM #20-30mM-WayneAlberts2012 
    #from 29jan2020 switched to use Calvetti2018 jGLCtr

    #########################################################################################
    ############################# Glycogenolysis astrocyte ##################################
    #########################################################################################

    ############################ Lambeth2002 Glycogenolysis
    # astrocyte glycogenolysis
    # 2. Glycogen phosphorylase: glycogen <--> Glucose-1-P
    # glial glycogen metabolism
    KeqGPa = 0.42 #ref 28 unitless
    KGLYfGPa = 1.7 # ref 23 Rabbit
    KPiGPa = 4.0 # ref 23
    KiGLYfGPa = 2.0 # ref 23 ##############
    KiPiGPa = 4.7 # ref 23
    KGLYbGPa = 0.15  # ref 23
    KG1PGPa = 2.7 # ref 23
    KiG1PGPa = 10.1 # ref 23
    Vmaxfgpa =  0.25*20.0/60.0  #20.0 # 0.02 # M/min ref 55 Rabbit # 20.0 # 
    Vmaxrgpa = (Vmaxfgpa*KGLYbGPa*KiG1PGPa) / (KiGLYfGPa*KPiGPa*KeqGPa)

    #GLY GPb 
    # 2. Glycogen phosphorylase: glycogen <--> Glucose-1-P
    KeqGPb = 0.42
    KPiGPb = 0.2 
    KiPiGPb = 4.6 
    KiGLYfGPb = 15.0 
    KG1PGPb = 1.5 
    KiG1PGPb = 7.4 
    KiGLYbGPb = 4.4 
    Vmaxfgpb =  0.25*30.0/60.0  #mM/s 
    Vmaxrgpb = (Vmaxfgpb*KiGLYbGPb*KG1PGPb) / (KiGLYfGPb*KPiGPb*KeqGPb)
    KAMPGPb = 1.9e-6 
    AMPnHGPb = 1.75

    #PGLM
    # 3. Phosphoglucomutase and phosphoglucoisomerase ; Glucose-1-P <--> Glucose-6-P
    KeqPGLM = 16.62 # ref 28 unitless
    KG1PPGLM = 0.05 #0.063 # ref 30 Rabbit !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KG6PPGLM = 0.7 #0.03  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Vmaxfpglm = 0.08*480.0/60.0  #0.09*480.0/60.0 #480.0/60.0 #-worked # 0.48 # M/min ref 55 Pig 
    Vmaxrpglm = (Vmaxfpglm*KG6PPGLM) /(KG1PPGLM*KeqPGLM)

    # Glycogen synthase; udpgluco + GLY ⇒ 2*GLY
    kL2 = 3.33 # 200/60 #200.0 # Jay expand ### was 3.4 ini   # check
    kmL2 = 0.57
    k_L2 = 0.33 # 20/60  #20.0 # Jay expand ## was 0.34 ini  # check
    km_L2 = 1.4
       
    # see ref in code.. Basal .. 2001
    #VmaxGS = 2.85
    #KmGS = 0.5

    # UDP-glucose pyrophosphorylase; G1P + UTP ⇒ UDPgluco +2*Pi  # rates adapted from Hester and Raushel 1987 and AndersonWright1980
    VmaxfUDPGP = 150.0/(4*60.0)  #/4 because in HR1987 Vf = Vr/4 #150.0/60.0  #-AndersonWright1980 #2.5/60.0 # 2.5 # min-1 Hester and Raushel 1987  # ### check it
    VmaxrUDPGP = 150.0/60.0  #150.0/60.0  #-AndersonWright1980 #10.0/60.0 #10.0 # min-1  Hester and Raushel 1987  ## check it
    KutpUDPGP = 0.05 ##0.06 # mM or 0.061 ### 0.05 #mM-AndersonWright1980
    Kg1pUDPGP = 0.1 #mM-AndersonWright1980 ##0.016 #0.013 # mM or 0.016
    KpiUDPGP = 0.2 #mM-AndersonWright1980 #0.22 #0.25 # mM or 0.22
    KglucoUDPGP = 0.05 #mM-AndersonWright1980#0.18 #0.07 #0.07 # mM or 0.18 

    # PP1 active conc  # Jay Newest
    kmaxd_PP1 = 3.2e-03 #mM
    kmind_PP1 = 2e-06 #mM
    kd_mg_PP1 = 1.0 #?5 mM 
    #k_a  = 1.0 #/min*mM 
    # ka = k_a/kd

    #GSAJAY
    st_GSAJay = 0.003
    kg8_GSAJay = 5.0
    kmg8_GSAJay = 0.00012
    s1_GSAJay = 100.0
    kg2_GSAJay = 0.5
    kg7_GSAJay = 20.0
    kmg7_GSAJay = 0.015
    kg2_GSAJay = 0.5

    # Phosphorylase kinase  act
    kg3_PHKact = 20.0
    kt_PHKact = 0.0025
    kmg3_PHKact = 0.004
    kg4_PHKact = 5.0
    kmg4_PHKact = 0.0011

    # PHK GPb->GPa
    kg5_PHK = 20.0
    pt_PHK = 0.007 #0.07
    kmg5_PHK = 0.01 # check it
    s1_PHK = 100.0
    kg2_PHK = 0.5
    kg6_PHK = 5.0
    kmg6_PHK = 0.005 # check it
    s2_PHK = 0.001
    kgi_PHK = 10.0
    kmaxd_PHK = 3.2e-03 #mM
    kmind_PHK = 2e-06 #mM
    kd_mg_PHK = 1.0 #?5 mM 

    # Protein kinase A (=cAMP-dependent protein kinase); VPKA(), PKAb + 4*cAMP -> R2CcAMP4 + 2*PKAa by Jay, Xu-Gomez # or it was PKAb + ATP ⇒ PKAa + ADP
    kgc1_PKA12 = 1e-6
    k_gc1_PKA12 = 1e-2
    kgc2_PKA12 = 1e-6
    k_gc2_PKA12 = 1e-2


    #########################################################################################
    ############################# PPP astrocyte #############################################
    #########################################################################################
    ############
    ##########################################################################
    # PPP: Stincone 2015; Nakayama 2005; Sabate 1995 rat liver; Kauffman1969 mouse brain; Mulukutla BC, Yongky A, Grimm S, Daoutidis P, Hu W-S (2015) Multiplicity of Steady States in Glycolysis and Shift of Metabolic State in Cultured Mammalian Cells. PLoS ONE; Cakir 2007 
    # check directions
    # Glucose 6-phosphate Dehydrogenase (G6PDH):  Glucose 6-phosphate(G6P) + NADP+(NADP) ↔ 6-phospho-glucono-1,5-lactone(GL6P) + NADPH + H+ ### from Sabate1995
    # ordered sequenttial bi-bi irreversible mechanism
    Vmax1G6PDHppp_a = 5.9e-06 # mM
    KiNADPG6PDHppp_a = 0.009 #mM
    Kg6pG6PDHppp_a = 0.036 #mM
    KNADPG6PDHppp_a = 0.0048 #mM
    KiNADPhG6PDHppp_a = 0.0011 #mM

    # 6-Phosphogluconolactonase (6PGL): 6-Phosphoglucono-1,5-lactone(GL6P) + H2 O → 6-phosphogluconate(GO6P) ### from Sabate1995
    Vmax1f6PGLppp_a = 5.9e-06
    Vmax2r6PGLppp_a = 1.232e-09
    KsGL6Pppp_a = 0.08
    KsGO6Pppp_a = 0.08

    # 6-Phosphogluconate Dehydrogenase (6PGDH): 6-Phosphogluconate(GO6P) + NADP+ → ribulose 5-phosphate(RU5P) + CO2 +NADPH+H+ ### from Sabate1995
    # ordered bi-ter sequential mechanism
    V16PGDHppp_a = 4.93e-06
    V26PGDHppp_a = 1.064e-13
    KiNADP6PGDHppp_a = 0.0048
    KGO6P6PGDHppp_a = 0.0292 # K6PGlc
    Kco26PGDHppp_a = 0.034
    Kico26PGDHppp_a = 0.01387
    KiRu5p6PGDHppp_a = 4.48e-08
    KiNADPh6PGDHppp_a = 0.0051
    KNADPh6PGDHppp_a = 0.00022
    KiGO6P6PGDHppp_a = 0.002176 # was a mistype in paper, inferred by analogy ### from Sabate1995 not clear is K=Ki and why Keq is there   ############################################################## check it!!!!!
    KNADP6PGDHppp_a = 0.0135
    KRu5p6PGDHppp_a = 0.02
    ########## check it!!!!

    # Ribose Phosphate Isomerase (RPI): Ribulose 5-phosphate(RU5P) ↔ ribose 5- phosphate(R5P)
    # Michaelian reversible competitively inhibited by GO6P
    V1rpippp_a = 5.9e-06
    V2rpippp_a = 1.1225e-05
    Kru5prpippp_a = 0.78 # mM
    Kr5prpippp_a = 2.2 #mM
    KiGO6Prpippp_a = 6.0 # mM # try to find mammalian value, as Sabate infered value from other specie
    # check eq for rate


    # Ribulose Phosphate Epimerase (RPE): Ribulose 5-phosphate(RU5P) ↔ xylulose 5-phosphate(X5P)
    V1rpeppp_a = 5.9e-06
    V2rpeppp_a = 8.48e-06
    Kru5prpeppp_a = 0.19
    Kx5prpeppp_a = 0.5
    # check eq for rate

    # Transketolase (TKL1)
    #ping pong: R5P + X5P -> S7P + GAP
    K1tklppp_a = 6.0e-04
    K2tklppp_a = 1.0e-09
    K3tklppp_a = 1.006e-05
    K4tklppp_a = 9.9e-10

    K5tklppp_a = 1.09
    K6tklppp_a = 0.0032
    K7tklppp_a = 15.5

    K8tklppp_a = 0.38
    K9tklppp_a = 0.001548

    K10tklppp_a = 0.38
    K11tklppp_a = 1267.0
    K12tklppp_a = 6050.0
    K13tklppp_a = 0.01
    K14tklppp_a = 1000.0

    K15tklppp_a = 0.01
    K16tklppp_a = 8.6
    K17tklppp_a = 1000.0
    K18tklppp_a = 86400.0
    K19tklppp_a = 8640.0

    Kir5ptklppp_a = 9.8 # mM
    Kix5ptklppp_a = 3.6 # mM
    Kdashr5ptklppp_a = 17.0 # mM
    Kdashx5ptklppp_a = 13.0 # mM

    # Transketolase (TKL2) 
    #ping pong: E4P + X5P -> F6P + GAP
    K20tklppp_a = 5.9e-06
    K21tklppp_a = 2.2e-09
    K22tklppp_a = 3.802e-7
    K23tklppp_a = 5.9e-10

    # Transaldolase (TAL): S7P + GAP ↔ E4P + F6P
    V1talppp_a = 5.9e-06
    V2talppp_a = 1.776e-06
    Kis7ptalppp_a = 0.18
    Kgaptalppp_a = 0.22
    Ks7ptalppp_a = 0.18
    Kf6ptalppp_a = 0.2
    Kie4ptalppp_a = 0.007
    Ke4ptalppp_a = 0.007
    Kif6ptalppp_a = 0.2



    #########################################################################################
    ############################# TCA astrocyte #############################################
    #########################################################################################

    ################

    #PYRH PYRtrcyt2mito_a #Mulukutla2015
    #VmPYRtrcyt2mito_a = 0.0001*1e13/3600.0 # mM/s
    #adapted from Berndt2015
    VmPYRtrcyt2mito_a = 128.0  #0.1*128.0 #- worked
    KmPyrCytTr = 0.15
    KmPyrMitoTr = 0.15 #0.015 #0.15
    #psiPYRtrcyt2mito_a = VmPYRtrcyt2mito_a*(u[23]*C_H_cyt_a - u[120]*C_H_mito_a)/( (1+u[23]/KmPyrCytTr)*(1+u[120]/KmPyrMitoTr) )

    # Pyruvate dehydrogenase complex (PDH); PYRmito + CoAmito + NADmito ⇒ AcCoAmito + CO2 + NADHmito # Berndt 2012
    VmaxPDHCmito_a = 307.0/60.0 # Zhang2018 #13.1 #### # Berndt 2015
    AmaxCaMitoPDH_a = 1.7
    KaCaMitoPDH_a = 0.001
    KmPyrMitoPDH_a = 0.01*0.0252 #0.09
    KmNADmitoPDH_a = 0.01*0.035 #0.036
    KmCoAmitoPDH_a = 0.01*0.0149 #0.0047

    # Citrate synthase: Oxa + AcCoA -> Cit # Berndt 2015
    VmaxCSmito_a = 0.01*1280.0  
    KmOxaMito_a = 0.01*0.0045
    KiCitMito_a = 0.01*3.7
    KmAcCoAmito_a = 0.01*0.005
    KiCoA_a = 0.01*0.025 # Berndt 2012

    # Aconitase; Cit <-> IsoCit # Berndt 2015
    VmaxAco_a = 0.01*1600000.0 #### 
    KeqAco_a = 0.067
    KmCit_a = 0.01*0.48
    KmIsoCit_a = 0.01*0.12


    # NAD-dependent isocitrate dehydrogenase (IDH); ISOCITmito + NADmito  ⇒ AKGmito + NADHmito
    #VmaxIDH_a = 64.0 #### CHECK IT!!!! UNITS!!!! 
    #nIsoCitmitoIDH_a = 1.9
    #KmIsoCitmito1IDH_a = 0.11
    #KmIsoCitmito2IDH_a = 0.06
    #KaCaidhIDH_a = 0.0074
    #nCaIdhIDH_a = 2.0
    #KmNADmitoIDH_a = 0.091 
    #KiNADHmitoIDH_a = 0.041

    ######### IDH_a
    #Wu2007
    VmaxfIDH_a = 425.0 #0.01* #mM/s
    KmaIDH_a = 0.074
    KmbIDH_a = 0.183 # 0.059, 0.055, 0.183
    nH_IDH_a = 2.5 #1.9 # 2.67 #3.0 
    KibIDH_a = 0.0238 # 0.00766, 0.0238
    KiqIDH_a = 0.029
    Ki_atp_IDH_a = 0.091
    Ka_adp_IDH_a = 0.05
    Keq0_IDH_a = 3.5*(10^(-16))
    #Pakg_IDH_a = 1.0 
    #Pnadh_IDH_a = 
    #Pco2tot_IDH_a = 1+ 
    #Pnad_IDH_a = 
    #Picit_IDH_a = 
    Keq_IDH_a = 30.5 #-Mulukutla2015  #Keq0_IDH_a*(Pakg_IDH_a*Pnadh_IDH_a*Pco2tot_IDH_a)/(C_H_mitomatr*Pnad_IDH_a*Picit_IDH_a)



    # aKG dehydrogenase (KGDH); AKGmito + NADmito + CoAmito ⇒ SUCCOAmito + NADHmito      ### Berndt2012
    ### but in future for more details and regulation can consider Detailed kinetics and regulation of mammalian 2- oxoglutarate dehydrogenase 2011 Feng Qi1,2, Ranjan K Pradhan1, Ranjan K Dash1 and Daniel A Beard
    VmaxKGDH_a = 0.01*134.4 #### CHECK IT!!!! UNITS!!!!  # Berndt 2015
    KiCaKGDH_a = 0.01*0.0012 # McCormack 1979, Mogilevskaya 2006
    Km1KGDHKGDH_a = 0.01*2.5 # Berndt 2012
    Km2KGDHKGDH_a = 0.01*0.16 # McCormack 1979
    KiAKGCaKGDH_a = 0.01*1.33e-7 # calculated from McCormack 1979
    KiNADHKGDHKGDH_a = 0.01*0.0045 # Smith 1974
    KmNADkgdhKGDH_a = 0.01*0.021 # Smith 1974
    KmCoAkgdhKGDH_a = 0.01*0.0013 # Smith 1974
    KiSucCoAkgdhKGDH_a = 0.0039 # in brain! Luder 1990

    # Succinyl-CoA synthetase (SCS, STK); SUCCOAmito + ADPmito + Pimito ⇒ SUCmito + CoAmito + ATPmito # bidirect? 
    VmaxSuccoaATP_a = 0.01*19200.0 #0.01*19400.0  #-was #### CHECK IT!!!! UNITS!!!! #19200.0#- Berndt 2015
    AmaxPscs_a = 1.2 # Berndt 2015
    npscs_a = 3.0 # Berndt 2015
    Kmpscs_a = 2.5 # Berndt 2015
    Keqsuccoascs_a = 3.8 #exp(-1.26*1000.0/(R*T))*10.0^(8.0-7.0) #8.0 = pHmito #3.8 # Berndt 2015
    Kmsuccoascs_a =  0.041 #0.086 #
    KmADPscs_a = 0.25 #0.007 #
    KmPimitoscs_a = 0.72 #2.26 #
    Kmsuccscs_a = 1.6 #0.49 #
    Kmcoascs_a =  0.056 #0.036 #
    Kmatpmitoscs_a = 0.017 #0.036 #

    # Succinate dehydrogenase (SDH); SUCmito + Qmito  ⇒ FUMmito + QH2mito  # Mogilevskaya 2006  ## try eq from IvanChang for this reaction
    #kfSDH_a = 10000.0
    #Kesucsucmito_a = 0.01
    #Kmqmito_a = 0.0003
    #krSDH_a = 102.0
    #Kefumfummito_a = 0.29
    #Kmqh2mito_a = 0.0015
    #Kmsucsdh_a = 0.13
    #Kmfumsdh_a = 0.025
    #@reaction_func VSDH(SDHmito,SUCmito,Qmito,FUMmito,QH2mito) = SDHmito*(kfSDH*(SUCmito/Kesucsucmito)*(Qmito/Kmqmito) - krSDH*(FUMmito/Kefumfummito)*(QH2mito/Kmqh2mito)) / (1+(SUCmito/Kesucsucmito)+(Qmito/Kmqmito)*(Kmsucsdh/Kesucsucmito)+(SUCmito/Kesucsucmito)*(Qmito/Kmqmito)+(FUMmito/Kefumfummito)+(QH2mito/Kmqh2mito)*(Kmfumsdh/Kefumfummito) + (FUMmito/Kefumfummito)*(QH2mito/Kmqh2mito))
    ####################################################################################

    # Succinate dehydrogenase (SDH); SUCmito + Qmito  ⇒ FUMmito + QH2mito  # IvanChang 
    #VmaxDHchang_a = 0.28
    #KrDHchang_a = 0.100
    #pDchang_a =0.8

    #Mulukutla2015 
    Vf_SDH_a = 58100.0/3600.0
    Keq_SDH_a = 1.21
    KmSuc_SDH_a = 0.467
    KmQ_SDH_a = 0.48
    KmQH2_SDH_a = 0.00245
    KmFUM_SDH_a = 1.2
    KiSUC_SDH_a = 0.12
    KiFUM_SDH_a = 1.275
    KiOXA_SDH_a = 0.0015
    KaSUC_SDH_a = 0.45 
    KaFUM_SDH_a = 0.375

    # Fumarase (FUM); FUMmito  ⇒  MALmito  based on Berndt 2015
    Vmaxfum_a = 64000000.0 #### CHECK IT!!!! UNITS!!!! 
    Keqfummito_a = 4.4
    Kmfummito_a = 0.014 #0.14
    Kmmalmito_a = 0.03 #0.3
    # Malate dehydrogenase; MALmito + NADmito ⇒ OXAmito + NADHmito  based on Berndt 2015
    VmaxMDHmito_a = 32000.0  #### CHECK IT!!!! UNITS!!!! 
    Keqmdhmito_a = 0.0001
    Kmmalmdh_a = 0.0145 #0.145
    Kmnadmdh_a = 0.006 #0.06
    Kmoxamdh_a = 0.0017 #0.017
    Kmnadhmdh_a = 0.0044 #0.044





    #########################################################################################
    ############################# OXPHOS ETC astrocyte ######################################
    #########################################################################################

    # complex I: NADH-ubiquinone oxidoreductase 
    VmaxC1etc_a = 11.0/60.0 # or 1.1/60.0 #check it
    Ka_C1etc = 1.5e-3 #mM
    Kb_C1etc = 58.1e-3 #mM
    Kc_C1etc = 428.0e-3 #mM
    Kd_C1etc = 519e-3 #mM
    betaC1etc = 0.5
    Gibbs_C1etc = -69.37 *1000.0 #becase R in J/(K*mol)

    # complex III: ubiquinol-cytochrome c oxidoreductase
    VmaxC3etc_a = 22300.0/60.0
    Ka_C3etc = 4.66e-3 #mM
    Kb_C3etc = 3.76e-3 #mM
    Kc_C3etc = 4.08e-3 #mM
    Kd_C3etc = 4.91e-3 #mM
    betaC3etc = 0.5
    Gibbs_C3etc = -32.53 *1000.0 #becase R in J/(K*mol)

    # complex IV: cythochrome c-O2 oxidoreductase
    VmaxC4etc_a = 0.27/60.0
    Ka_C4etc = 680.0e-3 #mM
    Kb_C4etc = 5.4e-3 #mM
    Kc_C4etc = 680.0e-3 #mM
    betaC4etc = 0.5
    Gibbs_C4etc = -122.94 *1000.0 #becase R in J/(K*mol)

    # complex V: ATP-synthase: ADPm + Pim + 3*Hcyto -> ATPm + H2Om + 3*Hm
    #VmaxC5etc_a = 589.0/60.0
    #Ka_C5etc = 10.0e-3 #mM
    #Kb_C5etc = 0.5 #mM
    #Kc_C5etc = 1.0 #mM
    #betaC5etc = 0.5
    #Gibbs_C5etc = 36.03 *1000.0 #becase R in J/(K*mol) # 34-57 kJ/(K*mol) - Miller..Claycomb2013


    #  complex V: ATP-synthase: Wu2007
    #Wu2007
    VmaxC5etc = 0.1*5.95 # mM/s #adj for conc
    naC5etc = 2.5 #3.0 
    deltaG0C5etc = -4.51*1000.0 #J/mol

    pPiC5etc = 0.327 #mM/s permeability for Pi
    pAC5etc = 0.085 #mM/s permeability for nucleotides

    #ANT: adenine nucleotide translocase ATPm + ADPc -> ATPc + ADPm (transport ATP from mito matrix to intermembr space -> to cytosol)
    VmaxANTetc_a = 0.01*523.9/60.0
    K_ADP_ANTetc = 10.0e-3 #mM
    K_ATP_ANTetc = 10.0e-3 #mM
    K_ATPm_ANTetc = 0.025 #mM
    betaANTetc = 0.6


    #########################################################################################
    ############################# Other processes astrocyte #################################
    #########################################################################################

    # Witthoft2013
    JNKCCmax = 0.07557 # mM/s Witthoft2013
    RdcKA = 0.15 # s^(-1)
    gKirS = 144 # pS Witthoft2013 proportionality constant for perisynaptic process KirAS conductance # check units
    EKirProc = 26.797 # mV Nernst for perisyn proc KirAS channels # Witthoft2013
    Cast = 40.0 #pF Witthoft2013 # check units
    gLeakAst = 3.7 # pS Witthoft2013 # check units
    VleakAst = -40.0 # mV Witthoft2013
    gKirV = 25.0 #pS Witthoft2013 # check units
    EKirEndfoot = 31.147 # mV # Witthoft2013 # check units
    JNKAmax = 1.4593 # mM/sec
    KK0a = 16.0 #mM
    KNaiAst = 1.0 #mM

    # NCX
    JmaxNCX = 0.5 

    # Adenylate cyclase; ATP ⇒ cAMP + Pi #Ppi
    VmaxfAC = 0.1*30.0
    KACATP = 0.34
    VmaxrAC = 0.1*1.0
    KpiAC = 0.12
    KcAMPAC = 1.3 #2.3

    # Phosphodiesterase; cAMP ⇒ AMP  # Enzyme assays for cGMP hydrolysing Phosphodiesterases S.D. Rybalkin, T.R. Hinds, and J.A. Beavo https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4091765/
    VmaxPDE = 63000.0/60.0  # mM/min/mg  /mg is ok accorging to paper
    Kmcamppde = 5500.0 # mM # check it

    # 15. Adenylate Kinase ; SigmaATP + SigmaAMP <--> 2*SigmaADP #  Lambeth 2002 JayGliaExpand
    KeqADK = 2.21 # ref 17 Rabbit unitless
    KAMPADK = 0.32 # ref 62 Human
    KATPADK = 0.27 # ref 62
    KADPADK = 0.35 # ref 62
    Vmaxfadk = 880.0/60.0  # 0.88 # M/min ref 55 Pig # 880.0 # 
    Vmaxradk = (Vmaxfadk*(KADPADK^2)) / (KATPADK*KAMPADK*KeqADK)


    #########################################################################################
    ############################# Glutatione astrocyte ######################################
    #########################################################################################

    # Glutatione
    # GPX Mulukutla2015
    V_GPX_a = 0.1*15600.0/3600.0 # = 4.33
    # GSSGR Mulukutla2015
    Vmf_GSSGR_a = 0.1*55000.0/3600
    Vmr_GSSGR_a = 0.1*1.05/3600


    #########################################################################################
    ############################# Glutamate-Glutamine astrocyte #############################
    #########################################################################################

    ###################### EAAT2/GLT1
    Vol_syn = 1e-18 ### L Flanagan2018; 2.0145e-18 L Breslin2018 perisyn ECS, 8.5883e-16 L syn Breslin2018
    SA_ast_EAAT = 2.8274e-13 # m2 Flanagan2018
    Na_syn_EAAT = 140.0 # mM  Flanagan2018
    H_syn_EAAT = 40e-6 # 40 nM = 40*10^-6 mM  Flanagan2018
    H_ast_EAAT = 60e-6 # 60 nM = 60*10^-6 mM  Flanagan2018 
    K_ast_EAAT = 100.0 # mM Flanagan2018 ## this is oversimplific!!! Use kinetic value  u[73] = K_a_in instead  !!!  check its ini value

    alpha_EAAT = 1e-6  #1.9767e-8 #try because it is estimated param  #1.9767e-5 # A/m2 # Flanagan2018 0.0032 # Breslin2018 
    beta_EAAT = 0.0292 # mV^-1 # Flanagan2018
       

    ###################### GDH astrocyte
    #psiGDH_simplif_a: Mulukutla2015 + Botman2014
    VmGDH_a = 1.2/60.0 # mM/s Botman2014
    KeqGDH_a = 0.003 # Mulukutla2015

    KmGLU_GDH_a = 3.5 #Mulukutla2015
    KmNADH_GDH_a = 0.04 #Mulukutla2015
    KmAKG_GDH_a = 1.1
    KiAKG_GDH_a = 0.25
    KiGLU_GDH_a = 3.5
    KiNADH_GDH_a = 0.004
    KiNAD_GDH_a = 1.0

    ###################### Glutamine synthase astrocyte
    # GLN synhase
    VmaxGLNsynth_a = 2.3614/60.0 # mM/s #-Calvetti2011
    KmGLNsynth_a =  2.5 #0.003 #-Calvetti2011
    muGLNsynth_a = 0.01 #100.0 #or 0.01 in analogy with Calvetti2018

    ###################### SNAT GLN export from astrocyte #SNAT GLN transporter astrocyte
    TmaxSNAT_GLN_a = 2.3614/60.0 # mM/s #-Calvetti2011
    KmSNAT_GLN_a = 1.1 #0.07 # or 1.1 mM in Chaudhry1999


    ####################### AAT/GOT astrocyte mito
    VfAAT_a = 3.87*1000000.0/3600.0
    KeqAAT_a = 1.56 

    KmASP_AAT_a = 0.89 
    KmAKG_AAT_a = 3.22
    KmOXA_AAT_a = 0.088
    KmGLU_AAT_a = 32.5

    KiASP_AAT_a = 3.9
    #KiAKG_AAT_a = 0.73
    KiOXA_AAT_a = 0.048
    KiGLU_AAT_a = 10.7
    KiAKG_AAT_a = 26.5

    ######################  Pyruvate carboxylase astrocyte
    #PC: pyruvate carboxilase
    VmPYRCARB_a = 0.001*718.2/60.0
    KmPYR_PYRCARB_a = 0.22 #0.22
    KmCO2_PYRCARB_a = 3.2
    KeqPYRCARB_a = 1.0
    muPYRCARB_a = 0.01




    #########################################################################################
    #########################################################################################
    #########################################################################################
    #########################################################################################
    #########################################################################################


    #########################################################################################
    ############################# Glycolysis neuron #########################################
    #########################################################################################


    # HK neuron
    VmfHK_n =  180.0*0.0005 #180.0*0.0014 #-worked  #959.0/3600.0 #mM/s # 959mM/h ### Mulukutla 2014 HK2: 6380.0/3600.0 # mM/s  #6380.0 # mM/h !!!!!!!!!!!!!
    VmrHK_n = 1.16*0.0005 # 1.16*0.0014 # -worked #6.18/3600.0 #mM/s #6.18 mM/h ### Mulukutla 2014 HK2: 41.0/3600.0 # mM/s # 41 mM/h !!!!!!!!!!!!!!!!!

    KATPHK_n = 1.0 #0.68 #-worked #1.0 mM -Mulukutla2015  # 0.4 mM -Garfinkler1987   !!!!!!!!!!!!!!!!!
    KGLCHK_n = 1.0 #0.1 #0.053 #-worked  #0.1 mM -Mulukutla2015 # 0.049 mM -Garfinkler1987 # 0.04 mM -Gerber1974    !!!!!!!!!!!!!!!!!

    KiADPHK_n = 1.45 #1.0#-worked  # mM  -Mulukutla2015 # 1.45 mM -Garfinkler1987  !!!!!!!!!!!!!!!!!

    KG6PHK_n =  0.47 #0.334 #0.47#-worked  # mM  -Mulukutla2015   ######################################################### if GLC grows with high G6P -> set it to 4.7 #mM
    KiG6PHK_n = 0.02 #1.0#-worked  #0.074 #0.074 # mM -Garfinkler1987 #0.02 # mM -Mulukutla2015 ######################################################### if GLC grows with high G6P (more fine grained regulation than KG6PHK) -> increase it 

    KiG16BP_HK_n = 0.03 #mM -Mulukutla2015 
    KiGSH_HK_n = 3.0 #1.0 #3.0 #mM Mulukutla2015

    KiATPHK_n = 2.06 #2.55 # mM -Garfinkler1987 #2.06 mM -Gerber1974 # 1.0 # 2.06 #20.0 #1.0  # 1.0 #mM  # not given in Mulukutla2014! set to KATPHK try also 2.06 # mM  from their ref Gerber1974  # can be also 1.0 based on their ref1 from supp
    KiBPG23_n = 4.0 #2.0 #4.0 #mM Mulukutla2014 HK2
    ######

    #PGI n
    #mostly based on Mulukutla2015
    VmfPGI_n = 1.2*2400.0/3600.0 #1.5*2400.0/3600.0 #48000.0/3600.0 # 2400.0/3600.0 #mM/s # 2400 mM/h Mulukutla2015  ###48000.0/3600.0 # mM/s # 4.8*(10^4) mM/h -Mulukutla2014
    VmrPGI_n = 1.2*2000.0/3600 #1.5*2000.0/3600 #2000.0/3600.0 #40000.0/3600.0 # 2000.0/3600.0 #mM/s 2000.0 mM/h Mulukutla2015 ### 40000.0/3600.0 # mM/s # 4.0*(10^4) mM/h -Mulukutla2014
    KfPGI_n = 0.5 #0.6 #1.2 #0.96 #0.7 #0.96 #mM ###0.3 #mM-Mulukutla2014 # 0.7 
    KrPGI_n = 0.1 #0.123 #0.09 #0.13 #0.14 #0.3 #0.123 #mM-Mulukutla201  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #  PFK 1  n
    #from Mulukutla2014,2015

    VmfPFK_n = 0.00016*822.0  #0.00018*822.0 #-Mulquiney kcat # 1.8*263.0/3600.0  #2.0*263.0/3600.0 # mM/s # 263.0 # mM/h -Mulukutla2015 # 1550.0/3600.0  -Mulukutla2014
    VmrPFK_n = 0.00016*36.0 #0.00018*36.0 #-Mulquiney kcat #1.8*11.53/3600.0  #2.0*11.53/3600.0 # mM/s # 11.53 # mM/h -Mulukutla2015 # 67.8/3600.0  -Mulukutla2014

    #### activ ####
    ############################################################################
    KF16BPPFK_n = 0.008 #0.0055 #0.01 #0.5 #-Mulquiney #1.0 #0.05 #-worked #0.1 #0.3 # mM # 0.65  -Mulukutla2014
    #FBP_a = 1.52 #-worked #0.0723 # Lambeth # Jay Glia expand  # 1.52 #mM -Park2016 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KAMPPFK_n =  0.001 #0.005 #0.03#-worked #0.3 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #AMP_a = 0.01 #2e-5 #0.03 #-Mulukutla2015 #2e-5 # Lambeth # 0.01 # 2e-5 # 1e-5 # from 1e-5 to 0.05
    KADPPFK_n = 0.03 #0.03 #0.54 # mM
    #ADP_a = 0.03 #0.1 # 0.03 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KG16BPPFK_n = 0.005 #0.1 # mM
    #GBP = 0.01 #0.6 #0.3 #0.1 #mM  #for now fixed; check value # G16bp #Quick1974: 10 μm and 600 μm #C_g16bp_a = 0.1 # mM #GBP
    KF26BPPFK_n = 0.0042 #-Berndt #0.0055 #0.05 #0.0055 #worked #5.5e-3 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #f26bp_a0 = 0.015  #0.005-0.025 #mM #Mulukutla supp fig plot 6
    ##############################################################################

    #### inhib ####
    #####################################################################  
    KATPPFK_n = 1.5 #1.45 #1.4 #0.25 #0.1 #0.068 #6.8e-2 #mM
    #ATP_a = 1.4 #2.17  # or 1.4 ?
    KATP_minor_PFK_n = 0.2 #0.01#-worked #0.1 #mM
    #ATPminor=0.105*ATP
    KMgPFK_n = 0.4 #0.06#-worked #0.1 #0.2 #mM # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #C_Mg_a = 0.369  #0.7 # mM  # 0.369 mM 0.4 mM -Mulquiney1999  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KLACPFK_n = 30.0 # mM
    #Lac_a = 1.4 #1.3 # mM Calvetti2018; Lambeth   ## 0.602393 # Jay 181130 
    #####################################################################

    KF6PPFK_n = 0.1 #0.2 #0.25 #0.075 #-Mulquiney 0.06 #0.1 #-worked#0.6 #0.06 #6.0e-2 # mM
    #F6P_a = 0.0969 #mM -Park2016 # 0.228 mM -Lambeth2002 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    KPiPFK_n = 30.0 # mM
    #Pi_a = 4.1 # Lambeth # 40.0 # 4.1 # 31.3 # 4.1 # 4.0 Anderson&Wright 1979 # wide range from 1 to 40 mM


    L_pfk_n = 2e-3 

    K23BPGPFK_n = 5.0 #1.44 #0.5 # mM
    #BPG23 = 0.237 #mM -Park2016 #3.1 # mM #for now fixed; check value  #C_23BPG_a = 3.1 # mM #BPG23


    # ALD n            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #Mulukutla2015
    VmfALD_n = 0.4*68.0*0.01 #68.0*0.001 # 133.0/3600.0 #1.5*133.0/3600.0 #68.0*0.001 #68.0*0.001  #133.0/3600.0  #675.0/3600.0 # -Mulukutla2014  #133.0/3600.0 #mM/s # 133.0 mM/h #-Mulukutla2015  ###12.0/3600.0 #optimiz due to Vmax = kcat*ConcEnz = 68*ConcEnz # Mulquiney
    VmrALD_n = 0.4*234.0*0.01 #234.0*0.001 # 457.0/3600.0 #1.5*457.0/3600.0 #234.0*0.001 #234.0*0.001 #457.0/3600.0  #2320.0/3600.0 # -Mulukutla2014  #457.0/3600.0 #mM/s # 457.0 mM/h #-Mulukutla2015 ### 2320.0/3600.0 #optimiz due to Vmax = kcat*ConcEnz = 234*ConcEnz# # Mulquiney

    KmFBPALD_n = 1.0 #0.01 #0.5 #1.0 #0.05 #mM
    KiFBPALD_n = 1.0 #0.01 #0.198 #1.0 #0.05  #0.0198 #mM

    KmDHAPALD_n = 2.0 #0.048 #0.03 #1.7 #2.0 #1.7 #2.0 #0.35 #0.035 #mM
    KiDHAPALD_n =  2.0 #0.048 #0.03 #1.7  #2.0  #1.7 #2.0 #0.11 #0.011 #mM

    KmGAPALD_n = 0.3 #0.15 #0.015 #0.08 #0.2 #1.0 #0.189 #mM
    KiBPG23ALD_n = 0.5 #1.0 #4.0  #1.5 #0.15 #4.0 #1.5 #mM - Mulukutla2015  ### 4.0 possible !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # TPI  # fast; DHAP -> GAP (reversible)
    #Mulukutla2015; Mulquiney
    # DHAP -> Gap (reversibe)
    VmfTPI_n = 100.0*510.0/3600.0 #1280.0*0.01  #510.0/3600.0  #14560.0*0.01 # -Mulquiney #510.0/3600.0 #?   #  Vm depends of enzyme conc
    VmrTPI_n = 100.0*2760.0/3600.0  #14560.0*0.01 #2760.0/3600.0  #1280.0*0.01 # -Mulquiney #2760.0/3600.0 #?   #  Vm depends of enzyme conc
    KfTPI_n = 0.2 #0.2 #0.16 #0.1#-worked!!! #0.162 #0.43 #0.162 #mM -Mulukutla # #-Mulquiney is reversed directionality compared to Mulukutla2015
    KrTPI_n = 0.3 #0.4 #0.3 #0.43 #1.0#-worked!!! #0.43 #0.162 #0.43 #mM #-Mulquiney is reversed directionality compared to Mulukutla2015


    # GAPDH n
    #Mulukutla2015
    VmfGAPD_n = 1.1*5317.0/3600.0 # mM/s
    VmrGAPD_n = 1.1*3919.0/3600.0 # mM/s

    KNADgapd_n = 0.045 # mM
    KiNADgapd_n = 0.045 # mM

    KPIgapd_n =  4.0 #3.5 #2.5 # mM
    KiPIgapd_n = 4.0 #3.5 #2.5 # mM

    KGAPgapd_n = 0.1 #0.08 #0.095 #mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KiGAPgapd_n = 1.59e-16 # mM
    Ki1GAPgapd_n = 0.031 # mM

    KNADHgapd_n = 0.0033 # mM
    KiNADHgapd_n = 0.01 # mM

    KBPG13gapd_n = 0.00671 #0.000671 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KiBPG13gapd_n = 1.52e-18 # mM
    Ki1BPG13gapd_n = 0.001 # mM
    KeqGAPD_n = 1.9e-8 
    #NADH_a = 0.0007 #0.075  # Nad/NADH = 670-715 
    #NAD_a = 0.5

    # PGK n
    #Mulukutla2015
    VmfPGK_n =  0.08*59600.0/3600.0 # mM/s  -Mulukutla2015
    VmrPGK_n = 0.08*23900.0/3600.0 # mM/s  -Mulukutla2015
    KiADPPGK_n =  0.08 #mM
    KmBPG13PGK_n = 0.002 #0.01 #0.05 #0.1 #0.05 #0.04 #0.01 #0.002 # mM-Mulukutla2014
    KiBPG13PGK_n = 1.0 #0.07 #0.16 #1.6 # mM # 
    KiATPPGK_n = 0.186 #0.36  #0.186 # mM
    KmPG3PGK_n = 0.6 #1.1 #0.4 #0.6 #1.1 # mM
    KiPG3PGK_n = 0.205 #0.4 #0.205 # mM

     
    # PGM n
    #Mulukutla2015
    VmfPGM_n = 0.017*795*0.1 #489400.0/3600.0 # mM/s  ### kf=795 s-1
    VmrPGM_n = 0.017*714*0.1 #439500.0/3600.0 # mM/s  ### kr=714 s-1
    KmPG3pgm_n = 0.168 # mM
    KmPG2pgm_n = 0.008 #0.0256 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # ENOL n
    #Mulukutla2015
    VmfENOL_n = 0.08*21060.0/3600.0 # mM/s
    VmrENOL_n = 0.08*5542.0/3600.0 # mM/s
    KiMgENOL_n = 0.14 #mM ==KMgENOL
    KmPEPENOL_n = 0.11 # mM
    KiPEPENOL_n = 0.11 # mM not given
    KmPG2ENOL_n = 0.046 # mM 
    KiPG2ENOL_n = 0.046 # mM not given


    #PK n
    #Mulukutla2015
    VmfPK_n = 3.0*2020.0/3600.0 # mM/s
    VmrPK_n = 3.0*4.75/3600.0 # mM/s
    KpepPK_n = 0.02 #0.225 # mM
    KadpPK_n = 0.474 # mM
    KMgatpPK_n = 3.0 # mM
    KatpPK_n = 3.39 # mM
    KpyrPK_n = 0.4 #4.0 # mM
    KfbpPK_n = 0.005 #mM
    KgbpPK_n = 0.1 #mM
    L_pk_n = 0.389 # mM
    KalaPK_n = 0.02 #mM
    #PEP_a = 0.0194 # Lambeth # 0.014203938186866 # Jay 181130 this value taken from Jolivet PEPg # 0.0279754 # was working with 0.0170014 # was working with 0.0279754 - Glia_170726(1).mod # was working 0.015 # glia expand in between n and g ### check it
    #PYR_a = 0.15 #0.1–0.2 mM -Lajtha 2007  #0.35 # mM Calvetti2018  # 0.0994 # Lambeth # 0.202024 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    #LDH n
    #Mulukutla2015
    VmfLDH_n = 8.0*866.0/3600.0 #mM/s
    VmrLDH_n = 8.0*217.0/3600.0 #mM/s

    KmPYRldh_n = 0.3 #0.2 #mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KiPYRldh_n = 0.3 #0.228 #mM

    KiPYRPRIMEldh_n = 0.101 # mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    KmNADldh_n = 0.107 #mM
    KiNADldh_n = 0.503 #mM

    KmLACldh_n = 4.2 #10.1 #mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    KiLACldh_n = 4.2 #12.4 #30.0 #mM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    KmNADHldh_n = 0.00844 #mM
    KiNADHldh_n = 0.00245 #mM


    ######################### Mulukutla2015
    #VmfGlcTr = 7.67/3600.0 #mM/s
    #VmrGlcTr = 0.767/3600.0 #mM/s
    #KmfGLCtr = 1.0 #5.0 #1.0 #1.5 #mM #1.0-2.0mM-WayneAlberts2012
    #KmrGLCtr = 25.0 #mM #5.0 #1.0 #1.5 #mM #20-30mM-WayneAlberts2012 
    #from 29jan2020 switched to use Calvetti2018 jGLCtr

    #########################################################################################
    ############################# PPP neuron ################################################
    #########################################################################################
    ##########################################################################
    # PPP: Stincone 2015; Nakayama 2005; Sabate 1995 rat liver; Kauffman1969 mouse brain; Mulukutla BC, Yongky A, Grimm S, Daoutidis P, Hu W-S (2015) Multiplicity of Steady States in Glycolysis and Shift of Metabolic State in Cultured Mammalian Cells. PLoS ONE; Cakir 2007 
    # check directions
    # Glucose 6-phosphate Dehydrogenase (G6PDH):  Glucose 6-phosphate(G6P) + NADP+(NADP) ↔ 6-phospho-glucono-1,5-lactone(GL6P) + NADPH + H+ ### from Sabate1995
    # ordered sequenttial bi-bi irreversible mechanism
    Vmax1G6PDHppp_n = 5.9e-06 # mM
    KiNADPG6PDHppp_n = 0.009 #mM
    Kg6pG6PDHppp_n = 0.036 #mM
    KNADPG6PDHppp_n = 0.0048 #mM
    KiNADPhG6PDHppp_n = 0.0011 #mM

    # 6-Phosphogluconolactonase (6PGL): 6-Phosphoglucono-1,5-lactone(GL6P) + H2 O → 6-phosphogluconate(GO6P) ### from Sabate1995
    Vmax1f6PGLppp_n = 5.9e-06
    Vmax2r6PGLppp_n = 1.232e-09
    KsGL6Pppp_n = 0.08
    KsGO6Pppp_n = 0.08

    # 6-Phosphogluconate Dehydrogenase (6PGDH): 6-Phosphogluconate(GO6P) + NADP+ → ribulose 5-phosphate(RU5P) + CO2 +NADPH+H+ ### from Sabate1995
    # ordered bi-ter sequential mechanism
    V16PGDHppp_n = 4.93e-06
    V26PGDHppp_n = 1.064e-13
    KiNADP6PGDHppp_n = 0.0048
    KGO6P6PGDHppp_n = 0.0292 # K6PGlc
    Kco26PGDHppp_n = 0.034
    Kico26PGDHppp_n = 0.01387
    KiRu5p6PGDHppp_n = 4.48e-08
    KiNADPh6PGDHppp_n = 0.0051
    KNADPh6PGDHppp_n = 0.00022
    KiGO6P6PGDHppp_n = 0.002176 # was a mistype in paper, inferred by analogy ### from Sabate1995 not clear is K=Ki and why Keq is there   ############################################################## check it!!!!!
    KNADP6PGDHppp_n = 0.0135
    KRu5p6PGDHppp_n = 0.02
    ########## check it!!!!

    # Ribose Phosphate Isomerase (RPI): Ribulose 5-phosphate(RU5P) ↔ ribose 5- phosphate(R5P)
    # Michaelian reversible competitively inhibited by GO6P
    V1rpippp_n = 5.9e-06
    V2rpippp_n = 1.1225e-05
    Kru5prpippp_n = 0.78 # mM
    Kr5prpippp_n = 2.2 #mM
    KiGO6Prpippp_n = 6.0 # mM # try to find mammalian value, as Sabate infered value from other specie
    # check eq for rate

    # Ribulose Phosphate Epimerase (RPE): Ribulose 5-phosphate(RU5P) ↔ xylulose 5-phosphate(X5P)
    V1rpeppp_n = 5.9e-06
    V2rpeppp_n = 8.48e-06
    Kru5prpeppp_n = 0.19
    Kx5prpeppp_n = 0.5
    # check eq for rate

    # Transketolase (TKL1)
    #ping pong: R5P + X5P -> S7P + GAP
    K1tklppp_n = 6.0e-04
    K2tklppp_n = 1.0e-09
    K3tklppp_n = 1.006e-05
    K4tklppp_n = 9.9e-10

    K5tklppp_n = 1.09
    K6tklppp_n = 0.0032
    K7tklppp_n = 15.5

    K8tklppp_n = 0.38
    K9tklppp_n = 0.001548

    K10tklppp_n = 0.38
    K11tklppp_n = 1267.0
    K12tklppp_n = 6050.0
    K13tklppp_n = 0.01
    K14tklppp_n = 1000.0

    K15tklppp_n = 0.01
    K16tklppp_n = 8.6
    K17tklppp_n = 1000.0
    K18tklppp_n = 86400.0
    K19tklppp_n = 8640.0

    Kir5ptklppp_n = 9.8 # mM
    Kix5ptklppp_n = 3.6 # mM
    Kdashr5ptklppp_n = 17.0 # mM
    Kdashx5ptklppp_n = 13.0 # mM

    # Transketolase (TKL2) 
    #ping pong: E4P + X5P -> F6P + GAP
    K20tklppp_n = 5.9e-06
    K21tklppp_n = 2.2e-09
    K22tklppp_n = 3.802e-7
    K23tklppp_n = 5.9e-10


    # Transaldolase (TAL): S7P + GAP ↔ E4P + F6P
    V1talppp_n = 5.9e-06
    V2talppp_n = 1.776e-06
    Kis7ptalppp_n = 0.18
    Kgaptalppp_n = 0.22
    Ks7ptalppp_n = 0.18
    Kf6ptalppp_n = 0.2
    Kie4ptalppp_n = 0.007
    Ke4ptalppp_n = 0.007
    Kif6ptalppp_n = 0.2



    # Glutatione n
    # GPX Mulukutla2015
    V_GPX_n = 0.1*15600.0/3600.0 # = 4.33
    # GSSGR Mulukutla2015
    Vmf_GSSGR_n = 0.1*55000.0/3600
    Vmr_GSSGR_n = 0.1*1.05/3600


    #########################################################################################
    ############################# TCA neuron ################################################
    #########################################################################################


    #PYRH PYRtrcyt2mito_a #Mulukutla2015
    #VmPYRtrcyt2mito_a = 0.0001*1e13/3600.0 # mM/s

    #adapted from Berndt2015
    VmPYRtrcyt2mito_n = 128.0  #0.1*128.0 #- worked
    KmPyrCytTr_n = 0.15
    KmPyrMitoTr_n = 0.15 #0.015 #0.15
    #psiPYRtrcyt2mito_a = VmPYRtrcyt2mito_a*(u[23]*C_H_cyt_a - u[120]*C_H_mito_a)/( (1+u[23]/KmPyrCytTr)*(1+u[120]/KmPyrMitoTr) )


    # Pyruvate dehydrogenase complex (PDH); PYRmito + CoAmito + NADmito ⇒ AcCoAmito + CO2 + NADHmito # Berndt 2012
    VmaxPDHCmito_n = 189.7/3600.0 #Mulukutla2015 #0.1*307.0/60.0 # Zhang2018 #13.1 #### # Berndt 2015
    AmaxCaMitoPDH_n = 1.7 #1.7
    KaCaMitoPDH_n = 0.001
    KmPyrMitoPDH_n = 0.09 #0.1*0.068 #0.0252 #0.01*0.0252 #0.09#-Berndt2012
    KmNADmitoPDH_n = 0.036 #0.1*0.041 #0.035 #0.01*0.035 #0.036#-Berndt2012
    KmCoAmitoPDH_n = 0.0047 #0.1*0.0047 #0.0047 #0.0149 #0.01*0.0149 #0.0047#-Berndt2012


    #Mulukutla2015
    #VfPDH_n = 189.7/3600.0
    #KeqPDH_n = 0.1*12000.0
    #KpyrPDH_n = 0.3 #0.0388
    #KcoaPDH_n = 0.099 #0.0099
    #KnadPDH_n = 0.0607
    #KiAcCoaPDH_n = 0.040
    #KiNADHPDH_n = 0.0402


    # Citrate synthase: Oxa + AcCoA -> Cit # Berndt 2015
    VmaxCSmito_n = 0.01*1280.0  
    KmOxaMito_n = 0.01*0.0045
    KiCitMito_n = 0.01*3.7
    KmAcCoAmito_n = 0.01*0.005
    KiCoA_n = 0.01*0.025 # Berndt 2012

    ## Aconitase; Cit <-> IsoCit # Berndt 2015
    VmaxAco_n = 0.01*1600000.0 #### 
    KeqAco_n = 0.067
    KmCit_n = 0.01*0.48
    KmIsoCit_n = 0.01*0.12

    ##old version IDH
    ## NAD-dependent isocitrate dehydrogenase (IDH); ISOCITmito + NADmito  ⇒ AKGmito + NADHmito
    ##VmaxIDH_n = 64.0 #### CHECK IT!!!! UNITS!!!! 
    ##nIsoCitmitoIDH_n = 1.9
    ##KmIsoCitmito1IDH_n = 0.11
    ##KmIsoCitmito2IDH_n = 0.06
    ##KaCaidhIDH_n = 0.0074
    ##nCaIdhIDH_n = 2.0
    ##KmNADmitoIDH_n = 0.091 
    ##KiNADHmitoIDH_n = 0.041

    ########## IDH_n
    ##Wu2007
    VmaxfIDH_n = 0.01*425.0 #mM/s
    KmaIDH_n = 0.074
    KmbIDH_n = 0.183 # 0.059, 0.055, 0.183
    nH_IDH_n = 2.5 # 1.9 # 2.67 #3.0 
    KibIDH_n = 0.0238 # 0.00766, 0.0238
    KiqIDH_n = 0.029
    Ki_ntp_IDH_n = 0.091
    Ka_ndp_IDH_n = 0.05
    #Keq0_IDH_n = 3.5*(10^(-16))
    #Pakg_IDH_n = 1.0 
    #Pnadh_IDH_n = 
    #Pco2tot_IDH_n = 1+ 
    #Pnad_IDH_n = 
    #Picit_IDH_n = 
    Keq_IDH_n = 30.5 #-Mulukutla2015  #Keq0_IDH_n*(Pakg_IDH_n*Pnadh_IDH_n*Pco2tot_IDH_n)/(C_H_mitomatr*Pnad_IDH_n*Picit_IDH_n)

    ## aKG dehydrogenase (KGDH); AKGmito + NADmito + CoAmito ⇒ SUCCOAmito + NADHmito      ### Berndt2012
    #### but in future for more details and regulation can consider Detailed kinetics and regulation of mammalian 2- oxoglutarate dehydrogenase 2011 Feng Qi1,2, Ranjan K Pradhan1, Ranjan K Dash1 and Daniel A Beard
    VmaxKGDH_n = 0.01*134.4 #### CHECK IT!!!! UNITS!!!!  # Berndt 2015
    KiCaKGDH_n = 0.01*0.0012 # McCormack 1979, Mogilevskaya 2006
    Km1KGDHKGDH_n = 0.01*2.5 # Berndt 2012
    Km2KGDHKGDH_n = 0.01*0.16 # McCormack 1979
    KiAKGCaKGDH_n = 0.01*1.33e-7 # calculated from McCormack 1979
    KiNADHKGDHKGDH_n = 0.01*0.0045 # Smith 1974
    KmNADkgdhKGDH_n = 0.01*0.021 # Smith 1974
    KmCoAkgdhKGDH_n = 0.01*0.0013 # Smith 1974
    KiSucCoAkgdhKGDH_n = 0.0039 # in brain! Luder 1990

    ## Succinyl-CoA synthetase (SCS, STK); SUCCOAmito + ADPmito + Pimito ⇒ SUCmito + CoAmito + ATPmito # bidirect? 
    VmaxSuccoaATP_n = 0.01*19200.0 #0.01*19400.0  #-was #### CHECK IT!!!! UNITS!!!! #19200.0#- Berndt 2015
    AmaxPscs_n = 1.2 # Berndt 2015
    npscs_n = 2.5 #3.0#worked # Berndt 2015
    Kmpscs_n = 2.5 # Berndt 2015
    Keqsuccoascs_n = 3.8 #exp(-1.26*1000.0/(R*T))*10.0^(8.0-7.0) #8.0 = pHmito #3.8 # Berndt 2015
    Kmsuccoascs_n =  0.086 #0.041#worked #0.086 #
    KmADPscs_n = 0.25 #0.007 #
    KmPimitoscs_n = 0.72 #2.26 #
    Kmsuccscs_n = 0.49 #1.6#worked #0.49 #
    Kmcoascs_n =  0.056 #0.036 #
    Kmatpmitoscs_n = 0.017 #0.036 #

    # Succinate dehydrogenase (SDH); SUCmito + Qmito  ⇒ FUMmito + QH2mito  # Mogilevskaya 2006  ## try eq from IvanChang for this reaction
    #kfSDH_n = 10000.0
    #Kesucsucmito_n = 0.01
    #Kmqmito_n = 0.0003
    #krSDH_n = 102.0
    #Kefumfummito_n = 0.29
    #Kmqh2mito_n = 0.0015
    #Kmsucsdh_n = 0.13
    #Kmfumsdh_n = 0.025
    #@reaction_func VSDH(SDHmito,SUCmito,Qmito,FUMmito,QH2mito) = SDHmito*(kfSDH*(SUCmito/Kesucsucmito)*(Qmito/Kmqmito) - krSDH*(FUMmito/Kefumfummito)*(QH2mito/Kmqh2mito)) / (1+(SUCmito/Kesucsucmito)+(Qmito/Kmqmito)*(Kmsucsdh/Kesucsucmito)+(SUCmito/Kesucsucmito)*(Qmito/Kmqmito)+(FUMmito/Kefumfummito)+(QH2mito/Kmqh2mito)*(Kmfumsdh/Kefumfummito) + (FUMmito/Kefumfummito)*(QH2mito/Kmqh2mito))
    ####################################################################################

    # Succinate dehydrogenase (SDH); SUCmito + Qmito  ⇒ FUMmito + QH2mito  # IvanChang 
    #VmaxDHchang_n = 0.28
    #KrDHchang_n = 0.100
    #pDchang_n =0.8

    #Mulukutla2015 
    Vf_SDH_n = 58100.0/3600.0
    Keq_SDH_n = 1.21
    KmSuc_SDH_n = 0.467
    KmQ_SDH_n = 0.48
    KmQH2_SDH_n = 0.00245
    KmFUM_SDH_n = 1.2
    KiSUC_SDH_n = 0.12
    KiFUM_SDH_n = 1.275
    KiOXA_SDH_n = 0.0015
    KaSUC_SDH_n = 0.45 
    KaFUM_SDH_n = 0.375

    ## Fumarase (FUM); FUMmito  ⇒  MALmito  based on Berndt 2015
    Vmaxfum_n = 22100.0/3600.0  #64000000.0 #### CHECK IT!!!! UNITS!!!! 
    Keqfummito_n = 4.4
    Kmfummito_n = 0.014 #0.14
    Kmmalmito_n = 0.03 #0.3


    # Malate dehydrogenase; MALmito + NADmito ⇒ OXAmito + NADHmito  based on Berndt 2015
    VmaxMDHmito_n = 32000.0  #### CHECK IT!!!! UNITS!!!! 
    Keqmdhmito_n = 0.000402 #-Mulukutla2015 #0.0001
    Kmmalmdh_n = 0.0145 #0.145
    Kmnadmdh_n = 0.006 #0.06
    Kmoxamdh_n = 0.0017 #0.017
    Kmnadhmdh_n = 0.0044 #0.044



    #########################################################################################
    ############################# MAS neuron ################################################
    #########################################################################################

    #Berndt2015

    # Cytosolic malate dehydrogenase [60-62] Berndt 2015  Mal_in + NAD_in ↔ Oa_in + NADH_in

    VmaxcMDH_n = 0.3*32000.0 /60000.0  #10000.0 /60000.0 #### CHECK IT!!!! UNITS!!!! #McKenna1933 cytoMDH ~ 0.3*mitoMDH act
    Keqcmdh_n = 0.000402 #0.0001
    Kmmalcmdh_n = 0.077 #0.77
    Kmnadcmdh_n = 0.005 #0.05
    Kmoxacmdh_n = 0.004 #0.04
    Kmnadhcmdh_n = 0.005 #0.05

    #psicMDH_n = VmaxcMDH_n*(u[107]*u[34]-u[108]*u[32]/Keqcmdh_n)/ ((1.0+u[107]/Kmmalcmdh_n)*(1.0+u[34]/Kmnadcmdh_n) + (1.0+u[108]/Kmoxacmdh_n)*(1.0+u[32]/Kmnadhcmdh_n)-1.0) 


    # Cytosolic aspartate aminotransferase [63] Berndt 2015   Asp_cyt + akg_cyt ↔ oa_cyt + glu_cyt
    VmaxcAAT_n = 32.0 /60000 #### CHECK IT!!!! UNITS!!!! 
    KeqcAAT_n = 0.147 # 1.56 Mukukutla2015

    # Mitochondrial aspartate aminotransferase [63] Berndt 2015  ########## compare with Mulukutla2015 AAT  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    VmaxmitoAAT_n = 32.0 /60000 #### CHECK IT!!!! UNITS!!!! 
    KeqmitoAAT_n = 0.147 # 1.56 Mukukutla2015

    # Mitochondrial aspartate aminotransferase AAT Mulukutla2015
    ####################### AAT/GOT astrocyte mito
    #VfAAT_n = 3.87*1000000.0/3600.0
    #KeqAAT_n = 1.56 
    #KmASP_AAT_n = 0.89 
    #KmAKG_AAT_n = 3.22
    #KmOXA_AAT_n = 0.088
    #KmGLU_AAT_n = 32.5
    #KiASP_AAT_n = 3.9
    ##KiAKG_AAT_n = 0.73
    #KiOXA_AAT_n = 0.048
    #KiGLU_AAT_n = 10.7
    #KiAKG_AAT_n = 26.5




    # Aspartate/glutamate carrier [64] Berndt 2015 # ARALAR 
    # Asp_mito + glu_cyt + h_cyt ↔ Asp_cyt + glu_mito + h_mito
    #Vmaxagc_n = 3200.0 /60000 #### CHECK IT!!!! UNITS!!!! 
    ##Keqagc_n =  0.968 #!!!!! this needs to be variable, because it's part of Calcium-regulation of primiting mito activation!!!  # from NEDERGAARD 1991: 7.01/7.24 # hcyt/hext  # was 1.737 in Berndt 2015 - mistake??  # = Hcyt_n/Hext #
    ### u[161] Vmm_n = -140.0 #-0.14 #-140.0 mV
    #Kmaspmitoagc_n = 0.5 #0.05
    #KmgluCagc_n = 2.8
    #Kmaspagc_n = 1.5 #0.5 #0.05
    #KmgluMagc_n = 5.8 #2.8
    ##psiAGC_n = Vmaxagc_n*(u[105]*u[162] - u[109]*u[106]/ (exp(u[161])^(F/(R*T))*  (C_H_cyt_n/C_H_mito_n)) ) / ((u[105]+Kmaspmitoagc_n)*(u[162]+Kmgluagc_n) + (u[109]+Kmaspagc_n)*(u[106]+Kmgluagc_n))
        
    # Aspartate/glutamate carrier Mulukutla2015 # ARALAR 
    VmaxAGC_n = 0.001*24900.0/3600.0
    KeqAGC_n = 0.6
    KiASPmAGC_n = 0.028
    KiASPcAGC_n = 2.8
    KiGLUmAGC_n = 0.18
    KiGLUcAGC_n = 1.6
    KhAGC_n = 10^(-6.5)
    mAGC_n = 1.8



    # Malate/α-ketoglutarate carrier [65, 66] Berndt 2015  
    # Mal_cyt + akg_mito ↔ Mal_mito + akg_cyt
    #Vmaxmakgc_n = 0.5* 32.0 /60000 #### CHECK IT!!!! UNITS!!!! 
    #Kmmalmakgc_n = 0.36 #1.36
    #Kmakgmitomakgc_n = 0.1 #0.2
    #Kmmalmitomakgc_n = 0.71
    #Kmakgmakgc_n = 0.3 #0.1


    # Malate/α-ketoglutarate carrier Mulukutla2015
    # Mal_cyt + akg_mito ↔ Mal_mito + akg_cyt
    Vmax_makgc_n = 3190000.0/3600.0  
    Km_mal_makgc_n = 0.4
    Km_akgmito_makgc_n = 0.17
    Km_malmito_makgc_n = 10.0
    Km_akg_makgc_n = 1.3
        





    ##########################
    # GLUmito - GLUcyto transp
    VmGLUH_n = 3.87*(10^8)/3600.0
        

    ######################  GLN import to neuron # GLN transporter 
    TmaxSNAT_GLN_n = 2.356/60.0 # mM/s #-Calvetti2011
    KmSNAT_GLN_n = 7.0e-5 #-Calvetti2011


    #########################################################################################
    ############################# OXPHOS ETC neuron #########################################
    #########################################################################################

    # complex I: NADH-ubiquinone oxidoreductase 
    VmaxC1etc_n = 11.0/60.0 # or 1.1/60.0 #check it
    Ka_C1etc_n = 1.5e-3 #mM
    Kb_C1etc_n = 58.1e-3 #mM
    Kc_C1etc_n = 428.0e-3 #mM
    Kd_C1etc_n = 519e-3 #mM
    betaC1etc_n = 0.5
    Gibbs_C1etc_n = -69.37 *1000.0 #becase R in J/(K*mol)

    # complex III: ubiquinol-cytochrome c oxidoreductase
    VmaxC3etc_n = 22300.0/60.0
    Ka_C3etc_n = 4.66e-3 #mM
    Kb_C3etc_n = 3.76e-3 #mM
    Kc_C3etc_n = 4.08e-3 #mM
    Kd_C3etc_n = 4.91e-3 #mM
    betaC3etc_n = 0.5
    Gibbs_C3etc_n = -32.53 *1000.0 #becase R in J/(K*mol)

    # complex IV: cythochrome c-O2 oxidoreductase
    VmaxC4etc_n = 0.27/60.0
    Ka_C4etc_n = 680.0e-3 #mM
    Kb_C4etc_n = 5.4e-3 #mM
    Kc_C4etc_n = 680.0e-3 #mM
    betaC4etc_n = 0.5
    Gibbs_C4etc_n = -122.94 *1000.0 #becase R in J/(K*mol)

    # complex V: ATP-synthase: ADPm + Pim + 3*Hcyto -> ATPm + H2Om + 3*Hm
    #VmaxC5etc_a = 589.0/60.0
    #Ka_C5etc = 10.0e-3 #mM
    #Kb_C5etc = 0.5 #mM
    #Kc_C5etc = 1.0 #mM
    #betaC5etc = 0.5
    #Gibbs_C5etc = 36.03 *1000.0 #becase R in J/(K*mol) # 34-57 kJ/(K*mol) - Miller..Claycomb2013


    #  complex V: ATP-synthase: Wu2007
    #Wu2007
    VmaxC5etc_n = 0.1*5.95 # mM/s #adj for conc
    naC5etc_n = 2.5 #3.0 
    deltaG0C5etc_n = -4.51*1000.0 #J/mol

    pPiC5etc_n = 0.327 #mM/s permeability for Pi
    pAC5etc_n = 0.085 #mM/s permeability for nucleotides

    #ANT: adenine nucleotide translocase ATPm + ADPc -> ATPc + ADPm (transport ATP from mito matrix to intermembr space -> to cytosol)
    VmaxANTetc_n = 523.9/60.0 # 0.01*
    K_ADP_ANTetc_n = 10.0e-3 #mM
    K_ATP_ANTetc_n = 10.0e-3 #mM
    K_ATPm_ANTetc_n = 0.025 #mM
    betaANTetc_n = 0.6




    #########################################################################################
    ###################### GLS - glutaminase neuron
    #########################################################################################
    VmGLS_n = 38.8/3600.0
    KeqGLS_n = 1.03 # Demin2004
    KmGLNGLS_n = 12.0 
    KiGLUGLS_n = 55.0 

    #########################################################################################
    ###################### GABA neuron
    #########################################################################################





    #########################################################################################
    ############################# Fixed concentrations  #####################################
    #########################################################################################

    C_Mg_a = 0.369  #0.7 # mM  # 0.369 mM 0.4 mM -Mulquiney1999  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    C_Mg_n = 0.369  #0.7 # mM  # 0.369 mM 0.4 mM -Mulquiney1999  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    BPG23 = 0.237 #mM -Park2016 #3.1 # mM #for now fixed; check value  #C_23BPG_a = 3.1 # mM #BPG23
    BPG23_n = 0.237 #mM -Park2016 #3.1 # mM #for now fixed; check value  #C_23BPG_a = 3.1 # mM #BPG23

    GBP = 0.01 #0.6 #0.3 #0.1 #mM  #for now fixed; check value # G16bp #Quick1974: 10 μm and 600 μm #C_g16bp_a = 0.1 # mM #GBP
    GBP_n = 0.01 #0.6 #0.3 #0.1 #mM  #for now fixed; check value # G16bp #Quick1974: 10 μm and 600 μm #C_g16bp_a = 0.1 # mM #GBP

    #u[162] GLU_n = 10.0 # from MAS # 11.6 - From molecules to networks 2014 #check 
    #GSH = u[158] or u[156] for neuron and astrocyte #0.57 #1.0 #0.57 #mM #for now fixed; Koga2011: neuronal glutathione concentrations ranging from 0.2 to 1 mM, though some have suggested concentrations as high as 10 mM [7,33-37].

    C_ALA_a = 0.65 #-From Molecules to networks 2014 #1.0 #mM -Mulukutla2015  Alanine ### check more specific neuron and astroyte values
    C_ALA_n = 0.65 #-From Molecules to networks 2014 #1.0 #mM -Mulukutla2015  Alanine ### check more specific neuron and astroyte values

    C_H_cyt_a = 1000.0*10^(-7.2) #1000.0* for mM #Arce-Molina2019 biorxiv #10^(-7.3) # 10^(-7.01-7.4)
    C_H_cyt_n = 1000.0*10^(-7.2) #Arce-Molina2019 biorxiv #10^(-7.3) # 10^(-7.01-7.4)

    C_H_mito_a = 1000.0*10^(-7.8) #Arce-Molina2019 biorxiv #10^(-8) #10^(-7.8) 
    C_H_mito_n = 1000.0*10^(-7.8) #Arce-Molina2019 biorxiv #10^(-8) #10^(-7.8) 

    #Hin_n = 7.01 #7.0-7.4 Wiki # # from NEDERGAARD 1991: 7.01/7.24 # hcyt/hext  # 
    #Hmito_n = 7.8 # Mito mattrix Wiki # check it more precisely    ############################################################# check!!
    #Zhang2018 # MitoMembrPotent
    C_H_ims = 1e-4 #mM # pH=7.0 -> C = 10^(-7) M
    C_H_mitomatr = 1e-5 #mM # pH=8.0 -> C = 10^(-8) M
    C_H_ims_n = 1e-4 #mM # pH=7.0 -> C = 10^(-7) M
    C_H_mitomatr_n = 1e-5 #mM # pH=8.0 -> C = 10^(-8) M


    C_O2_mito_a = 0.01 #mM # Zhang2018 supp, consistent with whole cell O2 in Calvetti2018
    C_O2_mito_n = 0.01 #mM # Zhang2018 supp, consistent with whole cell O2 in Calvetti2018
        
    PPi_a = 0.0062 #mM pyrophosphate 
    PPi_n = 0.0062 #mM pyrophosphate 

    CO2_a = 1.2 #mM in cytosol Physiology of astroglia... book Verkhratsky #0.001 # mM from Sabate 1995 # PPP
    CO2_n = 1.2 #0.001 # mM from Sabate 1995 !!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECK IT!!!!!!!!!!!!!!!!!!!!!

    CO2_mito_a = 1.2 #21.4 #Wu2007 #or 1.2 as cytosol astrocyte? 
    CO2_mito_n = 1.2 #21.4 #Wu2007 #or 1.2 as cytosol astrocyte? 

    #from PFK Mulukutla, Mulquiney consider MgATP as main working ATP, and given their ratio [ATP]/[MgATP] = 0.105; set [ATP]_neuron = 0.105*u[28]; [ATP]_astrocyte = 0.105*u[29]



    #INa = p[1] # same with INaact !!!
    #INaact = p[1]
    
    ksi_ephys = p[2] ############# !!!!!
    ksi = p[2] #u[36] # placeholder for callback activation related to glu  #!!!!!!!!!!!!!!!!!!!!!!!!! CHECK IT !!!!!!!!!!!!!!!!!!!!!!!!!!!

    V = p[3] #now this is given from ndam #!!! was u[1]  #modified by Polina on 29 nov 2019  ############# COUPLING TO NDAM #############
    
    #Na_in = p[4] # !!!!!!!!!!!!
    
    Naout = p[5] # from STEPS #it was 140.0 - beta*(u[7] - 11.5) # mM  # u7 = Na0in  # !!!!! was 144.0 - beta*(u[7] - 11.5)  but should be 140 to be consistent with ndam 

    #IK = p[6] #gK*(n^4)*(V-VK) + gKleak*(V-VK)  # based on Calvetti2018 gKleak depends on ksi and glutamate activation
        
    K_out = p[7] #u8 before coupling with ndam
    pAKTPFK2 = p[8] # #!!!!!!!!!!!!!!!!!!!!!!!!! REGULATOR OF GLYCOLISYS FLUX !!!!!!!!!!!!!! #0.1 #0.17 # -Mulukutla2015 p.7,p.12 main text; p.11 fig5 main text: pAKT = 0.1mM;0.25mM;0.35mM;1.0mM <- high pAKT corresponds to increased glycolysis and cell growth;  #0.5 set to be consistent with Mulukutla 2014 Bistability in Glycolysis....  !!!!!!! REGULATOR OF GLYCOLISYS FLUX !!!!!!!
    
    
    
    m = u[2]
    h = u[3]
    n = u[4]

    du[36] = 0 # placeholder for callback ksi
    
    
    #VmfPFK_a  = p[9]
    #VmrPFK_a = p[10]
    
   # KATPPFK = p[11]
   # KATP_minor_PFK = p[12]
   # KADPPFK = p[13]
   # KMgPFK = p[14]
   # K23BPGPFK = p[15]
   # KLACPFK = p[16]
   # KF16BPPFK = p[17]
   # KAMPPFK = p[18]
   # KG16BPPFK = p[19]
   # KPiPFK = p[20]
   # KF26BPPFK = p[21]
   # GBP = p[22]
   # KF6PPFK = p[23]
    
    
    
    ######################################### blood Calvetti2018 #######################################################
    # blood 
    du[12]=0 # placeholder for callback (q=A(t)*Q0, blood flow)
    
    JGlc = TbGlc*( u[9]/(u[9] + KbGlc) - u[13]/(u[13] + KbGlc) )
    JLac = TbLac*( u[10]/(u[10] + KbLac) - u[14]/(u[14] + KbLac) )
    JO2 = lambda_b_O2*( 1.0/(u[11] + 4*Hct*Hb*(u[11]^2.5)/((KH^2.5)+(u[11]^2.5))  ) - u[15])^0.1    # k=0.1 eq3 Calvetti2018
    #JO2 = lambda_b_O2*(clamp( (u[11] + 4*Hct*Hb*(u[11]^2.5)/((KH^2.5)+(u[11]^2.5))  ),1e-12,(u[11] + 4*Hct*Hb*(u[11]^2.5)/((KH^2.5)+(u[11]^2.5))  ))^(-1) - u[15])^0.1 # k=0.1 eq3 Calvetti2018 # clamped to avoid /0
    
    du[9] = (1/eto_b) * ( (u[12]/Fr_blood)*( C_Glc_a -  u[9]) - JGlc ) # Glc_b
    du[10] = (1/eto_b) * ( (u[12]/Fr_blood)*( C_Lac_a -  u[10]) - JLac ) # Lac_b
    du[11] = (1/eto_b) * ( (u[12]/Fr_blood)*( C_O_a -  u[11]) - JO2 ) # O2_b
    
    ###################################################################################################################
    
    #############
    
    #jGLCtr_n = VmaxGlcTr_n*( (u[13]-u[18])/(1+ u[18]/KmGlcGlcTr_n + u[13]/KmGlcExt_GlcTr_n) )   #Berndt2015
    #jGLCtr_a = VmaxGlcTr_n*( (u[13]-u[19])/(1+ u[19]/KmGlcGlcTr_n + u[13]/KmGlcExt_GlcTr_n) )   #adapted from Berndt2015
    #jGLCtr_a = TnGlc*( u[13]/(u[13] + KnGlc) - u[19]/(u[19] + KnGlc) ) #Calvetti
    #jGLCtr_a = (VmfGlcTr*u[13]/KmGLCtr - VmrGlcTr*u[19]/KmGLCtr ) / (1+ u[13]/KmGLCtr + u[19]/KmGLCtr) #Mulukutla
    
    #jGLCtr_a = (VmfGlcTr*u[13]/KmfGLCtr - VmrGlcTr*u[19]/KmrGLCtr ) / (1+ u[13]/KmfGLCtr + u[19]/KmrGLCtr)  # adapted from Mulukutla2015
    jGLCtr_a = TaGlc*( u[13]/(u[13] + KaGlc) - u[19]/(u[19] + KaGlc) ) # Calvetti
    
    jGLCtr_n = TnGlc*( u[13]/(u[13] + KnGlc) - u[18]/(u[18] + KnGlc) ) #was called jGlc_n   #Calvetti2018
    
    
    #Calvetti2018
    jLac_a = TaLac*( u[14]/(u[14] + KaLac) - u[21]/(u[21] + KaLac) )    
    
    jLac_n = TnLac*( u[14]/(u[14] + KnLac) - u[20]/(u[20] + KnLac) )
    
    #adapted from Lambeth
    psiGPa_a = (u[65]/(u[65]+u[66]))*(Vmaxfgpa*u[60]*u[51]/(KiGLYfGPa*KPiGPa) - Vmaxrgpa*u[60]*u[59]/(KGLYbGPa*KiG1PGPa)) /  (1+ u[60]/KiGLYfGPa + u[51]/KiPiGPa + u[60]/KGLYbGPa + u[59]/KiG1PGPa + u[60]*u[51]/(KGLYfGPa*KiPiGPa) + u[60]*u[59]/(KGLYbGPa*KiG1PGPa) )
    #adapted from Lambeth
    psiGPb_a =  (u[65]/(u[65]+u[66]))*(Vmaxfgpb*u[60]*u[51]/(KiGLYfGPb*KPiGPb) - Vmaxrgpb*u[60]*u[59]/(KiGLYbGPb*KG1PGPb))* (((u[44]^AMPnHGPb)/KAMPGPb)/ (1+(u[44]^AMPnHGPb)/KAMPGPb)) /   (1+ u[60]/KiGLYfGPb + u[51]/KiPiGPb + u[60]/KiGLYbGPb + u[59]/KiG1PGPb + u[60]*u[51]/(KiGLYfGPb*KPiGPb) + u[60]*u[59]/(KiGLYbGPb*KG1PGPb) )
    #adapted from Lambeth
    psiPGLM_a =  (Vmaxfpglm*u[59]/KG1PPGLM - Vmaxrpglm*u[40]/KG6PPGLM) /(1 + u[59]/KG1PPGLM + u[40]/KG6PPGLM)
    
    # Glycogen synthase; UDPgluco + GLY ⇒ 2*GLY
    psiGS_a = kL2*u[64]*u[61] / (kmL2+u[61]) - k_L2*u[65] *u[60] / (km_L2+u[60]) # check it!!!!!!!!!!!!!!! adapted from Jay with replacement of g6p to UDP gluco for the more detailed system # 500.0 # because mM/min, see above, analogically with other params conversion #  0.5 min^-1   ## check it 
    #    psiGS_a = kL2*u[64]*u[61] / (kmL2+u[61])  

    # UDP-glucose pyrophosphorylase; G1P + UTP ⇒ udp-gluco +PPi  # rates adapted from Hester and Raushel 1987
    psiUDPGP_a =  (VmaxfUDPGP*u[62]*u[59]/(KutpUDPGP*Kg1pUDPGP) - VmaxrUDPGP*PPi_a*u[61]/(KpiUDPGP*KglucoUDPGP)) / (1 + u[59]/Kg1pUDPGP + u[62]/KutpUDPGP + (u[59]*u[62])/(Kg1pUDPGP*KutpUDPGP) + u[61]/KglucoUDPGP + PPi_a/KpiUDPGP + (PPi_a*u[61])/(KpiUDPGP*KglucoUDPGP))           

    # PP1 active conc  # Jay Newest
    psiPP1act = (-1.0/(kmaxd_PP1/(1.0+u[60]) + kmind_PP1 )*u[63]*u[65]) #+ GPb    ########################### !!!!!!!!!!!!!!!!!!!! check if GPb here !!!!!!!!!!!!!!!!!!!!!!!
    # 
    psiGSAJay = ((kg8_GSAJay*u[63]*(st_GSAJay-u[64]))/((kmg8_GSAJay/(1+s1_GSAJay*u[61]/kg2_GSAJay)) + (st_GSAJay-u[64]))) - ((kg7_GSAJay*(u[70]+u[68])*u[64]) /(kmg7_GSAJay*(1+s1_GSAJay*u[61]/kg2_GSAJay)+u[64]))      

    # Phosphorylase kinase act  0 ⇒ PHKa 
    psiPHKact = ((kg3_PHKact*u[68]*(kt_PHKact-u[70]))/(kmg3_PHKact+kt_PHKact-u[70])) - (((kg4_PHKact*u[63]*u[70]))/(kmg4_PHKact+u[70]))
    # PHK GPb->GPa #  GPb + 2*ATP ⇒ GPa + 2*ADP
    psiPHK = ((kg5_PHK*u[70]*(pt_PHK-u[65]))/(kmg5_PHK*(1+s1_PHK*u[40]/kg2_PHK) + (pt_PHK-u[65]))) - ((kg6_PHK*u[63]*u[65])/(kmg6_PHK/(1+s2_PHK*u[19]/kgi_PHK)+u[65])) - ((0.003198/(1+ u[60]) + kmind_PHK)*u[63]*u[65]) #+ GPb #### !!!!!!!!!!!!!!!!!!!!!! check if GPb here
    # Protein kinase A (=cAMP-dependent protein kinase); VPKA(), PKAb + 4*cAMP -> R2CcAMP4 + 2*PKAa by Jay, Xu-Gomez # or it was PKAb + ATP ⇒ PKAa + ADP
    psiPKA1  = kgc1_PKA12*u[69]*u[67]^2 - k_gc1_PKA12*u[71]*u[68] # check details for R2CcAMP2 and R2CcAMP2 in Jay 
    psiPKA2  = kgc2_PKA12*u[71]*u[67]^2 -  k_gc2_PKA12*u[72]*u[68] # check details for R2CcAMP2 and R2CcAMP2 in Jay 

    
    
    ##########################################
    # GLYCOLYSIS
    ##########################################
   
    # HK1
    psiHK_a =  ( VmfHK*u[29]*u[19]/(KATPHK*KGLCHK)  - VmrHK*u[31]*u[40]/(KiADPHK*KG6PHK) ) / ( 1.0 + u[29]/KiATPHK + u[40]/KiG6PHK + u[19]/KGLCHK + u[29]*u[19]/(KATPHK*KGLCHK) +  u[31]/KiADPHK  +  u[31]*u[40]/(KiADPHK*KG6PHK ) + u[19]*u[40]/(KGLCHK*KG6PHK) + u[19]*GBP/(KGLCHK*KiG16BP_HK) +   u[19]*BPG23/(KGLCHK*KiBPG23) +  u[19]*u[156]/(KGLCHK*KiGSH_HK) ) 
    
    psiHK_n =  ( VmfHK_n*u[28]*u[18]/(KATPHK_n*KGLCHK_n)  - VmrHK_n*u[30]*u[38]/(KiADPHK_n*KG6PHK_n) ) /  ( 1.0 + u[28]/KiATPHK_n + u[38]/KiG6PHK_n + u[18]/KGLCHK_n + u[28]*u[18]/(KATPHK_n*KGLCHK_n) +  u[30]/KiADPHK_n     +  u[30]*u[38]/(KiADPHK_n*KG6PHK_n ) + u[18]*u[38]/(KGLCHK_n*KG6PHK_n) + u[18]*GBP_n/(KGLCHK_n*KiG16BP_HK_n) + u[18]*BPG23_n/(KGLCHK_n*KiBPG23_n) +  u[18]*u[158]/(KGLCHK_n*KiGSH_HK_n) ) 

    # PGI
    
    psiPGI_a = (VmfPGI*u[40]/KfPGI - VmrPGI*u[41]/KrPGI )/ (1.0+u[40]/KfPGI+ u[41]/KrPGI)
    psiPGI_n = (VmfPGI_n*u[38]/KfPGI_n - VmrPGI_n*u[39]/KrPGI_n )/ (1.0+u[38]/KfPGI_n+ u[39]/KrPGI_n)
    
    
    # PFK 
    #a
    #Mulukutla2015
    N_PFK_a = 1.0 + L_pfk*( (1+0.105*u[29]/KATP_minor_PFK)*(1+ C_Mg_a/KMgPFK)*(1+BPG23/K23BPGPFK )*(1+ u[21]/KLACPFK)  / (  (1+u[41]/KF6PPFK + u[43]/KF16BPPFK)*(1+ u[44]/KAMPPFK)*(1+GBP/KG16BPPFK )*(1+u[51]/KPiPFK )*(1+ u[82]/KF26BPPFK)  )   )^4  ### note Mg dependence, check Mg conc in astrocyte        
    
    psiPFK_a =  ( (VmfPFK_a*u[29]*u[41]/(KF6PPFK*KATPPFK) ) - ( VmrPFK_a*u[31]*u[43]/(KF16BPPFK*KADPPFK) ) ) / (N_PFK_a * ( (1+ u[41]/KF6PPFK)*(1+ u[29]/KATPPFK) + (1+ u[43]/KF16BPPFK)*(1+u[31]/KADPPFK)  - 1) )                 
    
    
    psiPFK2_a =  VfPFK2*(u[29]*u[41]-u[31]*u[82]/KeqPFK2)*(0.2 + 0.8/(1+KAKTPFK2/pAKTPFK2))/ ((KiATPPFK2*KmF6PPFK2 + KmF6PPFK2*u[29] + KmATPPFK2*u[41] + KmADPPFK2*u[82]/KeqPFK2 + u[29]*u[41]+ KmF26BPPFK2*u[31]/KeqPFK2 + KmADPPFK2*u[29]*u[82]/(KeqPFK2*KiATPPFK2) + u[31]*u[82]/KeqPFK2 + KmATPPFK2*u[31]*u[41]/KiADPPFK2 + u[29]*u[41]*u[82]/KiF26BPPFK2  + u[31]*u[41]*u[82]/(KeqPFK2*KiF6PPFK2)) * (1+u[58]/KiPEPPFK2) )
    
    psiFBPFK2_a =  Vf26BPase*u[82]/ ( (1+u[41]/KiF6PF26BPase)*(KmF26BPF26BPase + u[82]) )
    
    # PFK
    #n
    #Mulukutla2015
    N_PFK_n = 1.0 + L_pfk_n*( (1.0+0.105*u[28]/KATP_minor_PFK_n)*(1.0+ C_Mg_n/KMgPFK_n)*(1.0+BPG23_n/K23BPGPFK_n )*(1+ u[20]/KLACPFK_n)  / (  (1+u[39]/KF6PPFK_n + u[42]/KF16BPPFK_n)*(1+ u[164]/KAMPPFK_n)*(1.0+GBP_n/KG16BPPFK_n )*(1.0+u[49]/KPiPFK_n )*(1.0+ u[83]/KF26BPPFK_n)  )   )^4  ### note Mg dependence, check Mg conc        
    
    psiPFK_n = ( (VmfPFK_n*u[28]*u[39]/(KF6PPFK_n*KATPPFK_n) ) - ( VmrPFK_n*u[30]*u[42]/(KF16BPPFK_n*KADPPFK_n) ) ) /  (N_PFK_n * ( (1.0+ u[39]/KF6PPFK_n)*(1.0+ u[28]/KATPPFK_n) + (1.0+ u[42]/KF16BPPFK_n)*(1.0+u[30]/KADPPFK_n)  - 1.0) )                 
    
    
    
    #ALD a
    psiALD_a = (VmfALD*u[43]/KmFBPALD - VmrALD*u[47]*u[48]/(KmGAPALD*KiDHAPALD) ) / (1.0+ BPG23/KiBPG23ALD + u[43]/KmFBPALD + (KmDHAPALD*u[47])*(1.0+ BPG23/KiBPG23ALD)/(KmGAPALD*KiDHAPALD)  + u[48]/KiDHAPALD   +  KmDHAPALD*u[43]*u[47]/(KiFBPALD*KmGAPALD*KiDHAPALD) +  u[48]*u[47]/(KmGAPALD*KiDHAPALD)   )              
    #ALD n
    psiALD_n = (VmfALD_n*u[42]/KmFBPALD_n - VmrALD_n*u[45]*u[46]/(KmGAPALD_n*KiDHAPALD_n) ) / (1.0+ BPG23_n/KiBPG23ALD_n + u[42]/KmFBPALD_n + (KmDHAPALD_n*u[45])*(1.0+ BPG23_n/KiBPG23ALD_n)/(KmGAPALD_n*KiDHAPALD_n)  + u[46]/KiDHAPALD_n   +  KmDHAPALD_n*u[42]*u[45]/(KiFBPALD_n*KmGAPALD_n*KiDHAPALD_n) +  u[46]*u[45]/(KmGAPALD_n*KiDHAPALD_n)   )              
    
    
    
    
    ###################### TPI a #####################
    #Mulquiney:
    #psiTPI = (VmfTPI*u[47]/KfTPI - VmrTPI*u[48]/KrTPI) / (1+ u[47]/KfTPI + u[48]/KrTPI)
    #Mulukutla:
    psiTPI =  (VmfTPI*u[48]/KfTPI - VmrTPI*u[47]/KrTPI) / (1+ u[48]/KfTPI + u[47]/KrTPI)
    ################################################
    
    ###################### TPI n #####################
    #Mulquiney:
    #psiTPI = (VmfTPI*u[47]/KfTPI - VmrTPI*u[48]/KrTPI) / (1+ u[47]/KfTPI + u[48]/KrTPI)
    #Mulukutla:
    psiTPI_n =  (VmfTPI_n*u[46]/KfTPI_n - VmrTPI_n*u[45]/KrTPI_n) / (1.0 + u[46]/KfTPI_n + u[45]/KrTPI_n)
    ################################################
    
    # GAPDH a
    #Mulukutla2015
    # check if it should depend on pH  pH = -log10(Ch)  Ch= 10^(-pH) pH=7.3  [H+]=(10^(-7.3))
    psiGAPDH =  (VmfGAPD*u[35]*u[51]*u[47]/(KNADgapd*KiPIgapd*KiGAPgapd) - VmrGAPD*u[52]*u[33]*(10^(-7.3))/(KiBPG13gapd*KNADHgapd) )  /(u[47]*(1+u[47]/Ki1GAPgapd)/KiGAPgapd + u[52]*(1+u[47]/Ki1GAPgapd) + KBPG13gapd*u[33]*(10^(-7.3))/(KiBPG13gapd*KNADHgapd) + KGAPgapd*u[35]*u[51]/(KNADgapd*KiPIgapd*KiGAPgapd) +u[35]*u[47]/(KiNADgapd*KiGAPgapd) + u[51]*u[47]*(1+u[47]/Ki1GAPgapd)/(KiPIgapd*KiGAPgapd) + u[35]*u[52]/(KiNADgapd*KiBPG13gapd)  + KBPG13gapd*u[51]*u[33]*(10^(-7.3))/(KiPIgapd*Ki1BPG13gapd*KNADHgapd) + u[47]*u[33]*(10^(-7.3))/(KiGAPgapd*KiNADHgapd) + u[52]*u[33]*(10^(-7.3))/(KiBPG13gapd*KNADHgapd) + u[35]*u[51]*u[47]/(KNADgapd*KiPIgapd*KiGAPgapd) + u[47]*u[35]*u[51]*u[52]/(KiGAPgapd*KNADgapd*KiPIgapd*Ki1BPG13gapd) + u[51]*u[47]*u[33]*(10^(-7.3))/(KiPIgapd*KiGAPgapd*KiNADHgapd)  + u[51]*u[52]*u[33]*(10^(-7.3))/(KiBPG13gapd*KNADHgapd*KiPIgapd*Ki1BPG13gapd )  )
    
    # GAPDH n
    #Mulukutla2015
    # check if it should depend on pH  pH = -log10(Ch)  Ch= 10^(-pH) pH=7.3  [H+]=(10^(-7.3))
    psiGAPDH_n =  (VmfGAPD_n*u[34]*u[49]*u[45]/(KNADgapd_n*KiPIgapd_n*KiGAPgapd_n) - VmrGAPD_n*u[50]*u[32]*(10^(-7.3))/(KiBPG13gapd_n*KNADHgapd_n) )  /(u[45]*(1.0+u[45]/Ki1GAPgapd_n)/KiGAPgapd_n + u[50]*(1.0+u[45]/Ki1GAPgapd_n) + KBPG13gapd_n*u[32]*(10^(-7.3))/(KiBPG13gapd_n*KNADHgapd_n) + KGAPgapd_n*u[34]*u[49]/(KNADgapd_n*KiPIgapd_n*KiGAPgapd_n) +u[34]*u[45]/(KiNADgapd_n*KiGAPgapd_n) + u[49]*u[45]*(1.0 +u[45]/Ki1GAPgapd_n)/(KiPIgapd_n*KiGAPgapd_n) + u[35]*u[52]/(KiNADgapd_n*KiBPG13gapd_n)  + KBPG13gapd_n*u[49]*u[32]*(10^(-7.3))/(KiPIgapd_n*Ki1BPG13gapd_n*KNADHgapd_n) + u[45]*u[32]*(10^(-7.3))/(KiGAPgapd_n*KiNADHgapd_n) + u[50]*u[32]*(10^(-7.3))/(KiBPG13gapd_n*KNADHgapd_n) + u[34]*u[49]*u[45]/(KNADgapd_n*KiPIgapd_n*KiGAPgapd_n) + u[45]*u[34]*u[49]*u[50]/(KiGAPgapd_n*KNADgapd_n*KiPIgapd_n*Ki1BPG13gapd_n) + u[49]*u[45]*u[32]*(10^(-7.3))/(KiPIgapd_n*KiGAPgapd_n*KiNADHgapd_n)  + u[49]*u[50]*u[32]*(10^(-7.3))/(KiBPG13gapd_n*KNADHgapd_n*KiPIgapd_n*Ki1BPG13gapd_n )  )
    
    
    #Mulukutla2015
    psiPGK  = (VmfPGK*u[52]*u[31]/(KiADPPGK*KmBPG13PGK) - VmrPGK*u[54]*u[29]/(KiATPPGK*KmPG3PGK)) /  ( 1.0 + u[52]/KiBPG13PGK + u[31]/KiADPPGK + u[52]*u[31]/(KiADPPGK*KmBPG13PGK) + u[54]/KiPG3PGK + u[29]/KiATPPGK + u[54]*u[29]/(KiATPPGK*KmPG3PGK) )
    
    #Mulukutla2015
    psiPGK_n  = (VmfPGK_n*u[50]*u[30]/(KiADPPGK_n*KmBPG13PGK_n) - VmrPGK_n*u[53]*u[28]/(KiATPPGK_n*KmPG3PGK_n)) /  ( 1.0 + u[50]/KiBPG13PGK_n + u[30]/KiADPPGK_n + u[50]*u[30]/(KiADPPGK_n*KmBPG13PGK_n) + u[53]/KiPG3PGK_n + u[28]/KiATPPGK_n + u[53]*u[28]/(KiATPPGK_n*KmPG3PGK_n) )
    
    # Mulukutla2015
    psiPGM_a =  (VmfPGM*u[54]/KmPG3pgm - VmrPGM*u[56]/KmPG2pgm) / (1.0+u[54]/KmPG3pgm + u[56]/KmPG2pgm)
    
    # Mulukutla2015
    psiPGM_n =  (VmfPGM_n*u[53]/KmPG3pgm_n - VmrPGM_n*u[55]/KmPG2pgm_n) / (1.0+u[53]/KmPG3pgm_n + u[55]/KmPG2pgm_n)
    
    # Mulukutla2015
    psiENOL_a =  (VmfENOL*u[56]*C_Mg_a/(KiMgENOL*KmPG2ENOL) - VmrENOL*u[58]*C_Mg_a/(KiMgENOL*KmPEPENOL) ) /   ( 1.0+ u[56]/KiPG2ENOL + 2*C_Mg_a/KiMgENOL + u[56]*C_Mg_a/(KiMgENOL*KmPG2ENOL) + u[58]/KiPEPENOL +  u[58]*C_Mg_a/(KiMgENOL*KmPEPENOL ) )  
    
    # Mulukutla2015
    psiENOL_n =  (VmfENOL_n*u[55]*C_Mg_n/(KiMgENOL_n*KmPG2ENOL_n) - VmrENOL_n*u[57]*C_Mg_n/(KiMgENOL_n*KmPEPENOL_n) ) /  ( 1.0+ u[55]/KiPG2ENOL_n + 2*C_Mg_n/KiMgENOL_n + u[55]*C_Mg_n/(KiMgENOL_n*KmPG2ENOL_n) + u[57]/KiPEPENOL_n +  u[57]*C_Mg_n/(KiMgENOL_n*KmPEPENOL_n ) )  
    
    #Mulukutla2015 
    psiPK_a = (VmfPK*u[58]*u[31]/(KpepPK*KadpPK) - VmrPK*u[23]*u[29]/(KpyrPK*KMgatpPK) ) / (  ((1.0 +u[58]/KpepPK)*(1+u[31]/KadpPK) + (1+ u[23]/KpyrPK)*(1+u[29]/KMgatpPK) -1) * (1 + L_pk * ( (1+ 0.105*u[29]/KatpPK) * (1+ C_ALA_a/KalaPK)/( (1+ u[58]/KpepPK + u[23]/KpyrPK)*(1+u[43]/KfbpPK + GBP/KgbpPK ) ) )^4  )   )
    
    #Mulukutla2015 
    psiPK_n = (VmfPK_n*u[57]*u[30]/(KpepPK_n*KadpPK_n) - VmrPK_n*u[22]*u[28]/(KpyrPK_n*KMgatpPK_n) ) / (((1.0 +u[57]/KpepPK_n)*(1.0+u[30]/KadpPK_n) + (1+ u[22]/KpyrPK_n)*(1+u[28]/KMgatpPK_n) -1.0) * (1.0 + L_pk_n * ( (1.0 + 0.105*u[28]/KatpPK_n) * (1.0 + C_ALA_n/KalaPK_n)/( (1.0 + u[57]/KpepPK_n + u[22]/KpyrPK_n)*(1.0+u[42]/KfbpPK_n + GBP_n/KgbpPK_n ) ) )^4  )   )
    
    
    
    ########### PCr-Cr - Calvetti
    psiCr_n = V_Cr_n * ( (u[28]/u[30])/ ( mu_Cr_n + (u[28]/u[30]) )) * (u[26] / (u[26] + K_Cr_n))
    psiCr_a = V_Cr_a * ((u[29]/u[31]) / ( mu_Cr_a + u[29]/u[31] ))* (u[27] / (u[27] + K_Cr_a))   
    psiPCr_n = V_PCr_n * ((1/(u[28]/u[30])) / ( mu_PCr_n + (1/(u[28]/u[30])) ))  * (u[24] / (u[24] + K_PCr_n))
    psiPCr_a = V_PCr_a * ((1/(u[29]/u[31])) / ( mu_PCr_a + (1/(u[29]/u[31])) ))  * (u[25] / (u[25] + K_PCr_a))
    
    
    
    
    #Mulukutla2015
    psiLDH_a =  (VmfLDH*u[33]*u[23]/(KiNADHldh*KmPYRldh) -  VmrLDH*u[35]*u[21]/(KiNADldh*KmLACldh) )  /          ( (1.0 + KmNADHldh*u[23]/(KiNADHldh*KmPYRldh) + KmNADldh*u[21]/(KiNADldh*KmLACldh))*(1+ u[23]/KiPYRPRIMEldh) +     u[33]/KiNADHldh + u[35]/KiNADldh +  u[33]*u[23]/(KiNADHldh*KmPYRldh) +        KmNADldh*u[33]*u[21]/(KiNADldh*KiNADHldh*KmLACldh) +     KmNADHldh*u[35]*u[23]/(KiNADldh*KiNADHldh*KmPYRldh) +     u[35]*u[21]/(KiNADldh*KmLACldh) +    u[33]*u[23]*u[21]/(KiNADHldh*KmPYRldh*KiLACldh)+    u[35]*u[23]*u[21]/(KiNADldh*KiPYRldh*KmLACldh)  )    
    #Calvetti2018
    
    
    #Mulukutla2015
    psiLDH_n =  (VmfLDH_n*u[32]*u[22]/(KiNADHldh_n*KmPYRldh_n) -  VmrLDH_n*u[34]*u[20]/(KiNADldh_n*KmLACldh_n) )  /  ( (1.0 + KmNADHldh_n*u[22]/(KiNADHldh_n*KmPYRldh_n) + KmNADldh_n*u[20]/(KiNADldh_n*KmLACldh_n))*(1.0 + u[22]/KiPYRPRIMEldh_n) +     u[32]/KiNADHldh_n + u[34]/KiNADldh_n +  u[32]*u[22]/(KiNADHldh_n*KmPYRldh_n) + KmNADldh_n*u[32]*u[20]/(KiNADldh_n*KiNADHldh_n*KmLACldh_n) +  KmNADHldh_n*u[34]*u[22]/(KiNADldh_n*KiNADHldh_n*KmPYRldh_n) +   u[34]*u[20]/(KiNADldh_n*KmLACldh_n) +    u[34]*u[22]*u[20]/(KiNADHldh_n*KmPYRldh_n*KiLACldh_n)+    u[34]*u[22]*u[20]/(KiNADldh_n*KiPYRldh_n*KmLACldh_n)  )    
    
    
    # ATPase
    # for coupling see above
#    H2_metabo_atpase= 0.833*H1_metabo_atpase
    ### check if need to use u[..] instead of ATP ADP
    p_n_ratio = clamp(u[28],1e-12,u[28])/clamp(u[30],1e-12,u[30]) #u28 = ATP_n # u30 = ADP_n
    p_a_ratio = clamp(u[29],1e-12,u[29])/clamp(u[31],1e-12,u[31])
    r_n_ratio = clamp(u[32],1e-12,u[32])/clamp(u[34],1e-12,u[34]) # u32 = NADH_n # u34 = NAD_n
    r_a_ratio = clamp(u[33],1e-12,u[33])/clamp(u[35],1e-12,u[35])
     
#    JpumpNa = (p_n_ratio/(mu_pump_ephys + p_n_ratio)) * (rho/(1+exp((25.0 -u[7])/3))) * (1/(1+exp(5.5 - K_out)))  #K_out was u[8]
#    JgliaK = (p_a_ratio/(mu_glia_ephys + p_a_ratio)) * (glia/(1+exp((18.0 - K_out)/2.5))) # K_out was u[8]
#    JdiffK = epsilon*(K_out - kbath) #was with K_out = u[8] before ndam 
    
    #see above Naout = p[5] # from STEPS #it was 140.0 - beta*(u[7] - 11.5) # mM  # u7 = Na0in  # !!!!! was 144.0 - beta*(u[7] - 11.5)  but should be 140 to be consistent with ndam 
    #Naout = 144.0 - beta*(u[7] - 11.5) # mM  # u7 = Na0in
#    VNa = 26.64*log(clamp(Naout,1e-12,Naout)/clamp(u[7],1e-12,u[7])) # u7 = Na0in
#    gNaleak = (1+ksi)*gNa0leak
    
#    INaleak = gNaleak*(u[1]-VNa)
#    INaleak0 = gNa0leak*(u[1]-VNa) # check if this is correct, not given explicitly in Calvetti2018
#    INaact = INaleak - INaleak0
    
    
### coupling with NDAM and STEPS:    
    Kin = 140.0 + (11.5 - p[4])  #140.0 + (11.5 - u[7]) # u7 = Na0in # p[4] = Na_in from ndam !!! modified by Polina on 29 nov 2019
    VK = 26.64*log(clamp(K_out,1e-12,K_out)/clamp(Kin,1e-12,Kin)) # clamp(x,1e-12,x) # u8 = K0out
    gKleak = (1+ksi_ephys)*gK0leak
    #without ndam it was IK = gK*(n^4)*(V-VK) + gKleak*(V-VK)  # based on Calvetti2018 gKleak depends on ksi and glutamate activation
    #IK = p[6] #gK*(n^4)*(V-VK) + gKleak*(V-VK)  # based on Calvetti2018 gKleak depends on ksi and glutamate activation
        

    #JpumpNa = (p_n_ratio/(mu_pump_ephys + p_n_ratio)) * (rho/(1+exp((25.0 -p[4])/3))) * (1/(1+exp(5.5 - K_out))) # p[4] = Na_in from ndam !!! modified by Polina on 29 nov 2019 # it was#(p_n_ratio/(mu_pump_ephys + p_n_ratio)) * (rho/(1+exp((25.0 -u[7])/3))) * (1/(1+exp(5.5 - K_out)))
    #JgliaK = (p_a_ratio/(mu_glia_ephys + p_a_ratio)) * (glia/(1+exp((18.0 - K_out)/2.5)))
    #JdiffK = epsilon*(K_out - kbath) # #was with K_out = u[8] before ndam 

    
    # ATPase
   # H2_metabo_atpase= 0.833*H1_metabo_atpase
#K_out
#    JpumpNa = (p_n_ratio/(mu_pump_ephys + p_n_ratio)) * (rho/(1+exp((25.0 -p[4])/3))) * (1/(1+exp(5.5 - K_out))) #(p_n_ratio/(mu_pump_ephys + p_n_ratio)) * (rho/(1+exp((25.0 -u[7])/3))) * (1/(1+exp(5.5 - K_out))) # p[4] = Na_in from ndam !!! modified by Polina on 29 nov 2019
#    JgliaK = (p_a_ratio/(mu_glia_ephys + p_a_ratio)) * (glia/(1+exp((18.0 - K_out)/2.5)))
#    JdiffK = epsilon*(K_out - kbath) #K_out


    #ndam !!!!!!!!!!!! VNa = 26.64*log(clamp(Naout,1e-12,Naout)/clamp(u[7],1e-12,u[7])) # u7 = Na0in
 #   gNaleak = (1+ksi)*gNa0leak

    #ndam !!!!!!!!!!!! INaleak = gNaleak*(u[1]-VNa)
    #ndam !!!!!!!!!!!! INaleak0 = gNa0leak*(u[1]-VNa) # check if this is correct, not given explicitly in Calvetti2018
    #ndam !!!!!!!!!!!! INaact = INaleak - INaleak0

    psiNKA_n = 0.5 #mM/s - both from our 4 cells and SciRep2019Yi,Grill paper #H1_metabo_atpase + s_metabo_atpase*(eto_n*JpumpNa + 0.33*(gamma/sigma_atpase)* INaact ) # INaact because ksi>0 in a model 
    psiNKA_a = 0.025 #based on ini rate estimate based on Calvetti #H2_metabo_atpase + s_metabo_atpase*(eto_ecs*JgliaK/2.0 + 2.33*(gamma/sigma_atpase)* INaact )



    ###############################################################################################################################################################    
    ############################## GLU-GLN astrocyte   
    
    # EAAT1  / GLT-1
    
    #VrevEAAT = (R*T/(2*F))* log( ((Na_syn_EAAT/u[74])^3) *  (H_syn_EAAT/H_ast_EAAT)  *  (u[81]/u[78])   *   (u[73]/K_out )  )
  #  VrevEAAT = (R*T/(2*F))* log( clamp(((Na_syn_EAAT/ clamp(u[74],1e-12,u[74])  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (u[81]/  clamp(u[78],1e-12,u[78])   ) * (K_ast_EAAT/  clamp(K_out,1e-12,K_out))  ,1e-12,  ((Na_syn_EAAT/ clamp(u[74],1e-12,u[74])  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (u[81]/  clamp(u[78],1e-12,u[78])   ) * (K_ast_EAAT/  clamp(K_out,1e-12,K_out))   ) ) # simplified K
    VrevEAAT = (R*T/(2*F))* log( clamp(((Na_syn_EAAT/ clamp(u[74],1e-12,u[74])  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (u[81]/  clamp(u[78],1e-12,u[78])   ) * (u[73]/  clamp(K_out,1e-12,K_out))  ,1e-12,  ((Na_syn_EAAT/ clamp(u[74],1e-12,u[74])  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (u[81]/  clamp(u[78],1e-12,u[78])   ) * (u[73]/  clamp(K_out,1e-12,K_out))   ) ) # simplified K
    
    VEAAT =  (1/(2*F))* SA_ast_EAAT * (  -alpha_EAAT*exp(-beta_EAAT*(u[75] - VrevEAAT)) )  # # u[75] = Va0
    psiEAAT12 = - 0.1*VEAAT / Vol_syn  
    # psiEAAT12 = - ((1/(2*F))* SA_ast_EAAT * (  -alpha_EAAT*exp(-beta_EAAT*(u[75] -           (R*T/(2*F))* log( clamp(((Na_syn_EAAT/ clamp(u[74],1e-12,u[74])  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (u[81]/  clamp(u[78],1e-12,u[78])   ) * (K_ast_EAAT/  clamp(K_out,1e-12,K_out))  ,1e-12,  ((Na_syn_EAAT/ clamp(u[74],1e-12,u[74])  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (u[81]/  clamp(u[78],1e-12,u[78])   ) * (K_ast_EAAT/  clamp(K_out,1e-12,K_out))   ) )          )) ) ) / Vol_syn 
     
    # GDH
    #psiGDH_simplif_a: Mulukutla2015 + Botman2014
    psiGDH_simplif_a = VmGDH_a*(u[131]*u[78] - u[132]*u[123]/KeqGDH_a)/ (KiNAD_GDH_a*KmGLU_GDH_a + KmGLU_GDH_a*u[131] + KiNAD_GDH_a*u[78] + u[78]*u[131] + u[78]*u[131]/KiAKG_GDH_a + KiNAD_GDH_a*KmGLU_GDH_a*u[132]/KiNADH_GDH_a +     KiNAD_GDH_a*KmGLU_GDH_a*KmNADH_GDH_a*u[123]/(KiAKG_GDH_a*KiNADH_GDH_a) + KmNADH_GDH_a*u[78]*u[132]/KiNADH_GDH_a +    KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a*u[132]/(KiAKG_GDH_a*KiNADH_GDH_a) +  KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a*u[123]*u[132]/(KiAKG_GDH_a*KiNADH_GDH_a)   +   u[78]*u[131]*u[123]/KiAKG_GDH_a + KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a/KiAKG_GDH_a +  KiNAD_GDH_a*KmGLU_GDH_a*u[78]*u[132]*u[123]/(KiGLU_GDH_a*KiAKG_GDH_a*KiNADH_GDH_a) +   KiNAD_GDH_a*KmGLU_GDH_a*u[123]*u[132]/(KiAKG_GDH_a*KiNADH_GDH_a)  +  KmNADH_GDH_a*KmGLU_GDH_a*u[123]*u[131]/(KiAKG_GDH_a*KiNADH_GDH_a) )
    
    #GLN synthase 
    #psiGLNsynth_a = VmaxGLNsynth_a*(u[78]/( KmGLNsynth_a  +  u[78]))*( (u[29]/u[31])/(muGLNsynth_a + u[29]/u[31] ) )   #Calvetti2011, Pamiljans1961
    psiGLNsynth_a = VmaxGLNsynth_a*(u[78]/( KmGLNsynth_a  +  u[78]))*( (u[29]/u[31])/(muGLNsynth_a + u[29]/u[31] ) )   #Calvetti2011, Pamiljans1961
    
    #SNAT GLN transporter
    psiSNAT_GLN_a = TmaxSNAT_GLN_a*u[80]/(KmSNAT_GLN_a+u[80])
    
    
    #PC=PYRCARB=pyruvate carboxylase
    psiPYRCARB_a = ( (u[135]/u[134])/(muPYRCARB_a +  (u[135]/u[134]))   )*VmPYRCARB_a*(u[120]*CO2_mito_a - u[128]/KeqPYRCARB_a)/(  KmPYR_PYRCARB_a*KmCO2_PYRCARB_a +  KmPYR_PYRCARB_a*CO2_mito_a + KmCO2_PYRCARB_a*u[120] + CO2_mito_a*u[120])
    
    
    # Adenylate cyclase; ATP ⇒ cAMP + Pi #Ppi
   psiAC = (VmaxfAC*u[29]/KACATP - VmaxrAC*u[67]*u[51]/(KpiAC*KcAMPAC))/(1 + u[29]/KACATP + u[67]/KcAMPAC + (u[67]*u[51])/(KcAMPAC*KpiAC) + u[51]/KpiAC)


    # Phosphodiesterase; cAMP ⇒ AMP  # Enzyme assays for cGMP hydrolysing Phosphodiesterases S.D. Rybalkin, T.R. Hinds, and J.A. Beavo https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4091765/
   psiPDE = VmaxPDE*u[67]/(Kmcamppde + u[67])  

    
    # 15. Adenylate Kinase ; SigmaATP + SigmaAMP <--> 2*SigmaADP #  Lambeth 2002 JayGliaExpand
    psiADK = ((Vmaxfadk*(u[29]*u[44])/(KATPADK*KAMPADK) - (Vmaxradk*(((u[31])^2)/(KADPADK^2)))) / (1 + u[29]/KATPADK + u[44]/KAMPADK + (u[29]*u[44])/(KATPADK*KAMPADK) + (2*u[31])/KADPADK + ((u[31])^2)/(KADPADK^2)))
   # psiADK = ((Vmaxfadk*(u[29]*u[44])/(KATPADK*KAMPADK) - (Vmaxradk*(((u[31])^2)/(KADPADK^2)))) / (1 + u[29]/KATPADK + u[44]/KAMPADK + (u[29]*u[44])/(KATPADK*KAMPADK) + (2*u[31])/KADPADK + ((u[31])^2)/(KADPADK^2)))/60000.0 

    
    #ninf(ccai) = ((75*1e-6 + JNCX)^2 ) /((75*1e-6 + JNCX)^2 + 0.1^2)
    #JNCX = 0.5*u[3]*  ((u[76])^2 ) /((u[76])^2 + 0.1^2)  *u[75]  # JmaxNCX*h*ninf(ccai)*u[75]  # const JmaxNCX = 0.5
    
    JNCX = JmaxNCX * u[74] *  ((u[76])^2 ) /((u[76])^2 + 0.1^2)  *u[75]  # JmaxNCX*h*ninf(ccai)*u[75]  # const JmaxNCX = 0.5
    # 0.5 * 15 *  ( ((75e-3)^2) /(  (75e-3)^2  + 0.1^2) )  * -90
    
    JNKCC = 0# eto_a*(JNKCCmax * log(  (clamp(K_out,1e-12,K_out)/clamp(u[73],1e-12,u[73])  ) * (Naout/clamp(u[74],1e-12,u[74] ) ) *( (u[5]/u[6] )^2  ) )) #  0.5* ... JNKCCmax in Witthoft is estimation because # Cl_in ~= Na_in in Astro by Withoft2013# check which log!!!!!
    
     
    #JKirAS = gKirS*sqrt(clamp(K_out,1e-12,K_out)   ) * (u[75] - EKirProc*log(clamp(K_out,1e-12,K_out)/clamp(u[73],1e-12,u[73]) ))  # check which log!!!!!
    #JKirAS = gKirS*sqrt(clamp(K_out,1e-12,K_out)   ) * (Va0 - EKirProc*log(clamp(K_out,1e-12,K_out)/clamp(u[73],1e-12,u[73]) ))  # check which log!!!!!
    
    #JBK = 
    JKirAV = gKirV*sqrt(clamp(K_out,1e-12,K_out))*(u[75] - EKirEndfoot*log(clamp(K_out,1e-12,K_out)/clamp(u[73],1e-12,u[73]) ) ) # check which log!!!!!  ####################### change with coupling p[]
    IleakAst = gLeakAst*(u[75] - VleakAst)

    
#latest    INKAastNa = (1/eto_a) * psiNKA_a #(p_a_ratio/(mu_glia_ephys + p_a_ratio)) * (rho/(1+exp((31.0 -u[74])/3))) * (1/(1+exp(5.5 - K_out))) # analogy to Calvetti
    INKAastNa = 0# psiNKA_a #(p_a_ratio/(mu_glia_ephys + p_a_ratio)) * (rho/(1+exp((31.0 -u[74])/3))) * (1/(1+exp(5.5 - K_out))) # analogy to Calvetti
    
    INKAastK = 0# JgliaK #/3.0 # -(2/3)*INKAastNa
    #INKAastK = JNKAmax*(  (K_out/(K_out + KK0a) ) *( (u[74]^1.5)/((u[74]^1.5) + KNaiAst^1.5 )   ) )
    #INKAastNa = -(3/2)*INKAastK
    
    #du[73] = s_metabo_atpase*eto_ecs*JgliaK  + JNKCC + JKirAS  - JBK - JKirAV - RdcKA* (u[73]- K_a0) # Witthoft2013   #(1/eto_a)*(JgliaK )   # check if IK accessible for it; Calvetti2018 # K_a
    #du[73] = (1/tau) *( INKAastK  + JNKCC + JKirAS  - JKirAV - RdcKA* (u[73]- K_a0) ) # no JBK    # Witthoft2013   #(1/eto_a)*(JgliaK )   # check if IK accessible for it; Calvetti2018 # K_a
    #du[73] = 0# (1/tau) *( INKAastK   + JKirAS  + JKirAV - RdcKA* (u[73]- K_a0) ) # no JBK    # Witthoft2013   #(1/eto_a)*(JgliaK )   # check if IK accessible for it; Calvetti2018 # K_a
    
    # ((1/p_a_ratio) / ( mu_Glc_a + (1/p_a_ratio) ))  * 
    
    
    jO2_n = lambda_n_O2*(u[15]- u[16])
    jO2_a = lambda_a_O2*(u[15]- u[17])

    # metabolism
    #psiOxphos_n = V_oxphos_n * ((1/p_n_ratio) / ( mu_oxphos_n + (1/p_n_ratio) )) * (r_n_ratio / (nu_oxphos_n + r_n_ratio) ) * (u[16] / (u[16] + K_oxphos_n))
    #psiOxphos_a = V_oxphos_a * ((1/p_a_ratio) / ( mu_oxphos_a + (1/p_a_ratio) )) * (r_a_ratio / (nu_oxphos_a + r_a_ratio) ) * (u[17] / (u[17] + K_oxphos_a))

    
    
    #Glutathione Mulukutla2015
    psiGPX_a = V_GPX_a * u[156]
    psiGSSGR_a = (Vmf_GSSGR_a*u[150]*u[157] - Vmr_GSSGR_a*u[156]*u[149]) / (  1.73 + 288.0*u[150] + 34.3*u[157] + 7.77*1e-5*u[156] + 24.7*u[149] + 4020.0*u[150]*u[157] + 0.013*u[150]*u[156] + 490.0*u[157]*u[149] + 5.55*1e-4*u[156]^2 +  0.0011*u[156]*u[149] + 1.24*u[156]*u[149] + 0.326*u[150]*u[156]*u[157] + 20.8*u[150]*u[156]*u[157] + 0.0925*u[150]*u[156]^2 + 24.5*u[157]*u[156]*u[149] +  0.178*u[149]*u[156]^2 + 2.32*u[150]*u[157]*u[156]^2 + 27.4*u[157]*u[149]*u[156]^2  )
    
    #Glutathione neuron Mulukutla2015
    psiGPX_n = V_GPX_n * u[158]
    psiGSSGR_n = (Vmf_GSSGR_n*u[114]*u[159] - Vmr_GSSGR_n*u[158]*u[113]) / ( 1.73 + 288.0*u[114] + 34.3*u[159] + 7.77*1e-5*u[158] + 24.7*u[113] + 4020.0*u[114]*u[159] + 0.013*u[114]*u[158] + 490.0*u[159]*u[113] + 5.55*1e-4*u[158]^2 +  0.0011*u[158]*u[113] + 1.24*u[158]*u[113] + 0.326*u[114]*u[158]*u[159] + 20.8*u[114]*u[158]*u[159] + 0.0925*u[114]*u[158]^2 + 24.5*u[159]*u[158]*u[113] +  0.178*u[113]*u[158]^2 + 2.32*u[114]*u[159]*u[158]^2 + 27.4*u[159]*u[113]*u[158]^2  )   
   
    
    
                          ################################################################################################################################
    
##################### MITO ASTROCYTE #####################
    
    #Mulukutla2015
    #psiPYRtrcyt2mito_a = VmPYRtrcyt2mito_a*(u[23]*C_H_cyt_a - u[120]*C_H_mito_a)
    #adapted from Berndt2015
    #psiPYRtrcyt2mito_a = VmPYRtrcyt2mito_a*(u[23]*C_H_cyt_a/KmPyrCytTr - u[120]*C_H_mito_a/KmPyrMitoTr)/( (1+u[23]/KmPyrCytTr)*(1+u[120]/KmPyrMitoTr) )
    psiPYRtrcyt2mito_a = VmPYRtrcyt2mito_a*(u[23]*C_H_cyt_a  - u[120]*C_H_mito_a )/( (1+u[23]/KmPyrCytTr)*(1+u[120]/KmPyrMitoTr) )
    #or:
    # cytosol - mitochondria metabolic communication
    # Pyruvate cyto-mito exchange, PYR ⇒ PYRmito, based on Berndt 2015
    #VmaxPyrEx_n = 128.0 #### CHECK IT!!!! UNITS!!!! 
    #KmPyrIn_n = 0.15
    #KmPyrMito_n = 0.15
    #Hin_n = 7.01 #7.0-7.4 Wiki # # from NEDERGAARD 1991: 7.01/7.24 # hcyt/hext  # 
    #Hmito_n = 7.8 # Mito mattrix Wiki # check it more precisely    ############################################################# check!!
    #@reaction_func VPYRex(PYR,Hin,PYRmito,Hmito) = VmaxPyrEx*(PYR*Hin-PYRmito*Hmito) / ((1+PYR/KmPyrIn)*(1+PYRmito/KmPyrMito))
    #psiPYRex_n  = VmaxPyrEx_n*(u[22]*10^(-Hin_n)-u[84]*10^(-Hmito_n)) / ((1+u[22]/KmPyrIn_n)*(1+u[84]/KmPyrMito_n))

    
    #TCA a
    
    psiPDH_a = VmaxPDHCmito_a*(1.0+AmaxCaMitoPDH_a*u[133]/(u[133] + KaCaMitoPDH_a)) * (u[120]/(u[120]+KmPyrMitoPDH_a)) * (u[131]/(u[131] + KmNADmitoPDH_a)) * (u[130]/(u[130] + KmCoAmitoPDH_a))

    psiCS_a = VmaxCSmito_a*(u[128]/(u[128] + KmOxaMito_a*(1.0 + u[121]/KiCitMito_a))) * (u[129]/(u[129] + KmAcCoAmito_a*(1.0+u[130]/KiCoA_a)))
 
    psiACO_a = VmaxAco_a*(u[121]-u[122]/KeqAco_a) / (1.0+u[121]/KmCit_a + u[122]/KmIsoCit_a)
    
        
    #Berndt2012
    #psiIDH_a = 0.001*VmaxIDH_a*((u[122]^nIsoCitmitoIDH_a) / (u[122]^nIsoCitmitoIDH_a + (KmIsoCitmito1IDH_a/(1+(u[133]/KaCaidhIDH_a)^nCaIdhIDH_a) + KmIsoCitmito2IDH_a)^nIsoCitmitoIDH_a) )    * (u[131]/(u[131] + KmNADmitoIDH_a*(1+u[132]/KiNADHmitoIDH_a)))   
    
    alpha_IDH_a = 1.0 + Ka_adp_IDH_a*(1.0 +(0.1*u[135])/Ki_atp_IDH_a)/(0.1*u[135]) # 90% ATP in mito are MgATP
    psiIDH_a = VmaxfIDH_a*( u[131]*u[122]^nH_IDH_a - (u[122]^(nH_IDH_a-1))*u[123]*u[132]*CO2_mito_a /Keq_IDH_a  ) /    (   u[131]*u[122]^nH_IDH_a + (KmbIDH_a^nH_IDH_a)*alpha_IDH_a*u[131] +  KmaIDH_a*(u[122]^nH_IDH_a + (KibIDH_a^nH_IDH_a)*alpha_IDH_a + u[132]*(KibIDH_a^nH_IDH_a)*alpha_IDH_a/KiqIDH_a  ) )
    
    psiKGDH_a = VmaxKGDH_a*(1-u[133]/(u[133]+KiCaKGDH_a))*(u[123]/(u[123]+(Km1KGDHKGDH_a/(1+u[133]/KiAKGCaKGDH_a)+Km2KGDHKGDH_a)*(1+u[132]/KiNADHKGDHKGDH_a))) *   (u[131]/(u[131]+KmNADkgdhKGDH_a*(1+u[132]/KiNADHKGDHKGDH_a))) * (u[130]/(u[130] + KmCoAkgdhKGDH_a*(1+u[124]/KiSucCoAkgdhKGDH_a)))

    psiSCS_a  = VmaxSuccoaATP_a*(1+AmaxPscs_a*((u[136]^npscs_a)/((u[136]^npscs_a)+(Kmpscs_a^npscs_a)))) * (u[124]*u[134]*u[136] -  u[125]*u[130]*u[135]/Keqsuccoascs_a)/((1+u[124]/Kmsuccoascs_a)*(1+u[134]/KmADPscs_a)*(1+u[136]/KmPimitoscs_a)+(1+u[125]/Kmsuccscs_a)*(1+u[130]/Kmcoascs_a)*(1+u[135]/Kmatpmitoscs_a))

    #psiSDH_a = VmaxDHchang_a*((u[131]/u[132])/(u[131]/u[132]+KrDHchang_a))

    #Mulukutla2015 
    alpha_SDH_a = (1.0 + u[128]/KiOXA_SDH_a + u[125]/KaSUC_SDH_a + u[126]/KaFUM_SDH_a ) / (1.0 + u[125]/KaSUC_SDH_a + u[126]/KaFUM_SDH_a  )
    
    psiSDH_a = Vf_SDH_a*( u[125]*u[137] - u[138]*u[126]/Keq_SDH_a  ) /     ( KiSUC_SDH_a*KmQ_SDH_a*alpha_SDH_a  + KmQ_SDH_a*u[125] + KmSuc_SDH_a*alpha_SDH_a*u[137] + u[125]*u[137] + KmSuc_SDH_a*u[137]*u[126]/KiFUM_SDH_a  +    (KiSUC_SDH_a*KmQ_SDH_a/(KiFUM_SDH_a*KmQH2_SDH_a) )*( KmFUM_SDH_a*alpha_SDH_a*u[138] + KmQH2_SDH_a*u[126] + KmFUM_SDH_a*u[125]*u[138]/KiSUC_SDH_a + u[138]*u[126]   )   )
    
    psiFUM_a = Vmaxfum_a*(u[126] - u[127]/Keqfummito_a)/(1+u[126]/Kmfummito_a+u[127]/Kmmalmito_a)

    psiMDH_a = VmaxMDHmito_a*(u[127]*u[131]-u[128]*u[132]/Keqmdhmito_a) / ((1+u[127]/Kmmalmdh_a)*(1+u[131]/Kmnadmdh_a)+(1+u[128]/Kmoxamdh_a)*(1+u[132]/Kmnadhmdh_a))


    #AAT/GOT astrocyte mito
    alpha_AAT_a = 1.0 + u[123]/KiAKG_AAT_a
    psiAAT_a = VfAAT_a*(u[141]*u[123] - u[128]*u[142]/KeqAAT_a) /     ( KmAKG_AAT_a*u[141] +  KmASP_AAT_a*alpha_AAT_a*u[123] + u[141]*u[123] + KmASP_AAT_a*u[123]*u[142]/KiGLU_AAT_a + (  KiASP_AAT_a*KmAKG_AAT_a/(KmOXA_AAT_a*KiGLU_AAT_a)  )*     ( KmGLU_AAT_a*u[141]*u[128]/KiASP_AAT_a + u[128]*u[142] +  KmGLU_AAT_a*alpha_AAT_a*u[128] + KmOXA_AAT_a*u[142] )  )
   

    
    ################################################################################################################################
    ## OXPHOS detailed
    # from Zhang2018 + Chang2011 + Heiske2017 + Wu2007...
    
    deltaGh = F*u[160] + R*T*log10(C_H_ims/C_H_mitomatr) #also try and compare deltaGh based on Heiske2017
    
    psiC1etc_a = VmaxC1etc_a*( u[132]*u[137]*exp(-betaC1etc*(4*deltaGh + Gibbs_C1etc)/(R*T)) - u[131]*u[138]*exp(-(betaC1etc-1.0)*(4*deltaGh + Gibbs_C1etc)/(R*T))  ) /  (Ka_C1etc*Kb_C1etc*(1.0 + u[132]/Ka_C1etc + u[131]/Kc_C1etc)*(1.0 + u[137]/Kb_C1etc + u[138]/Kd_C1etc)   )
    
    psiC3etc_a = VmaxC3etc_a*( u[138]*(u[140]^2)*exp(betaC3etc*(-4*deltaGh +2*F*u[160] - Gibbs_C3etc )/(R*T) )   -   u[137]*(u[139]^2)*exp((betaC3etc-1.0)*(-4*deltaGh +2*F*u[160] - Gibbs_C3etc )/(R*T) )   )/   ( Ka_C3etc*(Kb_C3etc^2)*(1+u[138]/Ka_C3etc +u[137]/Kc_C3etc )*(1 + (u[140]^2)/(Kb_C3etc^2) + (u[139]^2)/(Kd_C3etc^2) ) )
    
    psiC4etc_a = VmaxC4etc_a*( (u[139]^2)*(C_O2_mito_a^0.5)*exp(betaC4etc*( -2*deltaGh -2*F*u[160] - Gibbs_C4etc  )/(R*T)) - (u[140]^2)*exp((betaC4etc-1.0)*(-2*deltaGh -2*F*u[160] - Gibbs_C4etc)/(R*T))   ) / ( (Ka_C4etc^2)*(Kb_C4etc^0.5)*(1.0 + (u[139]^2)/(Ka_C4etc^2) + (u[140]^2)/(Kc_C4etc^2) ) * (1.0 + (C_O2_mito_a^0.5)/(Kb_C4etc^0.5)) )
  
#    psiC5etc_a = VmaxC5etc_a*( u[134]*u[136]*exp( 3*betaC5etc*( deltaGh - Gibbs_C5etc )/(R*T) ) - u[135]*exp(   3*(betaC5etc-1.0)*( deltaGh - Gibbs_C5etc )/(R*T) )  ) /    ( Ka_C5etc*Kb_C5etc*(1.0 + u[134]/Ka_C5etc + u[135]/Kc_C5etc )*(1.0 + u[136]/Kb_C5etc) )
    
    #Heiske2017 ATPsynthase
#    Kmm0ADP_C5etc = 0.2 # mM
#    gammaKa_C5etc = 0.419
#    ksiKa_C5etc = 0.000 #0-0.028
#    ksiKp_C5etc = 0.14 # 0.028-0.201
#    Kmm0ATP_C5etc = 0.1 # mM
#    Km0Pi_C5etc = 4.0 # mM
#    kf0_C5etc = 3.008e-12 #mM/s /Wmito
#    kb0_C5etc = 0.0307e-3 #mM/s /Wmito
#    gammaK_C5etc = 1.0 #0.973-1.0
#    ksiK_C5etc = 0.86 #0.78-0.97
    
#    na_C5etc = 2.67
#    deltaMh_C5etc = -2.3*R*T*log10(C_H_ims/C_H_mitomatr)#29.15*1000.0
    
    #check if should be -deltaGh here instead of deltaGh 
#    deltaGhC5 = -(R*T*log10(C_H_ims/C_H_mitomatr) + na_C5etc*deltaMh_C5etc  ) #(F*u[160] + R*T*log10(C_H_ims/C_H_mitomatr)) #also try and compare deltaGh based on Heiske2017
    
#    KmmADP_C5etc  = Kmm0ADP_C5etc*exp(gammaKa_C5etc*ksiKa_C5etc* (deltaGhC5/(R*T)) )
#    KmPi_C5etc  = Km0Pi_C5etc*exp(ksiKp_C5etc*(deltaGhC5/(R*T)))
#    KmmATP_C5etc  = Kmm0ATP_C5etc*exp(-(1.0-gammaKa_C5etc)*ksiKa_C5etc* (deltaGhC5/(R*T)) )
#    kf_C5etc = kf0_C5etc*exp(-gammaK_C5etc*ksiK_C5etc*deltaGhC5/(R*T))
#    kb_C5etc = kb0_C5etc*exp((1.0-gammaK_C5etc)*ksiK_C5etc*deltaGhC5/(R*T))
    
#    psiC5etc_a = (kf_C5etc*u[134]*u[136]/(KmmADP_C5etc*KmPi_C5etc)    - kb_C5etc*u[135]/KmmATP_C5etc    ) /     ( (1+ u[134]/KmmADP_C5etc + u[135]/KmmATP_C5etc ) * (1.0 + u[136]/KmPi_C5etc ) )

    
    #Korzeniewski2001 C5etc
    #deltaPH = (2.303*(R*T/F))*log10(C_H_ims/C_H_mitomatr)  # (2.303*(R*T/F))*(pHi-pHe), pHi,e = -log10(Hi,e /1000000) if Hi,e in uM  #check
    #naC5 = 2.5 #3.0 #2.67 #2.5
    #deltaGp0 = 31.9*1000.0 #J/mol #34-57
    
    #deltaGp = deltaGp0/F + (2.303*(R*T/F))*log10(u[135]/(u[134]+u[136]))
    
    #deltaGsn = naC5*(1.0/(1.0-0.861)*deltaPH) - deltaGp
    #gammaC5 = 10.0^(deltaGsn/ (2.303*(R*T/F)))
    
    #psiC5etc_a = 0.001*(34.316/60.0)*( (gammaC5  -1.0)/(gammaC5 +1.0)  )
    
    #Wu2007
    KeqC5etc0 = exp(-(deltaG0C5etc-naC5etc*F*u[160])/(R*T))

    KeqC5etc = KeqC5etc0*((C_H_ims^naC5etc)/(C_H_mitomatr^(naC5etc-1.0)))*(pAC5etc/(pAC5etc+pPiC5etc))

    psiC5etc_a = VmaxC5etc*( KeqC5etc*u[134]*u[136] - u[135]  )#/0.6 
    
    #Zhang2018
    psiANTetc = VmaxANTetc_a*( exp(betaANTetc*F*u[160]/(R*T))*(u[31]*u[135]/(K_ADP_ANTetc*K_ATP_ANTetc)) - exp((betaANTetc - 1.0)*F*u[160]/(R*T) )*(u[134]*u[29]/(K_ADP_ANTetc*K_ATP_ANTetc)) )  /  ( (1.0 + u[134]/K_ADP_ANTetc + (u[135]/K_ATP_ANTetc)*exp(betaANTetc*F*u[160]/(R*T))  )*( 1.0 + u[31]/K_ADP_ANTetc  + (u[29]/K_ATP_ANTetc)*exp((betaANTetc - 1.0)*F*u[160]/(R*T)  )    )   )
    #psiANTetc = VmaxANTetc_a*( exp(betaANTetc*F*u[160]/(R*T))*(u[31]*u[135]/(K_ADP_ANTetc*K_ATPm_ANTetc)) - exp((betaANTetc - 1.0)*F*u[160]/(R*T) )*(u[134]*u[29]/(K_ADP_ANTetc*K_ATP_ANTetc)) )  /  ( (1.0 + u[134]/K_ADP_ANTetc + (u[135]/K_ATPm_ANTetc)*exp(betaANTetc*F*u[160]/(R*T))  )*( 1.0 + u[31]/K_ADP_ANTetc  + (u[29]/K_ATP_ANTetc)*exp((betaANTetc - 1.0)*F*u[160]/(R*T)  )    )   )
    
    ################################################################################################################################
  
                                ################################################################################################################################
    
##################### MITO NEURON #####################
 
    
    # cytosol - mitochondria metabolic communication
    # Pyruvate cyto-mito exchange, PYR ⇒ PYRmito, based on Berndt 2015, Mulukutla 2015
    
    #Mulukutla2015
    #psiPYRtrcyt2mito_a = VmPYRtrcyt2mito_a*(u[23]*C_H_cyt_a - u[120]*C_H_mito_a)
    #adapted from Berndt2015
    #psiPYRtrcyt2mito_a = VmPYRtrcyt2mito_a*(u[23]*C_H_cyt_a/KmPyrCytTr - u[120]*C_H_mito_a/KmPyrMitoTr)/( (1+u[23]/KmPyrCytTr)*(1+u[120]/KmPyrMitoTr) )
    psiPYRtrcyt2mito_n = VmPYRtrcyt2mito_n*(u[22]*C_H_cyt_n  - u[84]*C_H_mito_n )/( (1+u[22]/KmPyrCytTr_n)*(1.0 +u[84]/KmPyrMitoTr_n) ) #analogy to astrocyte
    #or:
    # cytosol - mitochondria metabolic communication
    # Pyruvate cyto-mito exchange, PYR ⇒ PYRmito, based on Berndt 2015
    #VmaxPyrEx_n = 128.0 #### CHECK IT!!!! UNITS!!!! 
    #KmPyrIn_n = 0.15
    #KmPyrMito_n = 0.15
    #Hin_n = 7.01 #7.0-7.4 Wiki # # from NEDERGAARD 1991: 7.01/7.24 # hcyt/hext  # 
    #Hmito_n = 7.8 # Mito mattrix Wiki # check it more precisely    ############################################################# check!!
    #@reaction_func VPYRex(PYR,Hin,PYRmito,Hmito) = VmaxPyrEx*(PYR*Hin-PYRmito*Hmito) / ((1+PYR/KmPyrIn)*(1+PYRmito/KmPyrMito))
    #psiPYRex_n  = VmaxPyrEx_n*(u[22]*10^(-Hin_n)-u[84]*10^(-Hmito_n)) / ((1+u[22]/KmPyrIn_n)*(1+u[84]/KmPyrMito_n))

################################################################################################################################
#alternative    
#    VmaxPyrEx_n = 128.0 /60000 #### CHECK IT!!!! UNITS!!!! 
#    KmPyrIn_n = 0.15
#    KmPyrMito_n = 0.15
#    Hin_n = 7.01 #7.0-7.4 Wiki # # from NEDERGAARD 1991: 7.01/7.24 # hcyt/hext  # 
#    Hmito_n = 7.8 # Mito mattrix Wiki # check it more precisely    ############################################################# check!!
#    #@reaction_func VPYRex(PYR,Hin,PYRmito,Hmito) = VmaxPyrEx*(PYR*Hin-PYRmito*Hmito) / ((1+PYR/KmPyrIn)*(1+PYRmito/KmPyrMito))
#    psiPYRex_n  = VmaxPyrEx_n*(u[22]*Hin_n-u[84]*Hmito_n) / ((1+u[22]/KmPyrIn_n)*(1+u[84]/KmPyrMito_n))

    
    #TCA n
    #Berndt, Zhang2018
    psiPDH_n = VmaxPDHCmito_n*(1.0+AmaxCaMitoPDH_n*u[97]/(u[97] + KaCaMitoPDH_n)) * (u[84]/(u[84]+KmPyrMitoPDH_n)) * (u[95]/(u[95] + KmNADmitoPDH_n)) * (u[94]/(u[94] + KmCoAmitoPDH_n))
    
    #Mulukutla2015
    #alphaPDH_1 = 1.0 + u[93]/KiAcCoaPDH_n
    #alphaPDH_2 = 1.0 + u[96]/KiNADHPDH_n
    #psiPDH_n = VfPDH_n*( u[84]*u[94]*u[95] - u[93]*u[96]*CO2_mito_n/KeqPDH_n   ) /  ( KnadPDH_n*alphaPDH_1*u[84]*u[94] + KcoaPDH_n*alphaPDH_2*u[84]*u[95] + KpyrPDH_n*u[94]*u[95] + u[84]*u[94]*u[95] )
   
    
    psiCS_n = VmaxCSmito_n*(u[92]/(u[92] + KmOxaMito_n*(1.0 + u[85]/KiCitMito_n))) * (u[93]/(u[93] + KmAcCoAmito_n*(1.0+u[94]/KiCoA_n)))
 
    psiACO_n = VmaxAco_n*(u[85]-u[86]/KeqAco_n) / (1.0+u[85]/KmCit_n + u[86]/KmIsoCit_n)
    
          
    #Mulukutla 2015
    alpha_IDH_n = 1.0 + Ka_ndp_IDH_n*(1.0 +(0.1*u[99])/Ki_ntp_IDH_n)/(0.1*u[99]) # 90% ATP in mito are MgATP
    psiIDH_n = VmaxfIDH_n*( u[95]*u[86]^nH_IDH_n - (u[86]^(nH_IDH_n-1.0))*u[87]*u[96]*CO2_mito_n /Keq_IDH_n  ) / (   u[95]*u[86]^nH_IDH_n + (KmbIDH_n^nH_IDH_n)*alpha_IDH_n*u[95] +  KmaIDH_n*(u[86]^nH_IDH_n + (KibIDH_n^nH_IDH_n)*alpha_IDH_n + u[96]*(KibIDH_n^nH_IDH_n)*alpha_IDH_n/KiqIDH_n  ) )
    
    psiKGDH_n = VmaxKGDH_n*(1-u[97]/(u[97]+KiCaKGDH_n))*(u[87]/(u[87]+(Km1KGDHKGDH_n/(1+u[97]/KiAKGCaKGDH_n)+Km2KGDHKGDH_n)*(1+u[96]/KiNADHKGDHKGDH_n))) *   (u[95]/(u[95]+KmNADkgdhKGDH_n*(1+u[96]/KiNADHKGDHKGDH_n))) * (u[94]/(u[94] + KmCoAkgdhKGDH_n*(1+u[88]/KiSucCoAkgdhKGDH_n)))

    psiSCS_n  = VmaxSuccoaATP_n*(1+AmaxPscs_n*((u[100]^npscs_n)/((u[100]^npscs_n)+(Kmpscs_n^npscs_n)))) * (u[88]*u[98]*u[100] -  u[89]*u[94]*u[99]/Keqsuccoascs_n)/((1+u[88]/Kmsuccoascs_n)*(1+u[98]/KmADPscs_n)*(1+u[100]/KmPimitoscs_n)+(1+u[89]/Kmsuccscs_n)*(1+u[94]/Kmcoascs_n)*(1+u[99]/Kmatpmitoscs_n))

    #Mulukutla2015 
    alpha_SDH_n = (1.0 + u[92]/KiOXA_SDH_n + u[89]/KaSUC_SDH_n + u[90]/KaFUM_SDH_n ) / (1.0 + u[89]/KaSUC_SDH_n + u[90]/KaFUM_SDH_n  )  
    psiSDH_n = Vf_SDH_n*( u[89]*u[101] - u[102]*u[90]/Keq_SDH_n  ) / ( KiSUC_SDH_n*KmQ_SDH_n*alpha_SDH_n  + KmQ_SDH_n*u[89] + KmSuc_SDH_n*alpha_SDH_n*u[101] + u[89]*u[101] + KmSuc_SDH_n*u[101]*u[90]/KiFUM_SDH_n  +    (KiSUC_SDH_n*KmQ_SDH_n/(KiFUM_SDH_n*KmQH2_SDH_n) )*( KmFUM_SDH_n*alpha_SDH_n*u[102] + KmQH2_SDH_n*u[90] + KmFUM_SDH_n*u[89]*u[102]/KiSUC_SDH_n + u[102]*u[90] ) )     
    
    psiFUM_n = Vmaxfum_n*(u[90] - u[91]/Keqfummito_n)/(1.0+u[90]/Kmfummito_n+u[91]/Kmmalmito_n)

    psiMDH_n = VmaxMDHmito_n*(u[91]*u[95]-u[92]*u[96]/Keqmdhmito_n) / ((1.0+u[91]/Kmmalmdh_n)*(1.0+u[95]/Kmnadmdh_n)+(1.0+u[92]/Kmoxamdh_n)*(1.0+u[96]/Kmnadhmdh_n))

    

    ################################################################################################################################
    ## OXPHOS detailed neuron
    ################################################################################################################################
    ## OXPHOS detailed neuron
    # from Zhang2018 + Chang2011 + Heiske2017 + Wu2007...
    
    deltaGh_n = F*u[161] + R*T*log10(C_H_ims_n/C_H_mitomatr_n) #also try and compare deltaGh based on Heiske2017
    
    psiC1etc_n = VmaxC1etc_n*( u[96]*u[97]*exp(-betaC1etc_n *(4*deltaGh_n + Gibbs_C1etc_n)/(R*T)) - u[95]*u[102]*exp(-(betaC1etc_n-1.0)*(4*deltaGh_n + Gibbs_C1etc_n)/(R*T))  ) /  (Ka_C1etc_n*Kb_C1etc_n*(1.0 + u[96]/Ka_C1etc_n + u[95]/Kc_C1etc_n)*(1.0 + u[101]/Kb_C1etc_n + u[102]/Kd_C1etc_n)   )                                                                          
    
    psiC3etc_n = VmaxC3etc_n*( u[102]*(u[104]^2)*exp(betaC3etc_n*(-4*deltaGh_n +2*F*u[161] - Gibbs_C3etc_n )/(R*T) ) -  u[101]*(u[103]^2)*exp((betaC3etc_n-1.0)*(-4*deltaGh_n +2*F*u[161] - Gibbs_C3etc_n )/(R*T) )   )/    ( Ka_C3etc_n*(Kb_C3etc_n^2)*(1.0+u[102]/Ka_C3etc_n +u[101]/Kc_C3etc_n )*(1.0 + (u[104]^2)/(Kb_C3etc_n^2) + (u[103]^2)/(Kd_C3etc_n^2) ) )    
    
    psiC4etc_n = VmaxC4etc_n*( (u[103]^2)*(C_O2_mito_n^0.5)*exp(betaC4etc_n*( -2*deltaGh_n -2*F*u[161] - Gibbs_C4etc_n  )/(R*T)) - (u[104]^2)*exp((betaC4etc_n-1.0)*(-2*deltaGh_n -2*F*u[161] - Gibbs_C4etc_n)/(R*T))   ) /  ( (Ka_C4etc_n^2)*(Kb_C4etc_n^0.5)*(1.0 + (u[103]^2)/(Ka_C4etc_n^2) + (u[104]^2)/(Kc_C4etc_n^2) ) * (1.0 + (C_O2_mito_n^0.5)/(Kb_C4etc_n^0.5)) )
  
    #Wu2007
    KeqC5etc0_n = exp(-(deltaG0C5etc_n-naC5etc_n*F*u[161])/(R*T))

    KeqC5etc_n = KeqC5etc0_n*((C_H_ims_n^naC5etc_n)/(C_H_mitomatr_n^(naC5etc_n-1.0)))*(pAC5etc_n/(pAC5etc_n+pPiC5etc_n))

    psiC5etc_n = VmaxC5etc_n*( KeqC5etc_n*u[98]*u[100] - u[99]  )#/0.6 
    
    #Zhang2018
    psiANTetc_n = VmaxANTetc_n*( exp(betaANTetc_n*F*u[161]/(R*T))*(u[30]*u[99]/(K_ADP_ANTetc_n*K_ATP_ANTetc_n)) - exp((betaANTetc_n - 1.0)*F*u[161]/(R*T) )*(u[98]*u[28]/(K_ADP_ANTetc_n*K_ATP_ANTetc_n)) )  /  ( (1.0 + u[98]/K_ADP_ANTetc_n + (u[99]/K_ATP_ANTetc_n)*exp(betaANTetc_n*F*u[161]/(R*T))  )*( 1.0 + u[30]/K_ADP_ANTetc_n  + (u[28]/K_ATP_ANTetc_n)*exp((betaANTetc_n - 1.0)*F*u[161]/(R*T)  )    )   )
    #psiANTetc = VmaxANTetc_a*( exp(betaANTetc*F*u[160]/(R*T))*(u[31]*u[135]/(K_ADP_ANTetc*K_ATPm_ANTetc)) - exp((betaANTetc - 1.0)*F*u[160]/(R*T) )*(u[134]*u[29]/(K_ADP_ANTetc*K_ATP_ANTetc)) )  /  ( (1.0 + u[134]/K_ADP_ANTetc + (u[135]/K_ATPm_ANTetc)*exp(betaANTetc*F*u[160]/(R*T))  )*( 1.0 + u[31]/K_ADP_ANTetc  + (u[29]/K_ATP_ANTetc)*exp((betaANTetc - 1.0)*F*u[160]/(R*T)  )    )   )
    
    ################################################################################################################################
  
    
    ############ MAS (in neurons only)  # Berndt 2015


    psicMDH_n = VmaxcMDH_n*(u[107]*u[34]-u[108]*u[32]/Keqcmdh_n)/ ((1.0+u[107]/Kmmalcmdh_n)*(1.0+u[34]/Kmnadcmdh_n) + (1.0+u[108]/Kmoxacmdh_n)*(1.0+u[32]/Kmnadhcmdh_n)-1.0)  
    psiCAAT_n = VmaxcAAT_n*(u[109]*u[110]-u[108]*u[162]/KeqcAAT_n)
    
    #Berndt2015
    #psiAGC_n = Vmaxagc_n*(u[105]*u[162] - u[109]*u[106]/ (exp(u[161])^(F/(R*T))*  (C_H_cyt_n/C_H_mito_n)) ) / ((u[105]+Kmaspmitoagc_n)*(u[162]+KmgluCagc_n) + (u[109]+Kmaspagc_n)*(u[106]+KmgluMagc_n))
    
    #Mulukutla2015
    psiAGC_n = VmaxAGC_n*(KeqAGC_n*u[109]*u[106]*C_H_mito_n - u[105]*u[162]*C_H_cyt_n) /      ( KeqAGC_n*KiASPmAGC_n*KiGLUcAGC_n*KhAGC_n*( 2.0*mAGC_n + mAGC_n*u[109]/KiASPmAGC_n + u[109]*u[106]*C_H_mito_n/(KiASPmAGC_n*KiGLUcAGC_n*KhAGC_n) +     mAGC_n*u[105]*C_H_cyt_n/(KiASPcAGC_n*KhAGC_n)  + mAGC_n*u[105]/KiASPcAGC_n + u[105]*u[162]*C_H_cyt_n/(KiASPcAGC_n*KiGLUmAGC_n*KhAGC_n ) +         mAGC_n*u[109]*C_H_mito_n/(KiASPmAGC_n*KhAGC_n )  + mAGC_n*C_H_mito_n/KhAGC_n + mAGC_n*u[162]*C_H_cyt_n/(KiGLUmAGC_n*KhAGC_n) + mAGC_n*C_H_cyt_n/KhAGC_n + mAGC_n*u[106]*C_H_mito_n/(KiGLUcAGC_n*KhAGC_n)     )    )
    
    
    #Berndt
    #psiMAKGC_n = Vmaxmakgc_n*(u[107]*u[87] - u[91]*u[110]) / ((u[107]+Kmmalmakgc_n)*(u[87]+Kmakgmitomakgc_n)+(u[91]+Kmmalmitomakgc_n)*(u[110]+Kmakgmakgc_n))

    #Mulukutla2015
    #check!!!!!
    psiMAKGC_n = Vmax_makgc_n*( u[91]*u[110] - u[107]*u[87] )/     ( Km_akgmito_makgc_n*Km_mal_makgc_n*( 2.0 + u[107]/Km_mal_makgc_n + u[91]/Km_malmito_makgc_n + u[110]/Km_akg_makgc_n + u[87]/Km_akgmito_makgc_n + u[107]*u[87]/(Km_mal_makgc_n*Km_akgmito_makgc_n )   +  u[91]*u[110]/(Km_malmito_makgc_n*Km_akg_makgc_n)   )  )
    
    psiAAT_n  = VmaxmitoAAT_n*(u[105]*u[87]-u[92]*u[106]/KeqmitoAAT_n) # it was psiMAAT_n
    
    #AAT/GOT astrocyte mito  # was psiMAAT_n
    #alpha_AAT_n = 1.0 + u[87]/KiAKG_AAT_n
    #psiAAT_n = VfAAT_n*(u[105]*u[87] - u[92]*u[106]/KeqAAT_n) /  ( KmAKG_AAT_n*u[105] +  KmASP_AAT_n*alpha_AAT_n*u[87] + u[105]*u[87] + KmASP_AAT_n*u[87]*u[106]/KiGLU_AAT_n + (  KiASP_AAT_n*KmAKG_AAT_n/(KmOXA_AAT_n*KiGLU_AAT_n)  )*  ( KmGLU_AAT_n*u[105]*u[92]/KiASP_AAT_n + u[92]*u[106] +  KmGLU_AAT_n*alpha_AAT_n*u[92] + KmOXA_AAT_n*u[106] )  )
    
    
    
    #######
    # GLUmito - GLUcyto transp
    psiGLUTHtr_n = VmGLUH_n*(u[162]*C_H_cyt_n  - u[106]*C_H_mito_n)
    
    
    ####### GLS n
    psiGLS_n = VmGLS_n*( u[165] - 0.1*u[106]/KeqGLS_n )/ (KmGLNGLS_n*(1.0 + 0.1*u[106]/KiGLUGLS_n) + u[165]  ) # 0.1* is to account for compartmentation of GLU

    #GLN transporter n
    psiSNAT_GLN_n = TmaxSNAT_GLN_n*u[79]/(KmSNAT_GLN_n+u[79])
    
    ################################################################################################################################
      
    ############
    ##########################################################################
    # PPP: Stincone 2015; Nakayama 2005; Sabate 1995 rat liver; Kauffman1969 mouse brain; Mulukutla BC, Yongky A, Grimm S, Daoutidis P, Hu W-S (2015) Multiplicity of Steady States in Glycolysis and Shift of Metabolic State in Cultured Mammalian Cells. PLoS ONE; Cakir 2007 
    # check directions

    ### PPP neuron

    ############
    ##########################################################################
    # PPP: Stincone 2015; Nakayama 2005; Sabate 1995 rat liver; Kauffman1969 mouse brain; Mulukutla BC, Yongky A, Grimm S, Daoutidis P, Hu W-S (2015) Multiplicity of Steady States in Glycolysis and Shift of Metabolic State in Cultured Mammalian Cells. PLoS ONE; Cakir 2007 
    # check directions

    # Glucose 6-phosphate Dehydrogenase (G6PDH):  Glucose 6-phosphate(G6P) + NADP+(NADP) ↔ 6-phospho-glucono-1,5-lactone(GL6P) + NADPH + H+ ### from Sabate1995
    # ordered sequenttial bi-bi irreversible mechanism
    psiG6PDH_n = Vmax1G6PDHppp_n*u[113]*u[38]/ (KiNADPG6PDHppp_n*Kg6pG6PDHppp_n + Kg6pG6PDHppp_n*u[113] + KNADPG6PDHppp_n*u[38] + u[113]*u[38] + (Kg6pG6PDHppp_n*KiNADPG6PDHppp_n/KiNADPhG6PDHppp_n)*u[114] + (KNADPG6PDHppp_n/KiNADPhG6PDHppp_n)*u[38]*u[114] ) 

    # 6-Phosphogluconolactonase (6PGL): 6-Phosphoglucono-1,5-lactone(GL6P) + H2 O → 6-phosphogluconate(GO6P) ### from Sabate1995
    psi6PGL_n = (Vmax1f6PGLppp_n*u[111]/KsGL6Pppp_n - Vmax2r6PGLppp_n*u[112]/KsGO6Pppp_n) / (1+ u[111]/KsGL6Pppp_n + u[112]/KsGO6Pppp_n)

    # 6-Phosphogluconate Dehydrogenase (6PGDH): 6-Phosphogluconate(GO6P) + NADP+ → ribulose 5-phosphate(RU5P) + CO2 +NADPH+H+ ### from Sabate1995
    # ordered bi-ter sequential mechanism
    ########## check it!!!! especially (KGO6P6PGDHppp*KNADPh6PGDHppp*KiGO6P6PGDHppp*Kco26PGDHppp*KiRu5p6PGDHppp*KiNADPh6PGDHppp)*NADP*GO6P*CO2*RU5P
    psi6PGDH_n = (V16PGDHppp_n*(u[113]*u[112] - (V26PGDHppp_n/V16PGDHppp_n)*(KiNADP6PGDHppp_n*KGO6P6PGDHppp_n/(Kco26PGDHppp_n*KiRu5p6PGDHppp_n*KiNADPh6PGDHppp_n))*CO2_n*u[115]*u[114])) / (KiNADP6PGDHppp_n*KGO6P6PGDHppp_n  + KGO6P6PGDHppp_n*u[113] + KNADP6PGDHppp_n*u[112] + u[113]*u[112] + (KiNADP6PGDHppp_n*KGO6P6PGDHppp_n*KRu5p6PGDHppp_n/(Kco26PGDHppp_n*KiRu5p6PGDHppp_n))*CO2_n +   (KiNADP6PGDHppp_n*KGO6P6PGDHppp_n/(KiNADPh6PGDHppp_n*Kco26PGDHppp_n*KiRu5p6PGDHppp_n))*CO2_n*u[115]*u[114]  +(KGO6P6PGDHppp_n*KRu5p6PGDHppp_n/(KiGO6P6PGDHppp_n*Kco26PGDHppp_n*KiRu5p6PGDHppp_n))*u[113]*u[112]*CO2_n +   (KGO6P6PGDHppp_n*KiNADP6PGDHppp_n/(KiRu5p6PGDHppp_n*KiNADPh6PGDHppp_n))*u[115]*u[114] + (KGO6P6PGDHppp_n*KRu5p6PGDHppp_n/(KiRu5p6PGDHppp_n*Kco26PGDHppp_n))*u[113]*CO2_n +    (KNADP6PGDHppp_n/(KiRu5p6PGDHppp_n*KiNADPh6PGDHppp_n))*u[112]*u[115]*u[114] + (KGO6P6PGDHppp_n*KiNADP6PGDHppp_n*KNADPh6PGDHppp_n/(Kco26PGDHppp_n*KiRu5p6PGDHppp_n*KiNADPh6PGDHppp_n))*CO2_n*u[115] +   (KiNADP6PGDHppp_n*KGO6P6PGDHppp_n/KiNADPh6PGDHppp_n)*u[114] + (KNADP6PGDHppp_n/KiNADPh6PGDHppp_n)*u[112]*u[114] +(KiNADP6PGDHppp_n*KGO6P6PGDHppp_n*KRu5p6PGDHppp_n/(Kco26PGDHppp_n*KiRu5p6PGDHppp_n*KiNADPh6PGDHppp_n))*CO2_n*u[114]  +   (KGO6P6PGDHppp_n*Kico26PGDHppp_n*KNADPh6PGDHppp_n/(KiGO6P6PGDHppp_n*Kco26PGDHppp_n*KiRu5p6PGDHppp_n*KiNADPh6PGDHppp_n))*u[113]*u[112]*u[115] +  (KGO6P6PGDHppp_n*KNADPh6PGDHppp_n/(Kco26PGDHppp_n*KiRu5p6PGDHppp_n*KiNADPh6PGDHppp_n))*u[113]*CO2_n*u[115] +     (KGO6P6PGDHppp_n*KNADPh6PGDHppp_n*KiGO6P6PGDHppp_n*Kco26PGDHppp_n*KiRu5p6PGDHppp_n*KiNADPh6PGDHppp_n)*u[113]*u[112]*CO2_n*u[115]  +  (KNADP6PGDHppp_n/(Kico26PGDHppp_n*KiRu5p6PGDHppp_n*KiNADPh6PGDHppp_n))*u[112]*CO2_n*u[115]*u[114]  )   

    # Ribose Phosphate Isomerase (RPI): Ribulose 5-phosphate(RU5P) ↔ ribose 5- phosphate(R5P)
    # Michaelian reversible competitively inhibited by GO6P
    # check eq for rate
    psiRPI_n = V1rpippp_n*u[115]/(u[115] + Kru5prpippp_n*(1+u[112]/KiGO6Prpippp_n)) - V2rpippp_n*u[117]/(u[117] + Kr5prpippp_n*(1+u[112]/KiGO6Prpippp_n))


    # Ribulose Phosphate Epimerase (RPE): Ribulose 5-phosphate(RU5P) ↔ xylulose 5-phosphate(X5P)
    # check eq for rate
    psiRPEppp_n  = V1rpeppp_n*u[115]/(u[115] + Kru5prpeppp_n) - V2rpeppp_n*u[116]/(u[116] + Kx5prpeppp_n)

    # Transketolase (TKL1)
    #ping pong: R5P + X5P -> S7P + GAP
    psiTKL1_n = (K1tklppp_n*u[117]*u[116] + K2tklppp_n*u[39]*u[117] - K3tklppp_n*u[118]*u[45] - K4tklppp_n*u[118]*u[119]) / ( (K5tklppp_n*u[118] + K6tklppp_n*u[45] + K7tklppp_n*u[39] + K10tklppp_n*u[119] + K12tklppp_n*u[118]*u[45] + K13tklppp_n*u[118]*u[119] + K14tklppp_n*u[117]*u[116] + K18tklppp_n*u[45]*u[39] + K19tklppp_n*u[39]*u[119] )*(1+u[38]/ Kir5ptklppp_n )*(1+u[38]/ Kix5ptklppp_n) +  (K8tklppp_n + K11tklppp_n*u[118]+K15tklppp_n*u[39])*u[117]*(1+u[38]/Kdashr5ptklppp_n) + (K9tklppp_n+K16tklppp_n*u[45] + K17tklppp_n*u[119] )*u[116]*(1+u[38]/Kdashx5ptklppp_n) )

    # Transketolase (TKL2) 
    #ping pong: E4P + X5P -> F6P + GAP
    psiTKL2_n = (K20tklppp_n*u[116]*u[119] + K21tklppp_n*u[118]*u[119] - K22tklppp_n*u[39]*u[45] - K23tklppp_n*u[39]*u[117]) / ( (K5tklppp_n*u[118] + K6tklppp_n*u[45] + K7tklppp_n*u[39] + K10tklppp_n*u[119] +   K12tklppp_n*u[118]*u[45] + K13tklppp_n*u[118]*u[119] + K14tklppp_n*u[117]*u[116] + K18tklppp_n*u[45]*u[39] + K19tklppp_n*u[39]*u[119] )*(1+u[38]/ Kir5ptklppp_n )*(1+u[38]/ Kix5ptklppp_n) +  (K8tklppp_n + K11tklppp_n*u[118]+K15tklppp_n*u[39])*u[117]*(1+u[38]/Kdashr5ptklppp_n) + (K9tklppp_n+K16tklppp_n*u[45] + K17tklppp_n*u[119] )*u[116]*(1+u[38]/Kdashx5ptklppp_n) )

    # Transaldolase (TAL): S7P + GAP ↔ E4P + F6P
    psiTALppp_n = (V1talppp_n*(u[118]*u[45] - (V2talppp_n/V1talppp_n)*(Kis7ptalppp_n*Kgaptalppp_n/(Kf6ptalppp_n*Kie4ptalppp_n))*u[119]*u[39])) / (Kgaptalppp_n*u[118] + Ks7ptalppp_n*u[45] + u[118]*u[45] + (Kis7ptalppp_n*Kgaptalppp_n/Kie4ptalppp_n)*u[119] + (Kis7ptalppp_n*Kgaptalppp_n/(Kf6ptalppp_n*Kie4ptalppp_n))*u[119]*u[39] + (Kgaptalppp_n*Kis7ptalppp_n*Ke4ptalppp_n/(Kie4ptalppp_n*Kf6ptalppp_n))*u[39] +  (Kgaptalppp_n/Kie4ptalppp_n)*u[118]*u[119]  + (Ks7ptalppp_n/Kif6ptalppp_n)*u[45]*u[39]   )



    ################################################################################################################################
      
    
    ### PPP astrocyte
    

    ############
    ##########################################################################
    # PPP: Stincone 2015; Nakayama 2005; Sabate 1995 rat liver; Kauffman1969 mouse brain; Mulukutla BC, Yongky A, Grimm S, Daoutidis P, Hu W-S (2015) Multiplicity of Steady States in Glycolysis and Shift of Metabolic State in Cultured Mammalian Cells. PLoS ONE; Cakir 2007 
    # check directions

    # Glucose 6-phosphate Dehydrogenase (G6PDH):  Glucose 6-phosphate(G6P) + NADP+(NADP) ↔ 6-phospho-glucono-1,5-lactone(GL6P) + NADPH + H+ ### from Sabate1995
    # ordered sequenttial bi-bi irreversible mechanism
    psiG6PDH_a = Vmax1G6PDHppp_a*u[149]*u[40]/ (KiNADPG6PDHppp_a*Kg6pG6PDHppp_a + Kg6pG6PDHppp_a*u[149] + KNADPG6PDHppp_a*u[40] + u[149]*u[40] + 
        (Kg6pG6PDHppp_a*KiNADPG6PDHppp_a/KiNADPhG6PDHppp_a)*u[150] + (KNADPG6PDHppp_a/KiNADPhG6PDHppp_a)*u[40]*u[150] ) 

    # 6-Phosphogluconolactonase (6PGL): 6-Phosphoglucono-1,5-lactone(GL6P) + H2 O → 6-phosphogluconate(GO6P) ### from Sabate1995
    psi6PGL_a = (Vmax1f6PGLppp_a*u[147]/KsGL6Pppp_a - Vmax2r6PGLppp_a*u[148]/KsGO6Pppp_a) / (1+ u[147]/KsGL6Pppp_a + u[148]/KsGO6Pppp_a)

    # 6-Phosphogluconate Dehydrogenase (6PGDH): 6-Phosphogluconate(GO6P) + NADP+ → ribulose 5-phosphate(RU5P) + CO2 +NADPH+H+ ### from Sabate1995
    # ordered bi-ter sequential mechanism
    ########## check it!!!!   CO2concentration in astrocytes different from the one in this paper!!!!! check  especially (KGO6P6PGDHppp*KNADPh6PGDHppp*KiGO6P6PGDHppp*Kco26PGDHppp*KiRu5p6PGDHppp*KiNADPh6PGDHppp)*NADP*GO6P*CO2*RU5P
    psi6PGDH_a = (V16PGDHppp_a*(u[149]*u[148] - (V26PGDHppp_a/V16PGDHppp_a)*(KiNADP6PGDHppp_a*KGO6P6PGDHppp_a/(Kco26PGDHppp_a*KiRu5p6PGDHppp_a*KiNADPh6PGDHppp_a))*CO2_a*u[151]*u[150])) / 
    (KiNADP6PGDHppp_a*KGO6P6PGDHppp_a  + KGO6P6PGDHppp_a*u[149] + KNADP6PGDHppp_a*u[148] + u[149]*u[148] + (KiNADP6PGDHppp_a*KGO6P6PGDHppp_a*KRu5p6PGDHppp_a/(Kco26PGDHppp_a*KiRu5p6PGDHppp_a))*CO2_a +  
        (KiNADP6PGDHppp_a*KGO6P6PGDHppp_a/(KiNADPh6PGDHppp_a*Kco26PGDHppp_a*KiRu5p6PGDHppp_a))*CO2_a*u[151]*u[150]  +
        (KGO6P6PGDHppp_a*KRu5p6PGDHppp_a/(KiGO6P6PGDHppp_a*Kco26PGDHppp_a*KiRu5p6PGDHppp_a))*u[149]*u[148]*CO2_a +   
        (KGO6P6PGDHppp_a*KiNADP6PGDHppp_a/(KiRu5p6PGDHppp_a*KiNADPh6PGDHppp_a))*u[151]*u[150] + (KGO6P6PGDHppp_a*KRu5p6PGDHppp_a/(KiRu5p6PGDHppp_a*Kco26PGDHppp_a))*u[149]*CO2_a +    
        (KNADP6PGDHppp_a/(KiRu5p6PGDHppp_a*KiNADPh6PGDHppp_a))*u[148]*u[151]*u[150] + (KGO6P6PGDHppp_a*KiNADP6PGDHppp_a*KNADPh6PGDHppp_a/(Kco26PGDHppp_a*KiRu5p6PGDHppp_a*KiNADPh6PGDHppp_a))*CO2_a*u[151] +   
        (KiNADP6PGDHppp_a*KGO6P6PGDHppp_a/KiNADPh6PGDHppp_a)*u[150] + (KNADP6PGDHppp_a/KiNADPh6PGDHppp_a)*u[148]*u[150] +
        (KiNADP6PGDHppp_a*KGO6P6PGDHppp_a*KRu5p6PGDHppp_a/(Kco26PGDHppp_a*KiRu5p6PGDHppp_a*KiNADPh6PGDHppp_a))*CO2_a*u[150]  +   
        (KGO6P6PGDHppp_a*Kico26PGDHppp_a*KNADPh6PGDHppp_a/(KiGO6P6PGDHppp_a*Kco26PGDHppp_a*KiRu5p6PGDHppp_a*KiNADPh6PGDHppp_a))*u[149]*u[148]*u[151] +  
        (KGO6P6PGDHppp_a*KNADPh6PGDHppp_a/(Kco26PGDHppp_a*KiRu5p6PGDHppp_a*KiNADPh6PGDHppp_a))*u[149]*CO2_a*u[151] +     
        (KGO6P6PGDHppp_a*KNADPh6PGDHppp_a*KiGO6P6PGDHppp_a*Kco26PGDHppp_a*KiRu5p6PGDHppp_a*KiNADPh6PGDHppp_a)*u[149]*u[148]*CO2_a*u[151]  +  
        (KNADP6PGDHppp_a/(Kico26PGDHppp_a*KiRu5p6PGDHppp_a*KiNADPh6PGDHppp_a))*u[148]*CO2_a*u[151]*u[150]  )   

    # Ribose Phosphate Isomerase (RPI): Ribulose 5-phosphate(RU5P) ↔ ribose 5- phosphate(R5P)
    # Michaelian reversible competitively inhibited by GO6P
    # check eq for rate
    psiRPI_a = V1rpippp_a*u[151]/(u[151] + Kru5prpippp_a*(1+u[148]/KiGO6Prpippp_a)) - V2rpippp_a*u[153]/(u[153] + Kr5prpippp_a*(1+u[148]/KiGO6Prpippp_a))


    # Ribulose Phosphate Epimerase (RPE): Ribulose 5-phosphate(RU5P) ↔ xylulose 5-phosphate(X5P)
    # check eq for rate
    psiRPEppp_a  = V1rpeppp_a*u[151]/(u[151] + Kru5prpeppp_a) - V2rpeppp_a*u[152]/(u[152] + Kx5prpeppp_a)

    # Transketolase (TKL1)
    #ping pong: R5P + X5P -> S7P + GAP
    psiTKL1_a = (K1tklppp_a*u[153]*u[152] + K2tklppp_a*u[41]*u[153] - K3tklppp_a*u[154]*u[47] - K4tklppp_a*u[154]*u[155]) / 
    ( (K5tklppp_a*u[154] + K6tklppp_a*u[47] + K7tklppp_a*u[41] + K10tklppp_a*u[155] + K12tklppp_a*u[154]*u[47] + K13tklppp_a*u[154]*u[155] + 
            K14tklppp_a*u[153]*u[152] + K18tklppp_a*u[47]*u[41] + K19tklppp_a*u[41]*u[155] )*(1+u[40]/ Kir5ptklppp_a )*(1+u[40]/ Kix5ptklppp_a) +  
        (K8tklppp_a + K11tklppp_a*u[154]+K15tklppp_a*u[41])*u[153]*(1+u[40]/Kdashr5ptklppp_a) + (K9tklppp_a+K16tklppp_a*u[47] + K17tklppp_a*u[155] )*u[152]*(1+u[40]/Kdashx5ptklppp_a) )

    # Transketolase (TKL2) 
    #ping pong: E4P + X5P -> F6P + GAP
    psiTKL2_a = (K20tklppp_a*u[152]*u[155] + K21tklppp_a*u[154]*u[155] - K22tklppp_a*u[41]*u[47] - K23tklppp_a*u[41]*u[153]) / ( (K5tklppp_a*u[154] + K6tklppp_a*u[47] + 
            K7tklppp_a*u[41] + K10tklppp_a*u[155] +   K12tklppp_a*u[154]*u[47] + K13tklppp_a*u[154]*u[155] + K14tklppp_a*u[153]*u[152] + K18tklppp_a*u[47]*u[41] + 
            K19tklppp_a*u[41]*u[155] )*(1+u[40]/ Kir5ptklppp_a )*(1+u[40]/ Kix5ptklppp_a) +  (K8tklppp_a + K11tklppp_a*u[154]+K15tklppp_a*u[41])*u[153]*(1+u[40]/Kdashr5ptklppp_a) + 
        (K9tklppp_a+K16tklppp_a*u[47] + K17tklppp_a*u[155] )*u[152]*(1+u[40]/Kdashx5ptklppp_a) )

    # Transaldolase (TAL): S7P + GAP ↔ E4P + F6P
    psiTALppp_a = (V1talppp_a*(u[154]*u[47] - (V2talppp_a/V1talppp_a)*(Kis7ptalppp_a*Kgaptalppp_a/(Kf6ptalppp_a*Kie4ptalppp_a))*u[155]*u[41])) / (Kgaptalppp_a*u[154] + 
        Ks7ptalppp_a*u[47] + u[154]*u[47] + (Kis7ptalppp_a*Kgaptalppp_a/Kie4ptalppp_a)*u[155] + (Kis7ptalppp_a*Kgaptalppp_a/(Kf6ptalppp_a*Kie4ptalppp_a))*u[155]*u[41] + 
        (Kgaptalppp_a*Kis7ptalppp_a*Ke4ptalppp_a/(Kie4ptalppp_a*Kf6ptalppp_a))*u[41] +  (Kgaptalppp_a/Kie4ptalppp_a)*u[154]*u[155]  + (Ks7ptalppp_a/Kif6ptalppp_a)*u[47]*u[41]   )



 # changing conc of Na and K
    du[7] = 0 #(1/tau) * (-gamma*INa - 3*JpumpNa) # check if INa accessible for it; Calvetti2018 # Nain # p[4] = Na_in from ndam !!! modified by Polina on 29 nov 2019
    du[8] = 0 #(1/tau) * (gamma*beta*IK - 2*beta*JpumpNa - JgliaK - JdiffK)   # check if IK accessible for it; Calvetti2018 # Kout

    
    du[13] = (1/eto_ecs) * (JGlc + (-1)*jGLCtr_a  +(-1)*jGLCtr_n) #Glc_ecs ####################################
    du[14] = (1/eto_ecs) * (JLac + (-1)*jLac_a + (-1)*jLac_n  )  # Lac_ecs
    #du[15] = JO2 - jO2_a # O2_ecs
    
    du[15] = (1/eto_ecs) * ( JO2 - jO2_n - jO2_a)  #(1/eto_ecs) * () # O2_ecs

    du[16] = 0# (1/eto_n) * (jO2_n + (-1)*psiOxphos_n)  # O2_n
    du[17] = 0# (1/eto_a) * (jO2_a + (-1)*psiOxphos_a) #(1/eto_a) * ()# O2_a
    
    
    du[18] = jGLCtr_n + (-1)*psiHK_n ## Glc_n 
    du[19] =  jGLCtr_a +(-1)*psiHK_a  ## Glc_a 
    
    du[20] = jLac_n + psiLDH_n  # LAC_n   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    du[21] = jLac_a + psiLDH_a  # LAC_a   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    du[22] = psiPK_n   + (-1)*psiLDH_n  + (-1)*psiPYRtrcyt2mito_n #PYR_n
    
    du[23] = psiPK_a   + (-1)*psiLDH_a  + (-1)*psiPYRtrcyt2mito_a #PYR_a
    
    du[24] = (-1)*psiPCr_n + psiCr_n # u25 = PCr_a
    du[26] = (-1)*psiCr_n + psiPCr_n # u27 = Cr_a
  
    du[25] = (-1)*psiPCr_a + psiCr_a # u25 = PCr_a
    du[27] = (-1)*psiCr_a + psiPCr_a # u27 = Cr_a
  
    du[28] = (-1)*psiHK_n + (-1)*psiPFK_n + psiPGK_n + psiPK_n + psiPCr_n + (-1)*psiCr_n + (-1)*psiNKA_n +psiANTetc_n
    du[30] = psiHK_n + psiPFK_n + (-1)*psiPGK_n + (-1)*psiPK_n + (-1)*psiPCr_n + psiCr_n + psiNKA_n -psiANTetc_n
    
    du[29] = (-1)*psiHK_a + (-1)*psiPFK_a + (-1)*psiPFK2_a + psiPGK + psiPK_a + psiPCr_a + (-1)*psiCr_a + (-1)*psiNKA_a     #ATP_a
    du[31] = psiHK_a + psiPFK_a + psiPFK2_a + (-1)*psiPGK + (-1)*psiPK_a + (-1)*psiPCr_a + psiCr_a + psiNKA_a        #ADP_a
    
    
    
    du[32] =  2*psiGAPDH_n + (-1)*psiLDH_n -psicMDH_n #(1/eto_n) * () + 5*psiTCA_n + (-2)*psiOxphos_n  # u32 = NADH_n  
    du[33] =  2*psiGAPDH + (-1)*psiLDH_a
    
    du[34] = -2*psiGAPDH_n + psiLDH_n +psicMDH_n  #(1/eto_n) * () -5*psiTCA_n + 2*psiOxphos_n  # u34 = NAD_n
    du[35] = -2*psiGAPDH + psiLDH_a 
    
    du[38] = psiHK_n + (-1)*psiPGI_n
    du[39] = psiPGI_n + (-1)*psiPFK_n
    
    du[40] = psiPGLM_a + (-1)*psiPGI_a + psiHK_a                       ## G6P_a
    du[41] =  psiPGI_a + (-1)*psiPFK_a +psiFBPFK2_a + (-1)*psiPFK2_a   ## F6P_a    ########################### 0.3*(-1)*psiPFK_a
    du[42] = psiPFK_n + (-1)*psiALD_n
    
    du[43] =  psiPFK_a  +  (-1)*psiALD_a   ###-psiALD_a but deltaG>0  ## FBP_a   ########  psiPFK_a  >>  psiALD_a -> disbalance, growth of u[43]  # 0.3*psiPFK_a
    
    du[45] =  - psiGAPDH_n  + psiALD_n + psiTPI_n 
    du[46] = psiALD_n + (-1)*psiTPI_n 
    
    du[47] =  - psiGAPDH  + psiTPI  + psiALD_a  #     #psiALD_a but deltaG>0     ## GAP_a
    du[48] = psiALD_a + (-1)*psiTPI    #psiALD_a but deltaG>0 ## DNAP_a
    
    du[50] = psiGAPDH_n  + (-1)*psiPGK_n 
    du[52] = psiGAPDH + (-1)*psiPGK                 ## BPG13_a
    
    du[53] = psiPGK_n + (-1)*psiPGM_n   
    du[54] = psiPGK + (-1)*psiPGM_a                                 ## PG3_a
    du[55] = psiPGM_n  + (-1)*psiENOL_n 
    du[56] = psiPGM_a  + (-1)*psiENOL_a             ## PG2_a
    
    du[57] =  psiENOL_n + (-1)*psiPK_n             ## PEP_a
    
    du[58] =  psiENOL_a + (-1)*psiPK_a             ## PEP_a
    
    du[59] = psiGPa_a + psiGPb_a + (-1)*psiPGLM_a + (-1)*psiUDPGP_a   # G1P_a
    du[60] = (-1)*psiGPa_a + (-1)*psiGPb_a + psiGS_a   # GLY_a
    du[61] = (-1)*psiGS_a + psiUDPGP_a #  UDPgluco_a
    
    du[62] = (-1)*psiUDPGP_a #(1/eto_a) * ( ) # u62 = UTP
    du[63]  = psiPP1act #(1/eto_a) * () # u63 = PP1_a
    du[64]  = psiGSAJay  # u64 = GS_a
    du[65]  =  psiPHK #(1/eto_a) * () # u65 = GPa_a
    du[66]  =  (-1)*psiPHK # u66 = GPb_a
    du[67]  = (-1)*psiPDE + psiAC + (-1)*psiPKA1  + (-1)*psiPKA2  # u67 = cAMP_a
    du[68]  =  psiPKA1 + psiPKA2 # u68 = PKAa_a
    du[69]  = (-1)*psiPKA1  # u69 = PKAb_a
    du[70]  = psiPHKact  # u70 = PHKa_a
    du[71]  =  psiPKA1 + (-1)*psiPKA2  # u71 = R2CcAMP2_a
    du[72]  = psiPKA2  # u72 = R2CcAMP4_a

    
    du[73] = 0 #INKAastK  - JKirAV  - psiEAAT12 #+ JNKCC # + JKirAS/eto_ecs   #K_a
    
    #workedwell  but slow growth of u[74]   du[74] =  -  (p_a_ratio/(mu_pump_ephys + p_a_ratio)) * INKAastNa   +  JNKCC  - 3*JNCX #/eto_ecs  # *3
    du[74] = 0 # -  (p_a_ratio/(mu_pump_ephys + p_a_ratio)) * INKAastNa /eto_a   - 3*JNCX   + 3*psiEAAT12 #+  JNKCC    #/eto_ecs  # *3  #workedwell  
    
    du[75] = 0 # (1/Cast) * ( - (p_a_ratio/(mu_pump_ephys + p_a_ratio)) * INKAastNa  +   INKAastK   - JKirAV - JNCX  + psiEAAT12 )  # +2.0*JNKCC/Cast  #JNCX*(-3/eto_ecs + 2/eto_a)  # +JNCX because +3Na-Ca2+ #+ JKirAS*eto_ecs ) #  + IleakAst*eto_a ) # - JKirAS  - JKirAV - IleakAst  )  # 2.0*JNKCC*eto_ecs  because 2 ions: Na and K go in
    du[76] = 0 # JNCX #/eto_a  # Ca_a
    
    du[77] = 0  # u77 = GLUT_out  ECS is big space, consider synaptic compartment separately  
    
    
    # GLU-GLN
    du[78] = psiEAAT12  - u[78]*0.2 -psiGDH_simplif_a -psiGLNsynth_a # - u[78]/0.2  is decay due to diffusion and side reactions # u78 = GLUT_a
    du[79] = psiSNAT_GLN_a - psiSNAT_GLN_n   
    du[80] =  -psiSNAT_GLN_a +psiGLNsynth_a
    
    du[81] = 0 # affected by callback  -psiEAAT12   # u81 = GLUT_syn #### Breslin2018 microdomains & Flanagan2018  ### affected by callback of presyn release of Glut
    
    du[82] =  psiPFK2_a + (-1)*psiFBPFK2_a
    
    du[83] = 0 #no pfk2fb in neurons !!!!!!!!!!!!!!!
    
   #TCA n 
   #for all TCA was 0.1*()
    du[84] = -psiPDH_n +psiPYRtrcyt2mito_n # psiPYRex_n # PYRmito_n
    du[85] =   psiCS_n - psiACO_n # CITmito_n
    du[86] =  psiACO_n -psiIDH_n  # ISOCITmito_n
    du[87] =   - psiKGDH_n  + psiMAKGC_n  + psiAAT_n  + psiIDH_n # AKGmito_n
    du[88] =  psiKGDH_n -psiSCS_n # SUCCOAmito_n
    du[89] =  - psiSDH_n + psiSCS_n  # SUCmito_n
    du[90] =  psiSDH_n - psiFUM_n # FUMmito_n0
    du[91] =  psiFUM_n - psiMDH_n  - psiMAKGC_n # MALmito_n0
    du[92] =   -psiCS_n  + psiMDH_n -psiAAT_n  # OXAmito_n0
    du[93] =  -psiCS_n + psiPDH_n # AcCoAmito_n0
    du[94] = -psiPDH_n - psiKGDH_n +psiSCS_n  # CoAmito_n0
    du[95] =  -psiPDH_n  - psiKGDH_n +psiC1etc_n  -psiMDH_n -psiIDH_n  # NADmito_n0
    du[96] = psiPDH_n + psiKGDH_n - psiC1etc_n  + psiMDH_n + psiIDH_n   # NADHmito_n0
    du[97] =   0.0 # CaMito_n0
    du[98] =   -psiSCS_n  -psiC5etc_n +psiANTetc_n # ADPmito_n0
    du[99] =   psiSCS_n + psiC5etc_n -psiANTetc_n  # ATPmito_n0
    du[100] = 0.0 # Pimito_n0 
    du[101] =   -psiC1etc_n + psiC3etc_n -psiSDH_n  # Qmito_n0
    du[102] =   psiC1etc_n -psiC3etc_n +psiSDH_n   # QH2mito_n0
    du[103] =   psiC3etc_n -psiC4etc_n # CytCredmito_n0
    du[104] =  -psiC3etc_n +psiC4etc_n # CytCoxmito_n0
    du[105] =  psiAAT_n  -psiAGC_n  # ASPmito_n0
    
    du[106] =  -psiAAT_n +psiAGC_n + psiGLUTHtr_n + psiGLS_n #   # GLUmito_n0
    
    du[107] =  psiMAKGC_n + psicMDH_n # MAL_n0 # check directions
    du[108] =  psiCAAT_n -psicMDH_n   # OXA_n0  # check directions
    du[109] =  -psiCAAT_n  + psiAGC_n # ASP_n0
    du[110] =  - psiMAKGC_n -psiCAAT_n  # AKG_n0
    
   
    #### PPP n

    #all ppp were with 0.1*()
    du[111] = psiG6PDH_n -psi6PGL_n #  GL6P_n0
    du[112] = psi6PGL_n - psi6PGDH_n #GO6P_n
    du[113] = -psiG6PDH_n -psi6PGDH_n   # NADP_n0
    du[114] = psiG6PDH_n + psi6PGDH_n # NADPH_n0
    du[115] = psi6PGDH_n - psiRPI_n - psiRPEppp_n # RU5P_n0
    du[116] = psiRPEppp_n -psiTKL1_n -psiTKL2_n # X5P_n0
    du[117] = psiRPI_n-psiTKL1_n # R5P_n0
    du[118] = psiTKL1_n-psiTALppp_n  #  S7P_n
    du[119] = -psiTKL2_n +psiTALppp_n # E4P_n0
 
    
 #########################################################   
    
    
     
    
    du[120] =  psiPYRtrcyt2mito_a + (-1)*psiPDH_a -psiPYRCARB_a  #PYRmito_a  ################################################
    
    du[121] = psiCS_a - psiACO_a # CITmito_a
    du[122] = psiACO_a-psiIDH_a  # ISOCITmito_a
    du[123] = psiIDH_a - psiKGDH_a + psiGDH_simplif_a -psiAAT_a # + psiMAAT_a - psiMAKGC_a # AKGmito_a
    du[124] = psiKGDH_a -psiSCS_a # SUCCOAmito_a
    
    du[125] = - psiSDH_a + psiSCS_a # SUCmito_a
    
    
    du[126] =  psiSDH_a - psiFUM_a # FUMmito_a0
    du[127] =  psiFUM_a - psiMDH_a #+ psiMAKGC_a # MALmito_a0
    du[128] = psiPYRCARB_a -psiCS_a + psiMDH_a +psiAAT_a  #-psiMAAT_a #+ 0.5*psiMDH_a # OXAmito_a0
   
    du[129] =  -psiCS_a + psiPDH_a # AcCoAmito_a0
    du[130] =   -psiPDH_a - psiKGDH_a +psiSCS_a # CoAmito_a0
    
    
    
    du[131] =  psiC1etc_a  -psiPDH_a - psiIDH_a - psiKGDH_a # -psiMDH_a  # NADmito_a0  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! psiOxphos_a tmp replace for psiC1etc_a
    
    du[132] =  - psiC1etc_a + psiPDH_a + psiIDH_a + psiKGDH_a #  + psiMDH_a  # NADHmito_a0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! psiOxphos_a tmp replace for psiC1etc_a
    
    du[133] = 0.0 #  JNCX(check sign, direction)!!!!!!! +J_Dash2009_CaUniporter !!!!!!!!!!!!!!!! 0.0 # CaMito_a0
    
    du[134] = -psiSCS_a -psiC5etc_a +psiANTetc
    du[135] =  psiSCS_a + psiC5etc_a -psiANTetc  # ATPmito_a
    
    du[136] = 0.0 #  0.0 # Pimito_a0 
    
    du[137] =  psiC3etc_a - psiC1etc_a - psiSDH_a # Qmito_a0  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    du[138] =  psiC1etc_a -psiC3etc_a + psiSDH_a #+  QH2mito_a0  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    du[139] =  psiC3etc_a -psiC4etc_a # CytCredmito_a0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

    du[140] =  -psiC3etc_a +psiC4etc_a # CytCoxmito_a0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

   
    
    #many of following are 0 because MAS not active in astrocytes
    du[141] = -psiAAT_a # - psiAGC_a #psiMAAT_a # ASPmito_a0
    du[142] = psiAAT_a # psiAGC_a # -psiMAAT_a  # GLUmito_a0
    du[143] = 0 # -psiMAKGC_a #+ 0.5*psicMDH_a # MAL_a0
    du[144] = 0 # psiCAAT_a #-0.5*psicMDH_a   # OXA_a0
    du[145] = 0 #-psiCAAT_a  + psiAGC_a # ASP_a0
    du[146] = 0 #-psiCAAT_a  # + psiMAKGC_a) # AKG_a
    
    # PPP astro is almost not active for energy prod
    du[147] = 0.1*(psiG6PDH_a -psi6PGL_a) #  GL6P_a0
    du[148] = 0.1*(psi6PGL_a - psi6PGDH_a) #GO6P_a
    du[149] = 0.1*(-psiG6PDH_a -psi6PGDH_a)   # NADP_a0
    du[150] = 0.1*(psiG6PDH_a + psi6PGDH_a) # NADPH_a0
    du[151] = 0.1*(psi6PGDH_a - psiRPI_a - psiRPEppp_a) # RU5P_a0
    du[152] = 0.1*(psiRPEppp_a -psiTKL1_a -psiTKL2_a) # X5P_a0
    du[153] = 0.1*(psiRPI_a-psiTKL1_a) # R5P_a0
    du[154] = 0.1*(psiTKL1_a-psiTALppp_a)  #  S7P_a
    du[155] = 0.1*(-psiTKL2_a +psiTALppp_a) # E4P_a0
     

    
# u156 = GSH_a
# u157 = GSSG_a
    du[156] =  - psiGPX_a  + psiGSSGR_a  # GSH_a
    du[157] = psiGPX_a - psiGSSGR_a  # GSSG_a
   
    du[158] =  - psiGPX_n  + psiGSSGR_n  # GSH_n
    du[159] = psiGPX_n - psiGSSGR_n  # GSSG_n
   
    #du[160] 
    
    du[162] =  -psiAGC_n -psiGLUTHtr_n #+ GLUmitocyt_tr_n   #GLUcyto_n
    
    du[165] = -psiGLS_n + psiSNAT_GLN_n #GLN_n
    
 



    end""") # !!! remove # from here, it was temporary measure for readability
    return metabolism

##############################################################
# steady state/inf/initial values/u0


### initial conditions
VNeu0 = -70.0 #mV # -56.1999 # Calvetti2018
Va0 = -90.0 # mV  # -0.09 V # Breslin2018 microdomains

m0 =  0.1*(VNeu0+30.0)/(1.0-np.exp(-0.1*(VNeu0+30.0))) / (  0.1*(VNeu0+30.0)/(1.0-np.exp(-0.1*(VNeu0+30.0)))    +     4.0*np.exp(-(VNeu0+55.0)/18.0) )  # (alpha_m + beta_m)
h0 = 0.9002
n0 = 0.1558





ksi0 = 0.06

Conc_Cl_out = 130.0 #140 mM # Novel determinants of the neuronal Cl− concentration Eric Delpire 2014 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4215762/
Conc_Cl_in =  8.0 # mM # Novel determinants of the neuronal Cl− concentration Eric Delpire 2014 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4215762/

Na0in = 11.5604 # mM # 
K0out =  4.1 #mM +-1.8 (mouse,rat) - Takaneshi, Manaka, Sano 1981 #  6.2773 # mM 
K_a0 = 52.0 #Witthoft2013 # 100.0 #Flanagan2018  #  # 130.0 # approx  ## in intracellular K+ concentration [K+]in (from 110 to ~113 mM) dissipated over several seconds after [K+]out returned to 3 mM 
Na_a0 = 17.0 #Witthoft2013 
Ca_a0 = 75e-6 #75 nM # Physiology of Astroglia. Verkhratsky and Nedergaard 2018 # [Ca2+]i of 50−80 nM and [Na+]i of 15 mM, the ENCX could be as negative as about −85 to −90 mV, being thus very close (or even slightly more negative) to the resting membrane potential of astrocytes.

O2_ecs = 0.04 # mM Calvetti2018

#astrocyte
Glc_a = 1.2 #1.25  # ~1.5-2.5 mM From Molecules to Networks 2014 ######################################################
ATP_a = 1.4 #2.17  # or 1.4 ?  # atpc/adpc = 29 -> for 1.4 atp -> adp=0.05
ADP_a = 0.03 #0.1 # 0.03 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

NADH_a = 0.0007 #0.075  # Nad/NADH = 670-715 
NAD_a = 0.5

Glc_ecs = 1.3 #1.25 ####################################### try to split ECS_n and ECS_a see fig 2 in Felipe Barros 2017 https://doi.org/10.1002/jnr.23998 
Lac_ecs = 1.4 #1.3 #Calvetti

O2_a = 0.03 # mM Calvetti2018

G6P_a = 0.2 # #0.675 #mM-Park2016-worked #0.53 #0.75 # 0.53 also ok # 0.06-0.2 mM From Molecules to Networks 2014

F6P_a = 0.02 # 0.01 - 0.02 mM From Molecules to Networks 2014  #0.0969#-worked #mM -Park2016 # 0.228 mM -Lambeth2002 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
f26bp_a0 = 0.015  #0.005-0.025 #mM #Mulukutla supp fig plot 6

FBP_a = 0.1 # 0.01 - 0.1 mM From Molecules to Networks 2014  #1.52 #-worked #0.0723 # Lambeth # Jay Glia expand  # 1.52 #mM -Park2016 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Lac_a = 1.4 #1.3 # mM Calvetti2018; Lambeth   ## 0.602393 # Jay 181130 

AMP_a = 0.01 #2e-5 #0.03 #-Mulukutla2015 #2e-5 # Lambeth # 0.01 # 2e-5 # 1e-5 # from 1e-5 to 0.05
Pi_a = 4.1 # Lambeth # 40.0 # 4.1 # 31.3 # 4.1 # 4.0 Anderson&Wright 1979 # wide range from 1 to 40 mM

GAP_a = 0.05 # 0.04 - 0.05 mM From Molecules to Networks 2014  # 0.141 #mM -Park2016  #0.0355 # Lambeth # 0.0574386 # 0.0574386 is from latest Jay's data; it was 0.0046 in Jay Glia expand  !!!!!!!!!!!!!!!!!!!!!!!!!!
DHAP_a = 0.03 # 0.01 - 0.03 mM From Molecules to Networks 2014  #1.63 #mM  -Park2016 # 0.0764 # Lambeth # Jay Glia expand   !!!!!!!!!!!!!!!!!!!!!!!!!!

BPG13_a = 0.065 #mM Lambeth
#PG3_a,PG2_a worked: 0.52, 0.05
PG3_a = 0.375 #0.52 #0.0168 ####0.375 #-Park # 0.052 #mM Lambeth  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PG2_a = 0.00949 #0.02 #0.05 #0.00256 ####0.00949 #mM #-Park #0.02 #mM -Berndt #0.005  #mM Lambeth # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#KmPG3pgm = 0.168 # mM
#KmPG2pgm = 0.0256 # mM


PEP_a = 0.005 # 0.004-0.005 mM From Molecules to Networks 2014  # 0.0194 # Lambeth # 0.014203938186866 # Jay 181130 this value taken from Jolivet PEPg # 0.0279754 # was working with 0.0170014 # was working with 0.0279754 - Glia_170726(1).mod # was working 0.015 # glia expand in between n and g ### check it

PYR_a = 0.04 #mM 0.033-0.04 mM Arce-Molina2019 astrocyte cytosol #0.15 #0.1–0.2 mM -Lajtha 2007  #0.35 # mM Calvetti2018  # 0.0994 # Lambeth # 0.202024 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


G1P_a = 0.0589 #mM  -Lambeth #u59 
GLY_a = 5.0 #my main googledocs table # 1.12 # mM-Waitt2017 #u60
GPa_a = 0.00699071  #0.0699071 #Jay
GPb_a = 1.5*GPa_a #   0.000025 #Jay
UDPgluco_a = 0.589 #mM
GS_a  = 0.01 #0.003 # 0.0111569 # or 0.003 - both are in Jay's most recent # GSa_a0
UTP0_a = 0.23 #1.76 #0.23 # too low 5.17e-15 # 5170 pmol/10^6cells Lazarowski&Harden  # or 0.23 mM - Anderson&Wright 

cAMP_a0 = 0.0449285
PKAa_a0 = 0.0018 # from Jay 181130 = CC# 1.4339 # or 0.0018
PKAb_a0 = 0.00025 # 0.0823673 # or  0.00025

PHKa_a0 = 0.0025 # or 0.00899953
PHKb_a0 = 0.000025

PP1_a0 = 0.00025
R2CcAMP2_a0 = 1.0 # Jay 181130  # 0.3584 # 1.0 # 0.3584
R2CcAMP4_a0 = 1.0 # Jay 181130 #1.55948 # 1.0 # 1.55948


Glc_b = 4.51 # mM Calvetti2018
Lac_b = 1.24 # mM Calvetti2018
O2_b = 6.67 # mM Calvetti2018 # ?is it total or free?
#Q0 = 0.4 #mL/min
Q0 = 0.0067 #mL/s #0.4 # mL/min denoted as Q in table but seems to be q0 (baseline bloodflow) according to text and equations; used for callback
#Q0 = 0.0000067 #mL/ms #mL/ms doesn't work  #0.4 # mL/min denoted as Q in table but seems to be q0 (baseline bloodflow) according to text and equations; used for callback

PCr_a = 10.32 # mM Calvetti2018 #34.67 # Lambeth # 
Cr_a =  1.1e-3 # mM Calvetti2018 # 40-34.67 # Lambeth #



###### GLU_GLN astrocyte
#  Consequently, vesicular glutamate release from astrocytes creates localized extracellular glutamate accumulations of 1–100 μM [42] # Astrocyte glutamine synthetase: pivotal in health and disease  Rose,  Verkhratsky and Vladimir Parpura
GLUT_out0 = 25e-6 # mM # 25 nM  # Physiology of Astroglia. Verkhratsky and Nedergaard 2018 #The extracellular concentration of glutamate in resting conditions is around 25 nM (677)  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6050349/
GLUT_a0 = 0.3 #mM #0.3 mM (227) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6050349/ # Physiology of Astroglia. Verkhratsky and Nedergaard 2018   and the same value in Savtchenko-Rusakov ASTRO 2018
GLN_out0 = 0.2 #0.3 #mM # from 0.13 mM to 0.5 mM # Broer Brookes 2001 Transfer of glutamine between astrocytes and neurons #estimates of extracellular glutamine vary from 0.13 mm to 0.5 mm, the upper limit of this range being close to the mean around which plasma and CSF levels ¯uctuate (Jacobson et al. 1985; ErecinÂska and Silver 1990; Xu et al. 1998)
GLN_a0 = 2.0 #0.3 #0.2-2.0 mM Hertz2017 #0.25 #mM estimate based on "a bit lower than GLUT_a"   ##Broer Brookes 2001  Astrocytes and neurons, cultured in medium containing 2 mm glutamine, generate intracellular glutamine concentrations of 20 mm or more, when estimated on the basis of a solute-accessible water content of 4 mL per mg protein (Patel and Hunt 1985;Brookes 1992a)
GLUT_syn0 = 5e-3 # mM #Breslin2018 table2 ref52 
#Assuming a sudden arbi- trary depolarization from -􏰇80 to -􏰇40 mV ([Na􏰃]i 􏰉 5 mM, [Na􏰃]o 􏰉 140 mM, [K􏰃]i 􏰉 140 mM, [K􏰃]o 􏰉 5 mM, [Glu􏰇]i 􏰉 5 mM, and [Glu􏰇]o 􏰉 1 nM), the steady-state release rate of glutamate (1.2 molecules per EAAC1 per second) is predicted to increase 20-fold at short times after the depolarization   -- Transport direction determines the kinetics of substrate transport by the glutamate transporter EAAC1 2007 Zhou Zhang, Zhen Tao, Armanda Gameiro, Stephanie Barcelona, Simona Braams, Thomas Rauen, and Christof Grewer
#  Broer Brookes 2001 The overall concentration of glutamine in normal mammalian brain is an estimated 5±9 nmol/mg wet weight, equivalent to 6±11 mM, with little regional variation(ErecinÂska and Silver 1990)
# GLUT_n_cyt = 1−10 mM. # Physiology of Astroglia. Verkhratsky and Nedergaard 2018 #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6050349/ # the cytosolic concentration of glutamate in neurons is usually assumed to be in a range of 1−10 mM.
# # Physiology of Astroglia. Verkhratsky and Nedergaard 2018 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6050349/ Kinetics of the glutamate translocation by EAATs is relatively slow; both EAAT1 and EAAT2 transport ~30 molecules of glutamate per second (1286, 1956). The glutamate binding to the transporters (Km ~20 μM) is however much faster, and hence glutamate transporters concentrated at the perisynaptic processes act as almost instant buffers for glutamate. The higher is the density of transporters, the higher is their buffering capacity (1792). The EAAT2 in cultured hippocampal astrocytes has a remarkable lateral mobility regulated by glutamate, possibly allowing a continuous exchange of glutamate-bound and unbound transporters to maintain high buffer capacity of perisynaptic zones (1177).

################ neuron
Lac_n = 1.3
Glc_n = 1.19 # ~1.5-2.5 mM From Molecules to Networks 2014
O2_n = 0.03 # mM Calvetti2018


ATP_n = 2.18 #1.4 # 2.18 Calvetti2018
ADP_n = 0.03  #6.3e-3 # mM Calvetti2018 ### check !!

#NADH_n = 1.2e-3 # mM Calvetti2018    # Nad/NADH = 670-715  ### check !! 
#NAD_n = 0.03 # mM Calvetti2018  #NAD_n = 0.20574782239567735 # calc from Jay 181130 Rminusn 0.00625245  / 0.0303889 # was working with 0.5 # Jay Glia_170726(1).mod   
NADH_n = 0.0007 #0.075  # Nad/NADH = 670-715 
NAD_n = 0.5

PEP_n = 0.005 # 0.004-0.005 mM From Molecules to Networks 2014  # 
Pyr_n =  0.3 #0.05 #0.05-0.2mM From Molecules to Networks 2014  #  #0.38 # mM Calvetti2018

PCr_n = 10.33 # mM Calvetti2018
Cr_n = 3.0e-4 # mM Calvetti2018

G6P_n = 0.2 #0.06-0.2 mM From Molecules to Networks 2014 #0.7 #was working with 0.1 #  # both n and a values worked here # # 0.7 = a  ### 0.1 # approx from figure, so check it, Berndt 2015  
F6P_n = 0.02 # 0.01 - 0.02 mM From Molecules to Networks 2014 #0.035#-worked # 0.228 # 0.228  = a ### 0.035 # approx from figure, so check it, Berndt 2015
FBP_n = 0.1 # 0.01 - 0.1 mM From Molecules to Networks 2014  #0.035 #-approx from figure, so check it, Berndt 2015 # 1.52 #mM -Park2016 # !!!!!!!!!!!!!!!!!!   CHECK IT

GAP_n = 0.05 # 0.04 - 0.05 mM From Molecules to Networks 2014  # 0.0574386 # Jay 181130 # was working with 0.00460529 ## 0.0574386 # 0.0574386  = a ### 0.00460529 # Jay Glia_170726(1).mod
DHAP_n = 0.03 # 0.01 - 0.03 mM From Molecules to Networks 2014  # 0.05  # 0.0764 = a  ### 0.05 # approx from figure, so check it,  Berndt 2015    

Pi_n = 4.1  #1.0-worked before 18feb2020 # 4.1 # check it Jay Glia expand  
BPG13_n =  0.065 # check it Jay Glia expand   

PG3_n = 0.07 #0.375=latest astrocyte  # 0.052 =a  ### 0.07 # approx from figure, so check it,  Berndt 2015
PG2_n = 0.005 #0.00949 =latest astrocyte # Lambeth  # 0.02 # 0.005 # 0.005  = a ### 0.02 # approx from figure, so check it,  Berndt 2015

f26bp_n0 = 0.005 #0.015 #mM 0.005-0.025 mM Mulukutla2014 supp fig plot 6



#####################################################################################################################################################
# mitochondia neuron
# Berndt 2015 # fig S1 #mM and other ref, specified below
PYRmito_n0 = 0.4 # approx from figure, so check it,  Berndt 2015
CITmito_n0 = 0.35 #1.25 # approx from figure, so check it,  Berndt 2015
ISOCITmito_n0 = 0.035 #0.09 # approx # approx from figure, so check it,  Berndt 2015
AKGmito_n0 = 0.2 #0.6 # approx # approx from figure, so check it,  Berndt 2015
SUCCOAmito_n0 = 0.05 # approxm # approx from figure, so check it,  Berndt 2015
SUCmito_n0 = 0.5 ### 1.25 # Berndt 2015; or  0.5 mM from Succinate, an intermediate in metabolism, signal transduction, ROS, hypoxia, and tumorigenesis Laszlo Trette  2016
FUMmito_n0 = 0.055 #0.35 # in Berndt2015, but 1.94 in Mogilevskaya 2006
MALmito_n0 = 0.22 #2.0 # approx from figure, so check it,  Berndt 2015 ############## 0.03 by Chen ##########################################
OXAmito_n0 = 0.004 #0.0001 #mM Williamson 1967 #0.08 # OAmito # approx from figure, so check it,  Berndt 2015  # 0.1 #mM Shestov 2007

AcCoAmito_n0 = 0.07 #0.01  # The Regulatory Effects of Acetyl-CoA Distribution in the Healthy and Diseased Brain Ronowska 2018: the acetyl-CoA concentrations in neuronal mitochondrial and cytoplasmic compartments are in the range of 10 and 7 μmol/L, respectively. # 1mol/L = 1000 mM # very small and not visible on plot in Berndt
CoAmito_n0 = 0.16 #0.37  # 0.37 = a ### 0.001 - lead to domain error # Rock 2000 Pantothenate Kinase Regulation of the Intracellular Concentration of Coenzyme A  + very small and not visible on plot in Berndt 

NADmito_n0 = 0.14 #0.143622359  # 0.146538 =a ###  0.143622359 #calc from Gibson & Blas 1975 NAD+/NADH = 1.163 
NADHmito_n0 = 0.07 #0.123493 # 0.126012 ### 0.123493 # Jay Newest
#
#
CaMito_n0 = 0.001 # from Mogilevskaya 2006 # 5e-5 # Calcium  = 5.10258e-5 in  Jay, but check which part is mito....  see my pencil notes on printed Calvetti  p 242 about Ca in ddifferent organelles
# no AMPmito_n
ADPmito_n0 = 0.86 #8.23 #CHECK it !!!! #rat liver mito  Modeling of ATP–ADP steady‐state exchange rate mediated by the adenine nucleotide translocase in isolated mitochondria FEBS 2009 Eugeniy Metelkin  Oleg Demin  Zsuzsanna Kovács  Christos Chinopoulos
ATPmito_n0  = 3.36 #0.51 # CHECK it !!!!#rat liver mito  Modeling of ATP–ADP steady‐state exchange rate mediated by the adenine nucleotide translocase in isolated mitochondria FEBS 2009 Eugeniy Metelkin  Oleg Demin  Zsuzsanna Kovács  Christos Chinopoulos
Pimito_n0 = 2.7 #5.0 # from Mogilevskaya 2006 ### check it !!!!!

#Qmito+QH2mito= 0.136275 scaled from IvanChang commented defaulr to Jay's NADH
#Qmito+QH2mito= 0.27255 from IvanChang not commented
Qmito_n0 = 0.1 #0.07 # 0.630012*11.2355/(11.2355+88.7645)=0.07078499826 # 0.03062 #0.27255*11.2355/(11.2355+88.7645) = 0.03062235525 inferred from IvanChang steady state # 0.129 # 19.0 in Mogilevskaya 2006 # scaled from IvanChang 19*0.136275/20 # check it
QH2mito_n0 = 0.1 #0.559 #0.630012*88.7645/(11.2355+88.7645)=0.55922700174 #0.2419 # 0.27255*88.7645/(11.2355+88.7645) = 0.24192764475 inferred from IvanChang steady state # 0.0068 #1.0 in Mogilevskaya 2006 scaled from IvanChang 0.136275/20  # check it
#CytCredmito+CytCoxmito=30 in IvanChang(default,but commented) -> 0.03
# ratio from Cytochrome c is rapidly reduced in the cytosol after mitochondrial outer membrane permeabilization Ripple 2010: 62% mito cyt C oxidized
# CytCredmito+CytCoxmito=50 not commented -> 0.136275 scaled to Jay
CytCredmito_n0 = 0.01 #Zhang2018 #0.10489 # 0.315006*(16.6494/50.0) #0.045 # 0.136275*(16.6494/50.0) = 0.0453779397 inferred from IvanChang steady state  #0.0114 # scaled from IvanChang to Jay's NADH: 0.126012+0.146538=0.27255; IvanChang: NADH+NAD = 100 or 300 (default,but commented) Q+QH2 = 100 or 150 (default,but commented) CytCredmito+CytCoxmito = 50 or 30 (default,but commented)
CytCoxmito_n0 = 0.02  #Zhang2018  #0.21 #0.315006*(33.3506/50.0) = 0.21011278207200001 #0.09# #0.136275*(33.3506/50.0)= 0.09089706030000001 inferred from IvanChang steady state #0.0186

###
ASPmito_n0 = 2.0 #Shestov2007 #2.6 From Molecules to Networks2014  #1.5 ### Chen Abs Quant Big mmc1 # check and set neuronal!!!! 
GLUmito_n0 = 2.5 #5.3 #0.057 # by Chen # check and set neuronal!!!! 

### MAS cyto-mito
MAL_n0 = 0.45 # 2.0 #cytosol # 0.6 by Chen # 5.0 # check and set neuronal!!!! just assumption  from # Nonactivating behavior is observed at concentrations between 0.02 and 0.15 mM L-malate and activating behavior is observed between 0.15 and 0.5 mM L-malate.(Malate dehydrogenase. Kinetic studies of substrate activation of supernatant enzyme by L-malate. Mueggler PA, Wolfe RG.)
OXA_n0 = 0.1 #cytosol #0.005 # check and set neuronal!!!! just assumption from heart data now # Berndt 2015 re Indiveri C, Dierks T, Kramer R, Palmieri F. Reaction-Mechanism of the Reconstituted Oxoglutarate Carrier from Bovine Heart-Mitochondria 
ASP_n0 = 2.0 #6.0 #cytosol  ### Chen Abs Quant Big mmc1  # 5.0 # 2.0 # check and set neuronal!!!! 
AKG_n0 = 0.2 #1.2 #cytosol # by Chen  #0.265 Pritchard 1995 # check and set neuronal!!!! 


### PPP
GL6P_n0 =  5.0e-06 #mM from Sabate 1995 # 7.62e-06  mM Nakayama 2005 rbc 
GO6P_n0 = 0.0097 #mM calc from Kauffman1969 #2.72  Nakayama 2005 rbc #### from Sabate 1995: 0.018 mM
#GSH_0 = # glutathione Nakayama 2005 rbc 
#GSH_0 = u[158] neuron 0.91  # Duarte, Gruetter 2013. 10.1111/jnc.12333
#GSSG_0 = #glutathione disulfide Nakayama 2005 rbc
#PRPP_0 = 6.91e-05 # 5-Phosphoribosyl 1-phosphate Nakayama 2005 rbc
# NAD_0 = 0.0887 # Nakayama 2005 rbc
# NADH_0 = 3.13e-04 # Nakayama 2005 rbc
NADP_n0 = 1.0e-03 #mM from Sabate 1995  #8.06e-05 # Nakayama 2005 rbc  
NADPH_n0 = 2.0e-4 # mM from Sabate 1995  #6.58e-02 mM Nakayama 2005 rbc 
# NaSodium = 0.227 # Nakayama 2005 rbc
# KPotassium = 0.0126 or 0.0135 (lit) Nakayama 2005 rbc
RU5P_n0 = 0.0072 #mM calc from Kauffman1969 #1.48e-04 Nakayama 2005 rbc  #### from Sabate 1995: 0.012 mM
X5P_n0 = 0.0097 #mM calc from Kauffman1969 #4.3e-04  # Nakayama 2005 rbc #### from Sabate 1995: 0.018 mM
R5P_n0 =  0.009 #mM from Sabate 1995 #2.81e-04 # Nakayama 2005 rbc #### +S7P calc from Kauffman1969: 0.0643 mM +S7P 
S7P_n0 =  0.0643 #mM calc from Kauffman1969 #0.0749 # Nakayama 2005 rbc #### +R5P calc from Kauffman1969: 0.0643 mM #### from Sabate 1995:  0.068 mM
E4P_n0 = 0.002 # mM calc from Kauffman1969 #1.17 Nakayama 2005 rbc ####  calc from Kauffman1969: <0.002 mM #### from Sabate 1995: 0.004 mM
CO2_n0 = 0.001 # mM from Sabate 1995
 

SDHmito_n0 = 0.05 # Mogilevskaya 2006  # levels of sdh expression in astrocytes and neurons are almost the same by Sharma's data

###########################################################################################


### mito astrocyte
PYRmito_a = 0.021 #0.025 #0.0235 #mM  Arce-Molina2019 astrocyte mito #u120 !!!!!!!!!!!!
CITmito_a = 0.35 #0.25-0.35 From molecules to networks 2014; 0.2-0.4 Ronowska2018 # # u121
ISOCITmito_a = 0.035 #Frezza2017 Cit/Isocit =10; 0.02 From molecules to networks 2014 # u122
AKGmito_a = 0.2 #From molecules to networks 2014 # u123
SUCCOAmito_a = 0.05 #0.0068 #0.05 #0.0068 #Park2016 # u124
SUCmito_a = 0.45 # 0.7 #0.5 #0.45-0.7 From molecules to networks 2014 #u125
FUMmito_a = 0.055 #in between of Fink20018 and From molecules to networks 2014 #u126
MALmito_a = 0.22 #0.45 #in between of googledoc sources and From molecules to networks 2014  # u127
OXAmito_a = 0.004 #From molecules to networks 2014  #u128
AcCoAmito_a = 0.07 #Nazaret2008 #0.0045 #From molecules to networks 2014; Ronowska2018 #u129
CoAmito_a = 0.16 # in between of lit values #0.16 #Mogilevskaya #0.003  #From molecules to networks 2014 #u130

NADmito_a = 0.14 #Mito  NAD/NADH ~ 2.0-2.25 (Berndt 2015, Dienel review) #u131
NADHmito_a = 0.07 #Mito  NAD/NADH ~ 2.0-2.25 (Berndt 2015, Dienel review) #u132
CaMito_a = 0.001 # Mogilevskaya 2006 #u133

ADPmito_a = 0.86 #3.36/3.9=0.86 #1.28 #0.032 #u134
ATPmito_a = 3.36 # 1.4*2.4=3.36 #5.0  #0.51 #u135

Pimito_a = 2.7 #2.5 #5.0 #2.7 #2.0 #From molecules to networks 2014 #  #u136


############## CHECK Q & QH2 CONCENTRATIONS RATIO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Qmito+QH2mito= 0.136275 scaled from IvanChang commented defaulr to Jay's NADH
#Qmito+QH2mito= 0.27255 from IvanChang not commented
Qmito_a = 0.1 #0.129 #0.07 #u137 # 0.630012*11.2355/(11.2355+88.7645)=0.07078499826 # 0.03062 #0.27255*11.2355/(11.2355+88.7645) = 0.03062235525 inferred from IvanChang steady state # 0.129 # 19.0 in Mogilevskaya 2006 # scaled from IvanChang 19*0.136275/20 # check it             
QH2mito_a = 0.1 #0.0068 # 0.559  #u138#0.630012*88.7645/(11.2355+88.7645)=0.55922700174 #0.2419 # 0.27255*88.7645/(11.2355+88.7645) = 0.24192764475 inferred from IvanChang steady state # 0.0068 #1.0 in Mogilevskaya 2006 scaled from IvanChang 0.136275/20  # check it
#CytCredmito+CytCoxmito=30 in IvanChang(default,but commented) -> 0.03
# ratio from Cytochrome c is rapidly reduced in the cytosol after mitochondrial outer membrane permeabilization Ripple 2010: 62% mito cyt C oxidized
# CytCredmito+CytCoxmito=50 not commented -> 0.136275 scaled to Jay
CytCredmito_a =  0.01 #Zhang2018 # 0.10489  #u139 # 0.315006*(16.6494/50.0) #0.045 # 0.136275*(16.6494/50.0) = 0.0453779397 inferred from IvanChang steady state  #0.0114 # scaled from IvanChang to Jay's NADH: 0.126012+0.146538=0.27255; IvanChang: NADH+NAD = 100 or 300 (default,but commented) Q+QH2 = 100 or 150 (default,but commented) CytCredmito+CytCoxmito = 50 or 30 (default,but commented)
CytCoxmito_a =  0.02  #Zhang2018 # 0.21  #u140 #0.315006*(33.3506/50.0) = 0.21011278207200001 #0.09# #0.136275*(33.3506/50.0)= 0.09089706030000001 inferred from IvanChang steady state #0.0186


ASPmito_a = 2.0 #u141
GLUmito_a = 5.3 #mM Nazaret2008 #0.057 # Chen # check cell type specificity #u142 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MAL_a = 0.45 #From Molecules to Networks2014   #u143   ### check subcell specificity
OXA_a = 0.1 #cytosol #0.01 #mM Williamson 1967  #u144    ### check subcell specificity
ASP_a = 2.0 #u145    ### check subcell specificity
AKG_a = 0.2 #From molecules to networks 2014 ##u146      ### check subcell specificity


#CO2_a = 0.001 # mM from Sabate 1995  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
GL6P_a =   5.0e-06 #mM from Sabate 1995 # 7.62e-06  mM Nakayama 2005 rbc  #u147
GO6P_a = 0.0097 #mM calc from Kauffman1969 #2.72  Nakayama 2005 rbc #### from Sabate 1995: 0.018 mM #u148

NADP_a = 1.0e-03 #mM from Sabate 1995  #8.06e-05 # Nakayama 2005 rbc #u149
NADPH_a = 2.0e-4 # mM from Sabate 1995  #6.58e-02 mM Nakayama 2005 rbc #u150

RU5P_a = 0.0072 #mM calc from Kauffman1969 #1.48e-04 Nakayama 2005 rbc  #### from Sabate 1995: 0.012 mM #u151
X5P_a = 0.0097 #mM calc from Kauffman1969 #4.3e-04  # Nakayama 2005 rbc #### from Sabate 1995: 0.018 mM#u152
R5P_a =  0.009 #mM from Sabate 1995 #2.81e-04 # Nakayama 2005 rbc #### +S7P calc from Kauffman1969: 0.0643 mM +S7P  #u153
S7P_a = 0.0643 #mM calc from Kauffman1969 #0.0749 # Nakayama 2005 rbc #### +R5P calc from Kauffman1969: 0.0643 mM #### from Sabate 1995:  0.068 mM #u154
E4P_a =  0.002 # mM calc from Kauffman1969 #1.17 Nakayama 2005 rbc ####  calc from Kauffman1969: <0.002 mM #### from Sabate 1995: 0.004 mM#u155
# NaSodium = 0.227 # Nakayama 2005 rbc
# KPotassium = 0.0126 or 0.0135 (lit) Nakayama 2005 rbc


# Glutathione Astrocyte
GSH_a = 2.0 #2.6 #McBean2017: astrocytes contain one of the highest cytosolic GSH conc  #2.0 #0.57  # ~2.6 mM From molecules to networks 2014 #reduced
GSSG_a = 0.003*GSH_a #0.43*GSH_a #Raps1989 #GSH_a/0.9 #GSH/GSSG = 90% McBean2017 # glutathione disulfide
# Glutathione Neuron
GSH_n = 2.0 #2.0 #0.57 # ~2.6 mM From molecules to networks 2014 #reduced
GSSG_n = 0.003*GSH_n #0.3% of totGSH  # GSH_n/0.9 #GSH/GSSG = 90% McBean2017 # glutathione disulfide

MitoMembrPotent_a = 138.0 #138.0 #120.0 #138.0 #200.0 #138.0 # 120-160 mV negative inside #check which sign it should be !!!!!!!!!!!!!!!!!!!!!!!!! #u160
MitoMembrPotent_n = 138.0

GLU_n = 6.0 #10.0 # from MAS # 11.6 - From molecules to networks 2014 #check 
GABA_n = 1.0 #mM check! Diff for diff n types
AMP_n = 0.01 #2e-5 #0.03 #-Mulukutla2015 #2e-5 # Lambeth # 0.01 # 2e-5 # 1e-5 # from 1e-5 to 0.05

GLN_n = 5.0 #0.4 #5.0#with active GLS is depletes #0.4 # mM Shestov 2007 # for now no separation between cyto and mito GLN

u0 = [VNeu0,m0,h0,n0,Conc_Cl_out,Conc_Cl_in,Na0in,K0out,Glc_b,Lac_b,O2_b,Q0,   
    Glc_ecs,Lac_ecs,
    O2_ecs,O2_n,O2_a,
    Glc_n,Glc_a,Lac_n,Lac_a,Pyr_n,PYR_a,PCr_n,PCr_a,Cr_n,Cr_a,
    ATP_n,ATP_a,ADP_n,ADP_a,
    NADH_n,NADH_a,NAD_n,NAD_a,
    ksi0,ksi0, 
    G6P_n,F6P_n,G6P_a,F6P_a,FBP_n,FBP_a,AMP_a,GAP_n,DHAP_n,GAP_a,DHAP_a,
    Pi_n,BPG13_n,Pi_a,BPG13_a,PG3_n,PG3_a,PG2_n,PG2_a,PEP_n,PEP_a,G1P_a,GLY_a,UDPgluco_a,UTP0_a,
    PP1_a0,GS_a,GPa_a,GPb_a,cAMP_a0,PKAa_a0,PKAb_a0,PHKa_a0,R2CcAMP2_a0,R2CcAMP4_a0,K_a0,Na_a0,Va0,Ca_a0,GLUT_out0,GLUT_a0,GLN_out0,GLN_a0,GLUT_syn0,f26bp_a0,f26bp_n0,
    PYRmito_n0,CITmito_n0,ISOCITmito_n0,AKGmito_n0,SUCCOAmito_n0,SUCmito_n0,FUMmito_n0,MALmito_n0,OXAmito_n0,AcCoAmito_n0,CoAmito_n0,
    NADmito_n0,NADHmito_n0,CaMito_n0, ADPmito_n0,ATPmito_n0, Pimito_n0, Qmito_n0,QH2mito_n0,CytCredmito_n0,CytCoxmito_n0,ASPmito_n0,GLUmito_n0,
    MAL_n0,OXA_n0,ASP_n0,AKG_n0,GL6P_n0,GO6P_n0,NADP_n0,NADPH_n0,RU5P_n0,X5P_n0,R5P_n0,S7P_n0,E4P_n0,
    PYRmito_a,
    CITmito_a,ISOCITmito_a,AKGmito_a,SUCCOAmito_a,SUCmito_a,FUMmito_a,MALmito_a,OXAmito_a,AcCoAmito_a,CoAmito_a,NADmito_a,NADHmito_a,CaMito_a,ADPmito_a,ATPmito_a,Pimito_a,Qmito_a,QH2mito_a,CytCredmito_a,
    CytCoxmito_a,ASPmito_a,GLUmito_a,MAL_a,OXA_a,ASP_a,AKG_a,GL6P_a,GO6P_a,NADP_a,NADPH_a,RU5P_a,X5P_a,R5P_a,S7P_a,E4P_a,GSH_a,GSSG_a,GSH_n,GSSG_n,MitoMembrPotent_a,MitoMembrPotent_n,GLU_n,GABA_n,AMP_n,
    GLN_n];
    
#du = similar(u0);
pAKTPFK2=0.17

#################################################
# STEPS Model Build
#################################################

def gen_model():
    mdl = smodel.Model()
    vsys = smodel.Volsys(Volsys0.name, mdl)
    na_spec = smodel.Spec(Na.name, mdl)
    diff = smodel.Diff(Na.diffname, vsys, na_spec)
    diff.setDcst(Na.diffcst)
    k_spec = smodel.Spec(K.name, mdl)
    diff = smodel.Diff(K.diffname, vsys, k_spec)
    diff.setDcst(K.diffcst)
   # ca_spec = smodel.Spec(Ca.name, mdl)
   # diff = smodel.Diff(Ca.diffname, vsys, ca_spec)
   # diff.setDcst(Ca.diffcst)
   # cl_spec = smodel.Spec(Cl.name, mdl)
   # diff = smodel.Diff(Cl.diffname, vsys, cl_spec)
   # diff.setDcst(Cl.diffcst)
    return mdl

def gen_geom():
    mesh = meshio.loadMesh('cubeBig')[0] # cubeBig units are um
    ntets = mesh.countTets()
    comp = stetmesh.TmComp(Geom.compname, mesh, range(ntets))
    comp.addVolsys(Volsys0.name)
    return mesh

def gen_geom2():
    mesh = meshio.loadMesh('cube')[0] #cude units are m
    ntets = mesh.countTets()
    comp = stetmesh.TmComp(Geom.compname, mesh, range(ntets))
    comp.addVolsys(Volsys0.name)
    return mesh, ntets

def init_solver(model, geom):
    rng = srng.create('mt19937', 512)
    rng.initialize(2903)
    if STEPS_USE_MPI:
        import steps.mpi.solver as psolver
        tet_hosts = gd.linearPartition(geom, [1, 1, steps.mpi.nhosts])
        # if steps.mpi.rank == 0:
        #     gd.validatePartition(geom, tet_hosts)
        #     gd.printPartitionStat(tet_hosts)
        return psolver.TetOpSplit(model, geom, rng, psolver.EF_NONE, tet_hosts)
    else:
        import steps.solver as solvmod
        return solvmod.Tetexact(model, geom, rng)




######################################################################## the following get_sec_mapping function was updated by Dan in his dualrun6.py which he gave me on 02dec2019
# returns a list of tet mappings for the neurons
# each segment mapping is a list of (tetnum, fraction of segment)
def get_sec_mapping(ndamus, mesh):
    mapdir="./tetMapping"
    if not os.path.isdir(mapdir):
        os.mkdir(mapdir)
        print("tetMapping")
    neurSecmap = []
    secSegFractMap = []
    segment_3d_contribs = defaultdict(list)

    for c in ndamus.cells:        
#        print("Cell gid is ",c.CCell.gid)
        filename=mapdir+"/Neuron."+str(int(c.CCell.gid))+".pickle"
        #print(filename)
        secs = [sec for sec in c.CCell.all if (hasattr(sec, Na.current_var)&(hasattr(sec, K.current_var))&(hasattr(sec, ATP.atpi_var))&(hasattr(sec, ADP.adpi_var)))]
        #secs = [sec for sec in c.CCell.all if (hasattr(sec, Na.current_var)&(hasattr(sec, K.current_var))&hasattr(sec, Ca.current_var))] #modified from the line above by Polina on 3dec2019
        if os.path.isfile(filename):
            if os.path.getsize(filename) > 0: 
                #   print('file found')
                file_pi2 = open(filename, 'rb') 
                picklesecmap = pickle.load(file_pi2)
                secmap=[]
                for elem in picklesecmap:
                    newelem=(secs[elem[0]],elem[1])
                    secmap.append(newelem)
                file_pi2.close()
            else:
                print("Zero-sized file for cell ",c.CCell.gid)
                
                secmap=[]
            neurSecmap.append(secmap) 
            continue

        print("Creating file")
        # Our returned struct is #neurons long list of
        # [(sec1, nparray([pt1, pt2, ...])), (sec2, npa...]
        picklesecmap=[]
        secmap = []
        for secID,sec in enumerate(secs):
            npts = sec.n3d() # seems like it's a  number of 3d points in section
            
            if not npts:
                logging.warning("Sec %s doesnt have 3d points", sec)
                continue
            fractmap = []
            # Get points and distances to start of the section
            segs3d_lens = np.empty(npts-1)
            p0 = np.array([float(sec.x3d(0)), float(sec.y3d(0)), float(sec.z3d(0))]) #p0 = np.array([sec.x3d(0)*1e-6, sec.y3d(0)*1e-6, sec.z3d(0)*1e-6]) # *1e-6 in leite new
            total_length = sec.L #leite
            nrn_segm_len = total_length / sec.nseg #leite
            distance = .0 #leite

            for i in range(1, npts):
                # Get, according to the previous distance, the nrn segment we are in
                cur_nrn_seg = math.floor(distance / total_length)
                # Get the current point and get also the nrn segment
                p1 = np.array([float(sec.x3d(i)), float(sec.y3d(i)), float(sec.z3d(i))]) # p1 = np.array([sec.x3d(i)*1e-6, sec.y3d(i)*1e-6, sec.z3d(i)*1e-6]) 
                #map 3d points to segs
                distance += np.linalg.norm(p1 - p0)
                p1_nrn_seg = math.floor(distance / total_length)
                
                # If we are in the same segment, the contribution is 100%
                if cur_nrn_seg == p1_nrn_seg:
                    segment_3d_contribs[cur_nrn_seg].append([i, 1, p1])
                else:
                # Otherwise the 3d segment crossed two nrn segments and we have to append contributions to both
                    rel_length_next = (distance / total_length) - p1_nrn_seg
                    segment_3d_contribs[cur_nrn_seg].append([i, 1 - rel_length_next,p1])
                    segment_3d_contribs[cur_nrn_seg + 1].append([i, rel_length_next,p1])
                #p0 = p1

               #### sections
                pts = np.array([p0, p1])
                #print("pts is ",pts)
                out1 = mesh.intersect(pts)
                #print("p0 is this value",p0,"!") 
                #print("out1 is", out1)
 
                if len(out1) > 0:
                    #print(out1[0],"\n")
                    if str(out1) == "[[]]":
                       print("ThisNeedsToBeChecked!!!!!!!!!!")
                       continue 
                    if len(out1[0][0]) > 1:
                        if out1[0][0][1] > 1:
                            print("PROBLEM WITH FRACTION DETECTED")
                            print("CELL GID: ",c.CCell.gid)
                           # print(out1)
                            #out1[0][0][1] = 0 #temporary patch till STEPS get fixed
                            #exit() #should be restored
                fractmap.append(out1)

                p0 = p1
            picklesecmap.append((secID, fractmap))
            secmap.append((sec, fractmap))
            secSegFractMap.append((sec, segment_3d_contribs))

        #print("secmap: ",secmap)
        
        filehandler = open(filename, 'wb') 
        pickle.dump(picklesecmap, filehandler)

        neurSecmap.append(secmap)
        #secSegFractMap.append((sec, segment_3d_contribs))
        #print("secSegFractMap ",secSegFractMap)
    return(neurSecmap, secSegFractMap)


##############################################
# Runtime
##############################################

########################################################################
### old part, with fleite
#def get_currents_neuron(comps_coords):
#    return np.fromiter((sec.ina for sec in comps_coords),'f8')
########################################################################
                  
### ndam time
def timesteps(end: float, step: float):
    return ((i+1) * step for i in range(int(end/step)))


########################################################################
#run all together: outer loop - metabolism; intermediate - ndam; inner - steps
def main():
    #u0=def_ini_val_metabolism()
    #metabolism = gen_metabolism_model() # define Julia metabolism model
    
    ndamus = Neurodamus("BlueConfig", enable_reports=True, logging_level=None) #enable_reports=False True

    log_stage("Initializing steps model and geom")
    model = gen_model()
    tmgeom = gen_geom()

    logging.info("Computing segments per tet")
    neurSecmap,secSegMap = get_sec_mapping(ndamus, tmgeom) # [0] is added Polina at 13 dec 2019, get_sec_mapping modified to have return(neurSecmap, secSegFractMap)
    tmgeom, ntets = gen_geom2()
    #print("secSegMap len: ",len(secSegMap),"secSegMap 0", [x[0] for x in secSegMap], "s1 ",  [x[1] for x in secSegMap] )
    
   # for sec_item,seg_item in [(x[0],x[1]) for x in secSegMap]: #zip([x[0] for x in secSegMap],[x[1] for x in secSegMap]): #check it
   #     print("sec ",sec_item)       
   #     for seg_item_key, seg_item_value in seg_item.items():
   #         #print("key ",seg_item_key)
   #         for pts_coord in seg_item_value:
   #             #print("coord ",pts_coord[2])
   #             pts_coord_meters = [x*1e6 for x in pts_coord[2]] #this is because of Dan's Steps units stuff with cube and cubeBig
   #             tet_id = tmgeom.findTetByPoint(pts_coord_meters)
   #             print("tet_id is ",tet_id)

    #pt_coords_cube = [x*1e6 for x in secSegMap[]]
    #print("pt_coords_cube ",pt_coords_cube)
    #tmgeom.findTetByPoint(pt_coords_cube)

    #print("ntets: ",ntets)
    logging.info("Initializing simulations")
    ndamus.sim_init() ### IT WAS HERE and now moved down just to test
    steps_sim = init_solver(model, tmgeom)
    steps_sim.reset()
    # there are 0.001 M/mM
    steps_sim.setCompConc(Geom.compname, Na.name, Na.conc_0 * CONC_FACTOR)
    steps_sim.setCompConc(Geom.compname, K.name,  K.conc_0 * CONC_FACTOR)
    #steps_sim.setCompConc(Geom.compname, Ca.name, 0.001 * Ca.conc_0 * CONC_FACTOR)
    #steps_sim.setCompConc(Geom.compname, Cl.name, 0.001 * Cl.conc_0 * CONC_FACTOR)
    
    tetVol: List[float] = [tmgeom.getTetVol(x) for x in range(ntets)]

    log_stage("===============================================")
    log_stage("Running all Julia, Neuron, STEPS simultaneously")
    
    rank: int = comm.Get_rank()
    #if rank == 0 and REPORT_FLAG:
    #    f = open("tetConcs.txt", "w+")
    
    # !!!!!!! check if def fract is ok to be out of main or should it be in main
    def fract(neurSecmap):
        # compute the currents arising from each segment into each of the tets
        tet_currents_na = np.zeros((ntets,), dtype=float)
        #tet_enas = np.zeros((ntets,), dtype=float) #added by Polina, 25nov2019
        tet_voltage = np.zeros((ntets,), dtype=float) #added by Polina, 28nov2019
        tet_nais = np.zeros((ntets,), dtype=float) #added by Polina, 25nov2019
        
        tet_currents_k = np.zeros((ntets,), dtype=float) #10jan2020
        #tet_currents_ca = np.zeros((ntets,), dtype=float)
        #tet_currents_cl = np.zeros((ntets,), dtype=float)
        
        tet_atpi = np.zeros((ntets,), dtype=float) #added by Polina, 26feb2020
        tet_adpi = np.zeros((ntets,), dtype=float) #added by Polina, 26feb2020
        
        
        for secmap in neurSecmap:
            # for sec, fractlist in ProgressBar.iter(secmap):
            for sec, fractlist in secmap:
                for seg, fractitem in zip(sec.allseg(), fractlist):
                    if fractitem:
                        tet: int
                        fract: float
                        for tet, fract in fractitem[0]:
                            # there are 1e8 um2 in a cm2, final output in mA
                            tet_currents_na[tet] += seg.ina * seg.area() / (1e8) * fract
                            
                            #tet_enas[tet] += seg.ena #* seg.area() / (1e8) * fract #added by Polina, 28nov2019, check it!
                            tet_voltage[tet] += seg.v   #added by Polina, 28nov2019, check it!
                            #print("tet in loop ",tet )
                            tet_nais[tet] += seg.nai #* seg.area() / (1e8) * fract #added by Polina, 28nov2019, check it!
                            
                            tet_currents_k[tet] += seg.ik * seg.area() / (1e8) * fract #10jan2020
                            #tet_currents_ca[tet] += seg.ica * seg.area() / (1e8) * fract
                            #tet_currents_cl[tet] += seg.icl * seg.area() / (1e8) * fract
                            
                            tet_atpi[tet] += seg.atpi #* seg.area() / (1e8) #* fract #26feb2020  ### THIS SHOULD BE CHANGED, it's just for test
                            tet_adpi[tet] += seg.adpi #* seg.area() / (1e8) #* fract #26feb2020  ### THIS SHOULD BE CHANGED, it's just for test
                            
                            
        #return(tet_currents_na,tet_enas,tet_voltage,tet_nais,tet_currents_k,tet_currents_ca) # or should I return list?
        return(tet_currents_na,tet_voltage,tet_nais,tet_currents_k,tet_atpi,tet_adpi) 
   
    um = dict() #np.zeros((11,ntets))#, dtype=float)        
    
   
    for idxm in range(2):  ### loop for metabolism time in seconds #!!!  # this is inefficient, I need to reduce two loops (tets and timesteps) somehow
        print("idxm: ",idxm)
        #ndamus.sim_init() ### TMP TO TEST!!!!
        #### or this can be also put as callback event for julia-metabolism which is triggereg by t condition for every coarse-grained t=1sec
        ## !!! the following loop needs to be modified to keep record for the entire coarse-grained time range 
        
        for t in ProgressBar(int(SIM_END / DT))(timesteps(SIM_END, DT)): # this time is in ms: SIM_END = 0.1 #ms, DT = 0.025  #ms i.e. = 25 usec which is timstep of ndam 
            with timer('steps_cum'):
                steps_sim.run((idxm*SIM_END +  t)/1000)  # ms to sec

           #test if feedback works
            for c in ndamus.cells:
                secs_test = [sec for sec in c.CCell.all if (hasattr(sec, Na.current_var)&(hasattr(sec, K.current_var))&(hasattr(sec, ATP.atpi_var))&(hasattr(sec, ADP.adpi_var))   )]   #10jan2020,26feb2020
                #secs_test = [sec for sec in c.CCell.all if hasattr(sec, Na.current_var)] #it was this line prior to 10jan2020
                for sec_t in secs_test:
                    seg_ts = sec_t.allseg()
                    for seg_t in seg_ts:
                        #print("test nai feedback ",seg_t.nai)
                        with open("/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/test_output_segt_nai_13mar2020.txt", "a") as test_outputfile:
                            test_outputfile.write(str(seg_t.nai))
                            test_outputfile.write("\n")
                        with open("/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/test_output_segt_atpi_13mar2020.txt", "a") as test_outputfile:
                            test_outputfile.write(str(seg_t.atpi))
                            test_outputfile.write("\n")
                        with open("/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/test_output_segt_adpi_13mar2020.txt", "a") as test_outputfile:
                            test_outputfile.write(str(seg_t.adpi))
                            test_outputfile.write("\n")

            with  timer('neuron_cum'):
                ndamus.solve(idxm*SIM_END + t)
    ##############next line was modified from the original Polina 28nov2019 because return(tet_currents_na, tet_enas, tet_voltage, tet_nais)
            with timer("comm_allred"):
                #print("fract neurosecmap ",fract(neurSecmap)) 
                collect_var = np.array(fract(neurSecmap))
                #(tet_currents_na_tot,tet_enas_tot,tet_voltage_tot,tet_nais_tot,tet_currents_k_tot,tet_currents_ca_tot) = comm.allreduce(collect_var, op=MPI.SUM) 
                (tet_currents_na_tot,tet_voltage_tot,tet_nais_tot,tet_currents_k_tot,tet_atpi_tot,tet_adpi_tot) = comm.allreduce(collect_var, op=MPI.SUM) 
                #tet_enas_mean = tet_enas_tot / steps.mpi.nhosts
                tet_voltage_mean = tet_voltage_tot / steps.mpi.nhosts
                tet_nais_mean = tet_nais_tot / steps.mpi.nhosts
                #tet_atp_mean = tet_atpi_tot / steps.mpi.nhosts
                #tet_adp_mean = tet_adpi_tot / steps.mpi.nhosts
                 
                 
                 

            #with open("/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/output_tet_voltage_mean.txt", "ab") as tet_voltage_mean_outputfile:
            #    pickle.dump(tet_voltage_mean, tet_voltage_mean_outputfile)
            
            #with open("/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/output_tet_nais_mean.txt", "ab") as tet_nais_mean_outputfile:
            #    pickle.dump(tet_nais_mean, tet_nais_mean_outputfile)
            
            #with open("/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/output_tet_currents_na_tot.txt", "ab") as tet_currents_na_tot_outputfile:
            #    pickle.dump(tet_currents_na_tot, tet_currents_na_tot_outputfile)
            
            #np.savetxt('out_tet_nais_mean.txt', tet_nais_mean, delimiter="\t")
            with timer('float_assign'):
            # update the tet concentrations according to the currents
                tidx: int
                curr: float
                tetConcs_na: List[float] = steps_sim.getBatchTetConcs(list(range(ntets)), Na.name)
                tetConcs_k: List[float] = steps_sim.getBatchTetConcs(list(range(ntets)), K.name)
            #tetConcs_ca: List[float] = steps_sim.getBatchTetConcs(list(range(ntets)), Ca.name)
            #tetConcs_cl: List[float] = steps_sim.getBatchTetConcs(list(range(ntets)), Cl.name)
            
            #tet_currents_na_tot_tidx = np.zeros((ntets,), dtype=float) #list() #dict()
            #tet_voltage_mean_tidx  = np.zeros((ntets,), dtype=float) # = list() # = dict() #: List[float] = []
            #tet_nais_mean_tidx = np.zeros((ntets,), dtype=float) #  = list() #= dict() #: List[float] = []
            
            
            with timer('curr2conc'):
                for tidx, curr in enumerate(tet_currents_na_tot):
                # 0.001A/mA 6.24e18 particles/coulomb 1000L/m3
                # the following concentration will be in M
                    tetConcs_na[tidx] += CA * curr * tetVol[tidx]
            #with open("tmp.txt","wb") as outf:
            #    pickle.dump(tetConcs_na, outf)
            for tidx, curr in enumerate(tet_currents_k_tot):
                # 0.001A/mA 6.24e18 particles/coulomb 1000L/m3
                # the following concentration will be in M
                tetConcs_k[tidx] += curr * CA * tetVol[tidx]
            
            #for tidx, curr in enumerate(tet_currents_ca_tot):
            #    # 0.001A/mA 6.24e18 particles/coulomb 1000L/m3
            #    # the following concentration will be in M
            #    tetConcs_ca[tidx] += curr * CA * tetVol[tidx]
            
            #for tidx, curr in enumerate(tet_currents_cl_tot):
            #    # 0.001A/mA 6.24e18 particles/coulomb 1000L/m3
            #    # the following concentration will be in M
            #    tetConcs_cl[tidx] += curr * CA * tetVol[tidx]
            with timer('setBatchTetConcs'):
                with open("test_output_tetConcs_na.txt", "a") as tetConcs_na_outfile:
                    tetConcs_na_outfile.write("\n".join(("%e" % x for x in tetConcs_na)))
                steps_sim.setBatchTetConcs(list(range(ntets)), Na.name, tetConcs_na)
                steps_sim.setBatchTetConcs(list(range(ntets)), K.name, tetConcs_k)
            #steps_sim.setBatchTetConcs(list(range(ntets)), Ca.name, tetConcs_ca)
            #steps_sim.setBatchTetConcs(list(range(ntets)), Cl.name, tetConcs_cl)
            
            #if rank == 0 and REPORT_FLAG:
            #    f.write(" ".join(("%e" % x for x in tetConcs_na)))
                
        #if rank == 0 and REPORT_FLAG:
        #    f.close()
        
        
        ### try this hack to fix segfault 03dec2019
            #for tidx, curr in enumerate(tet_currents_na_tot):
            #    tet_currents_na_tot_tidx[tidx] += curr
            #for tidx, curr in enumerate(tet_voltage_mean):
            #    tet_voltage_mean_tidx[tidx] += curr
            #for tidx, curr in enumerate(tet_nais_mean):
            #    tet_nais_mean_tidx[tidx] += curr
            
        ##################### METABOLISM RUN NOW!
        
        log_stage("===============================================")
        log_stage("Run one step of metabolism for every tet")
        
        #Non-zero voltage only in 238 and 239 tets with given mesh !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for tidx in list(range(ntets))[238:240]: # [237:241]: # tet_nais_mean):  # this is inefficient, I need to reduce these two loops (tets and timesteps) somehow. Can we give different tets to different MPI processes?
            #with open("/gpfs/bbp.cscs.ch/home/shichkov/dan/nrnsteps/output_tet_voltage_mean_9dec2019_linstim_v1.txt", "ab") as tet_voltage_mean_outputfile:
            #    pickle.dump(tetConcs_na[tidx], tet_voltage_mean_outputfile)
            #print("tet_voltage_tot[tidx] ",tidx, ": ", tet_voltage_tot[tidx])
            
            metabolism = gen_metabolism_model() # define Julia metabolism model

            tspan_m = (float(idxm),float(idxm)+1)  #tspan_m = (float(t/1000.0),float(t/1000.0)+1) # tspan_m = (float(t/1000.0)-1.0,float(t/1000.0)) 
            um[(0,tidx)] = u0 #{(idxmk,tidx): u0 for c in ndamus.cells}

            vm=um[(idxm,tidx)] #vm=um[idxmk][int(curr_cell_seg._cellref.gid)] # to get metab specific to current cell at curret idxm timestep
            param = [tet_currents_na_tot[tidx], 0.06, tet_voltage_mean[tidx],tet_nais_mean[tidx],140.0+tetConcs_na[tidx]/CONC_FACTOR,tet_currents_k_tot[tidx],2.0+tetConcs_k[tidx], pAKTPFK2] #it was [tet_currents_tot[tidx], 0.06, dINa_mMperSec]

            #param = [1,2,3,4,5,6,7, pAKTPFK2] # , VmfPFK_a,VmrPFK_a,KATPPFK,KATP_minor_PFK,KADPPFK,KMgPFK,K23BPGPFK,KLACPFK,KF16BPPFK,KAMPPFK,KG16BPPFK,KPiPFK,KF26BPPFK,GBP,KF6PPFK ] # 1-7 are coupling with ndam and steps; p8=pAKTPFK2;


            
            
            #with open("/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/output_tetConcs_nai_param.txt", "a") as naiparam_outputfile:
            #    naiparam_outputfile.write(str(idxm))
            #    naiparam_outputfile.write("\t")
            #    naiparam_outputfile.write(str(tidx))
            #    naiparam_outputfile.write("\t tetConcs_na:")
            #    naiparam_outputfile.write(str(tetConcs_na[tidx]))

            #    naiparam_outputfile.write("\t tet_nais_mean: ")
             #   naiparam_outputfile.write(str(tet_nais_mean[tidx]))
             #   naiparam_outputfile.write("\t tet_voltage_mean: ")
             #   naiparam_outputfile.write(str(tet_voltage_mean[tidx]))
             #   naiparam_outputfile.write("\t tet_currents_na_tot: ")
             #   naiparam_outputfile.write(str(tet_currents_na_tot[tidx]))

            #    naiparam_outputfile.write("\t tet_currents_k_tot: ")
            #    naiparam_outputfile.write(str(tet_currents_k_tot[tidx]))
            #    naiparam_outputfile.write("\n")


            



            prob_metabo = de.ODEProblem(metabolism,vm,tspan_m,param)
            #print("solving for tidx ",tidx)
            with timer('julia'):
                sol = de.solve(prob_metabo,de.Tsit5(),maxiters=1e6) #alg_hints=[:stiff]) #,maxiters=1e6  #de.AutoTsit5(de.Rosenbrock23())) #, de.Rodas5()) # ,de.Tsit5()) #,callback = callbacksetBloodKsi) # dtmin=10.0 - dt less than min
            #print(sol.u[len(sol.u)])
            um[(idxm+1,tidx)] = sol.u[len(sol.u)-1] #u0 for tests #np.median(np.transpose(sol.u),axis=1)  #sol.u[len(sol.u)-1] # np.median(np.transpose(sol.u),axis=1)  #transpose and take medians #!!!! better to replace it with median of sol.u, but sol.u is 2dim and I need to find a way to calculate median properly (1 variable + multiple timesteps = 1 median, i.e. separate medians of values in timesteps for every variable)  



    
#feedback loop to constrain ndamus by metabolism outout
   
        #tet_sec_seg = [] #defaultdict(list) 
        tmgeom2 = gen_geom()
        #print("number of sec seg items ",len([(x[0],x[1]) for x in secSegMap]))
        for sec_item,seg_item in [(x[0],x[1]) for x in secSegMap]: #zip([x[0] for x in secSegMap],[x[1] for x in secSegMap]): #check it
            #print("sec ",sec_item)
            for seg_item_key, seg_item_value in seg_item.items():
                tet_sec_seg = [] #defaultdict(list) 
                #print("key ",seg_item_key) #," val ",seg_item_value)
                for pts_i,pts_coord in enumerate(seg_item_value):
                    #print("coord ",pts_coord[2])
                    #pts_coord_meters = [x*1e6 for x in pts_coord[2]] #this is because of Dan's Steps units stuff with cube and cubeBig
                    #tet_id = tmgeom.findTetByPoint(pts_coord_meters)
                    tet_id = tmgeom2.findTetByPoint(pts_coord[2])
                    #print("tet_id is ",tet_id) 
                    
                    tet_sec_seg.append((tet_id,sec_item,pts_coord[0],pts_coord[1]))
                    #print("tet_sec_seg ") #,tet_sec_seg)
        #print("tet_sec_seg: ",tet_sec_seg)
            #print("nseg: ",sec_item.nseg())

                for seg_elem in sec_item.allseg():
                    #nai_weighted_mean = 10.0 # this is a baseline value which was observed as default in ndamus
                    atpi_weighted_mean = 2.2 #1.5 # kind of baseline
                    adpi_weighted_mean =  6.3e-3



                    if str(seg_elem).split("(")[1].rstrip(")") == str(seg_item_key):
                        #print("seg elem ",seg_elem)
                        # assign Nai from um[idxm+1,tidx] to seg
                        #print("tet_sec_seg ",tet_sec_seg)
                        
                        nai_l = []
                        nai_l_w = []
                        adpi_l = []
                        for elem_i in tet_sec_seg:   
                            #print("# elem_i[0,1,3] # ",elem_i[0]," # ",elem_i[1]," # ", elem_i[3])
                            if (idxm+1,elem_i[0]) in um.keys():
                            #    print("um27 ",um[(idxm+1,elem_i[0])][27])
                                nai_l.append(um[(idxm+1,elem_i[0])][27]) #for atpi #nai_l.append(elem_i[3]*um[(idxm+1,elem_i[0])][6]) #for nai # elem_i[3]* for fraction contribution to tet scaling
                                nai_l_w.append(elem_i[3])
                                adpi_l.append(um[(idxm+1,elem_i[0])][28]) 
                        with open("test_output_ATP.txt", "a") as ATP_outfile:
                            ATP_outfile.write(str(sum(nai_l)/len(nai_l)))
                            ATP_outfile.write("\n")
 
                                #print("! ",elem_i[0]," #" ,um[(idxm+1,elem_i[0])][6]) # Nai
                                #seg_elem.nai = um[(idxm+1,elem_i[0])][6]  #asssign Na_i to seg for next big t step (idxm+1) #check it!
                        atpi_weighted_mean = sum(nai_l)/len(nai_l)    #sum(nai_l)/sum(nai_l_w) 
                        adpi_weighted_mean = sum(adpi_l)/len(adpi_l)  #6.3e-3  #sum(nai_l)/sum(nai_l_w)  
                          
                        seg_elem.atpi = atpi_weighted_mean #um[(idxm+1,elem_i[0])][6]  #1.5 #TO TEST!!!!! should be:  #asssign Na_i to seg for next big t step (idxm+1) #check it!
                        seg_elem.adpi = adpi_weighted_mean #um[(idxm+1,elem_i[0])][6]  #1.5 #TO TEST!!!!! should be:  #asssign Na_i to seg for next big t step (idxm+1) #check it!

#                for secmap in : #neurSecmap:
#                    print(secmap)
                    
                    #for sec, fractlist in secmap:
                    #     for seg, fractitem in zip(sec.allseg(), fractlist):
 

 
            #with open("output_u.txt", "a") as u_outputfile:
            #    u_outputfile.write(sol.u)
            #with open("output_t.txt", "a") as t_outputfile:
            #    t_outputfile.write(sol.t)
                                
        #with open("/gpfs/bbp.cscs.ch/project/proj34/scratch/polina/output_um.txt", "wb") as um_outputfile:
        #    pickle.dump(um, um_outputfile)
        
        #i += 1
        
    logging.info(textwrap.dedent("""\
        Simulation finished. Timings:
	   - STEPS: {steps_cum:g}
	   - Neuron: {neuron_cum:g}
	   - comm_allred: {comm_allred:g}
	   - float_assign: {float_assign:g}
	   - curr2conc: {curr2conc:g}
	   - setBatchTetConcs: {setBatchTetConcs:g}
	   - Julia: {julia:g}""".format_map(_timings)))

if __name__ == "__main__":
    main()
