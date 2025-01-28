

import numpy as np
import cantera as ct
import os
import time
import pandas as pd
from pickle import load
import pickle
#from PyQt4 import uic
#from PyQt4.QtGui import * 
#from PyQt4.QtCore import * import pickle
#from src.core.def_tools import *
#from src.ct.def_ct_tools import Xstr
#from src.ct.senkin import senkin
#from src.ct.psr import S_curve
from src.ck.def_cheminp import skeletal
from src.ct.ck2cti_GPS import ck2cti

#from dialog_GPS import dialog_GPS
#from dialog_PFA import dialog_PFA
#from dialog_database import dialog_database
#from dialog_mech import dialog_mech
#from dialog_view_mech import dialog_view_mech

#from find_tau_ign import find_tau_ign


#from src.ct.def_ct_tools import load_raw
from src.ct.def_ct_tools import soln2raw
from src.ct.def_ct_tools import save_raw_npz
from src.core.def_GPS import GPS_algo
from src.core.def_build_graph import build_flux_graph
#from networkx.readwrite import json_graph
#from src.core.def_tools import st2name
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from matplotlib.path import Path

def write_sk_inp(species_kept, dir_mech_de, dir_mech_sk, notes):

	species_kept = list(species_kept)
	n_sp = len(species_kept)
	print('total: '+str(n_sp))

	notes.append('! number of species = '+str(n_sp))
	skeletal(dir_mech_de, dir_mech_sk, species_kept, notes=notes)
	ck2cti(dir_mech_sk)

	f = open(os.path.join(dir_mech_sk,'ns.txt'),'w')
	f.write(str(n_sp))
	f.close()

def auto_ign_build(soln, T0, atm, X0, start, end, norm_dt, ign_dt, t_ign=None, true_start=0.0, dir_raw = None):
    ign=False
    raw_all = None
    soln.TPX = T0, atm*101400, X0
    reactor = ct.IdealGasConstPressureReactor(soln)
    network = ct.ReactorNet([reactor])
    t = start
    dt = norm_dt
    while t <= end:
        network.advance(t)
        raw_all = soln2raw(t, 'time', soln, raw_all)
        network.advance(t)
        if raw_all['heat_release_rate'][-1] > ign_threshold and not ign:
            dt = ign_dt
            ign = True
        elif raw_all['heat_release_rate'][-1] < ign_threshold and ign:
            dt = norm_dt
            ign =False
        if t_ign is not None:
            if t+true_start>t_ign-3*ign_dt and t<t_ign+1*ign_dt and dt != ign_dt/5:
                dt_prev = dt
                dt = ign_dt/5
            elif t+true_start>t_ign+1*ign_dt:
                dt=dt_prev
        t += dt
    raw = save_raw_npz(raw_all, 'ign')
    return raw

def auto_ign_build_det(soln, T0, atm, X0, start, dt, end_threshold, end, dir_raw = None):
    raw_all = None
    soln.TPX = T0, atm*101400, X0
    reactor = ct.IdealGasConstPressureReactor(soln)
    network = ct.ReactorNet([reactor])
    t = start
    ign = False
    while t <= end:
        network.advance(t)
        raw_all = soln2raw(t, 'time', soln, raw_all)
        network.advance(t)
        t += dt
        if raw_all['heat_release_rate'][-1] > end_threshold and not ign:
            ign=True
        elif raw_all['heat_release_rate'][-1] < end_threshold and ign:
            break
    raw = save_raw_npz(raw_all, 'ign')
    return raw

def findIgnInterval(hrr, threshold):
    ign_count = 0
    ss_count = 0
    ign = False
    for i in range(len(hrr)):
        if ign_count > 0.5 and ign_count < 10.5:
            ign_count += 1
            continue
        if ss_count > 0.5 and ss_count < 10.5:
            ss_count += 1
            continue
        elif hrr[i] > threshold and not ign:
            ign_start = i
            ign = True
            ign_count+=1
        elif hrr[i] < threshold and ign:
            ign_end = i
            ign = False
            ss_count=1
    try:
        ign_start
    except NameError:
        ign_start = 0
    try:
        ign_end
    except NameError:
        ign_end = len(hrr)-1
    return ign_start, ign_end
def convertChemNames(names):
    newNames=[]
    for name in names:
        parenthetical = False
        newNameLst=['$']
        numbers=('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')
        for character in name:
            if character == '(':
                parenthetical = True
            if character in numbers and not parenthetical:
                newNameLst.append('_'+character)
            else:
                newNameLst.append(character)
        newNameLst.append('$')
        newNames.append(''.join(newNameLst))
    return newNames

#'CH4', 'H2O', 'OH', 'H', 'CO', 'O2', 'CO2', 'O', 'CH3', 'CH', 'H2', 'N2', 'NH3', 'NO'
T0_in = 1800.0
X0_in = 'CH4:1.0, N2:0.767, O2:0.233'; #CO2:0.06305, H2O:0.05335, NH3:0.15';
atm = 1.0

tracked_specs = ['CH4', 'H2O', 'OH', 'H', 'CO', 'O2', 'CO2', 'O', 'CH3', 'CH', 'H2']
t_end = 0.005
dt = 2e-7
#2e-8
norm_dt = 2e-5
ign_dt = 2e-6
end_threshold = 2e5 #2e5 before
soln_in = ct.Solution('gri30.cti')


T0 = T0_in
X0 = X0_in




auto_ign_det = auto_ign_build_det(soln_in, T0, atm, X0, 0, dt, end_threshold, t_end, dir_raw = 'ign')
norm_Dt = auto_ign_det['axis0'][-1]/10
ign_Dt = auto_ign_det['axis0'][-1]/30
norm_dt = norm_Dt/10
#10
ign_dt = norm_Dt/40
#40
ign_threshold = max(auto_ign_det['heat_release_rate'])/4e2
t_ign = auto_ign_det['axis0'][np.argmax(np.array(auto_ign_det['heat_release_rate']))]
det_start, det_end = findIgnInterval(auto_ign_det['heat_release_rate'], ign_threshold)
elements = ['C', 'H', 'O']
sources = ['CH4', 'O2']
targets = ['CO2', 'H2O']
det_spec_strs = []
for spec in soln_in.species():
    det_spec_strs.append(str(spec)[9:-1])
    
species_kept = set(sources) | set(targets) | set(['N2'])
hub_scores = {}
for i_pnt in range(49):
#        if i_pnt == len(list(auto_ign['temperature']))-1 or auto_ign['temperature'][i_pnt+1]-auto_ign['temperature'][i_pnt] > 50: 
            for e in elements:
                flux_graph = build_flux_graph(soln_in, auto_ign_det, e, path_save='flux_graph', overwrite=True, i0=i_pnt*len(auto_ign_det['axis0'])//50, i1='eq', constV=False)
                for s in sources:
                    for t in targets:
                        if s in flux_graph.nodes() and t in flux_graph.nodes():
                            GPS_results = GPS_algo(soln_in, flux_graph, s, t, path_save=None, K=1, alpha=0.001, beta=0.5, normal='max', iso=None, overwrite=False, raw='unknown',notes=None, gamma=None)
                            new_species = GPS_results['species'].keys()
                            species_kept |= new_species
                            for hub in list(GPS_results['hubs'].keys()):
                                if hub not in list(hub_scores.keys()):
                                    hub_scores[hub] = GPS_results['hubs'][hub]['score']
                                else:
                                    if GPS_results['hubs'][hub]['score'] > hub_scores[hub]:
                                        hub_scores[hub] = GPS_results['hubs'][hub]['score']  
write_sk_inp(species_kept, 'de_mech', 'sk_mech', notes=[])
const_sk_mech = ct.Solution(os.path.join('sk_mech', 'chem.cti'))
const_spec_strs = []
for spec in const_sk_mech.species():
    const_spec_strs.append(str(spec)[9:-1])
    const_specs=const_spec_strs
auto_ign_test = auto_ign_build(const_sk_mech, T0, atm, X0, 0, auto_ign_det['axis0'][-1], norm_dt, ign_dt, t_ign=t_ign, dir_raw = 'ign')
const_n_spec = len(list(const_sk_mech.species()))
const_n_rxn = len(list(const_sk_mech.reactions()))

const_start, const_end = findIgnInterval(auto_ign_test['heat_release_rate'], ign_threshold)

tot_const_hubs = []
sorted_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)
real_sorted_hubs = []
for hub in sorted_hubs:
    real_sorted_hubs.append(hub[0])
for hub in real_sorted_hubs[:4]:
    if hub not in tot_const_hubs:
        tot_const_hubs.append(hub)
const_hubs = real_sorted_hubs[:4]

var_specs = pd.read_csv(os.path.join('TrainingData/Sandia_100sims', 'var_spec_nums.csv')).columns.to_list()[:-1]
always_specs = pd.read_csv(os.path.join('TrainingData/Sandia_100sims', 'always_spec_nums.csv')).columns.to_list()[:-1]
if var_specs[0][:2] == '# ':
    var_specs[0] = var_specs[0][2:]
if always_specs[0][:2] == '# ':
    always_specs[0] = always_specs[0][2:]   


ad_times = []
ad_temps = []
ad_CH4 = []
ad_CH2O = []
ad_CO2 = []
ad_CO = []
ad_OH = []
ad_O = []
ad_H = []
ad_H2O = []
ad_CH3 = []
ad_CH = []
ad_HRR = []
ad_hubs = []
tot_ad_hubs = []

ad_temp_err = []
ad_CH4_err = []
ad_CO2_err = []
ad_CO_err = []
ad_OH_err = []
ad_O_err = []
ad_H_err = []
ad_H2O_err = []
ad_CH3_err = []
ad_CH_err = []
ad_HRR_err = []

t_steps = []
n_ad_reactions = []
n_ad_species = []
tick = 0.0
Dt = norm_Dt
ign = False
ign_count = 0
i=0
model = load_model('Artificial Neural Networks/Sandia_100sims.h5')

t_end = auto_ign_det['axis0'][-1]

ad_specs = []
while tick < t_end:
#    if os.path.exists('ign/mole_fraction.csv'):
#        os.remove('ign/mole_fraction.csv')
#    if os.path.exists('ign/net_reaction_rate.csv'):
#        os.remove('ign/net_reaction_rate.csv')
#    if os.path.exists('ign/pressure.csv'):
#        os.remove('ign/pressure.csv')
#    if os.path.exists('ign/reaction_list.csv'):
#        os.remove('ign/reaction_list.csv')
#    if os.path.exists('ign/species_list.csv'):
#        os.remove('ign/species_list.csv')
#    if os.path.exists('ign/temperature.csv'):
#        os.remove('ign/temperature.csv')
#    if os.path.exists('ign/time.csv'):
#        os.remove('ign/time.csv')     
#    t0 = time.perf_counter()
    check=0
    while True:
        auto_ign = auto_ign_build(soln_in, T0, atm, X0, 0, Dt, norm_dt, ign_dt, t_ign=t_ign, true_start=tick, dir_raw = 'ign') 
        alpha = 0.001
        species_kept = set(sources) | set(targets) | set(['N2'])
        hub_scores = {}
        for i_pnt in range(len(auto_ign['axis0'])):
    #            if i_pnt == len(list(auto_ign['temperature']))-1 or auto_ign['temperature'][i_pnt+1]-auto_ign['temperature'][i_pnt] > 20: 
                for e in elements:
                    flux_graph = build_flux_graph(soln_in, auto_ign, e, path_save='flux_graph', overwrite=True, i0=i_pnt, i1='eq', constV=False)
                    #print(flux_graph.nodes())
                    for s in sources:
                        for t in targets:
                            if s in flux_graph.nodes() and t in flux_graph.nodes():
                                GPS_results = GPS_algo(soln_in, flux_graph, s, t, path_save=None, K=1, alpha=alpha, beta=0.5, normal='max', iso=None, overwrite=False, raw='unknown',notes=None, gamma=None)
                                new_species = GPS_results['species'].keys()
                                species_kept |= new_species
                                for hub in list(GPS_results['hubs'].keys()):
                                    if hub not in list(hub_scores.keys()):
                                        hub_scores[hub] = GPS_results['hubs'][hub]['score']
                                    else:
                                        if GPS_results['hubs'][hub]['score'] > hub_scores[hub]:
                                            hub_scores[hub] = GPS_results['hubs'][hub]['score']                 
        write_sk_inp(species_kept, 'de_mech', 'sk_mech', notes=[])
        sk_mech = ct.Solution(os.path.join('sk_mech', 'chem.cti'))
        
        n=0
        if i > 0.5:
            new_spec_list = []
            for spec in sk_mech.species():
                new_spec_list.append(str(spec)[9:-1])
            x_remove = []
            for spec in spec_list:
                if spec not in new_spec_list:
                    x_remove.append(x_labels[spec_list.index(spec)])
            new_labels = x_labels.copy()
            for x in x_remove:
                new_labels.remove(x)
            X0 = ','.join(new_labels)

    #    t0 = time.perf_counter()
        auto_ign_temp = auto_ign_build(sk_mech, T0, atm, X0, 0, Dt, norm_dt, ign_dt, t_ign=t_ign, true_start=tick, dir_raw = 'ign')  
        if check > 0.5: 
            break
        elif ign_count > 0.5 and ign_count <= norm_Dt//ign_Dt+1:
            ign_count+=1
            break
        elif auto_ign_temp['heat_release_rate'][-1] < ign_threshold and not ign:
            break
        elif auto_ign_temp['heat_release_rate'][-1] >= ign_threshold and not ign:
            Dt = ign_Dt
            ign = True
            ign_count=1
            check += 1            
        elif auto_ign_temp['heat_release_rate'][0] >= ign_threshold and ign:
            break
        else:
            Dt = norm_Dt
            ign = False
            check += 1
            
#    adapt_sim_time += time.perf_counter() - t0
    auto_ign_temp_time = auto_ign_temp['axis0'][:]
    auto_ign_temp_t = auto_ign_temp['temperature'][:]
    t_steps.append(tick)
    n_ad_reactions.append(len(list(sk_mech.reactions())))
    n_ad_species.append(len(list(sk_mech.species())))
    x_labels = []
    spec_list = []
    n = 0
    for spec in sk_mech.species():
        x_labels.append(str(spec)[9:-1] + ':' + str(auto_ign_temp['mole_fraction'].tolist()[-1][n]))
        spec_list.append(str(spec)[9:-1])
        n += 1
    if 'CH4' in spec_list:
        auto_ign_temp_CH4 = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CH4')])
    else: auto_ign_temp_CH4 = [0.0] * len(auto_ign_temp['axis0'])
    if 'CH2O' in spec_list:
        auto_ign_temp_CH2O = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CH2O')])
    else: auto_ign_temp_CH2O = [0.0] * len(auto_ign_temp['axis0'])
    if 'CO2' in spec_list:
        auto_ign_temp_CO2 = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CO2')])
    else: auto_ign_temp_CO2 = [0.0] * len(auto_ign_temp['axis0'])
    if 'CO' in spec_list:
        auto_ign_temp_CO = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CO')])
    else: auto_ign_temp_CO = [0.0] * len(auto_ign_temp['axis0'])
    if 'OH' in spec_list:
        auto_ign_temp_OH = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('OH')])
    else: auto_ign_temp_OH = [0.0] * len(auto_ign_temp['axis0'])
    if 'O' in spec_list:
        auto_ign_temp_O = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('O')])
    else: auto_ign_temp_O = [0.0] * len(auto_ign_temp['axis0'])
    if 'H' in spec_list:
        auto_ign_temp_H = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('H')])
    else: auto_ign_temp_H = [0.0] * len(auto_ign_temp['axis0'])
    if 'H2O' in spec_list:
        auto_ign_temp_H2O = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('H2O')])
    else: auto_ign_temp_H2O = [0.0] * len(auto_ign_temp['axis0'])
    if 'CH3' in spec_list:
        auto_ign_temp_CH3 = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CH3')])
    else: auto_ign_temp_CH3 = [0.0] * len(auto_ign_temp['axis0'])
    if 'CH' in spec_list:
        auto_ign_temp_CH = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CH')])
    else: auto_ign_temp_CH = [0.0] * len(auto_ign_temp['axis0'])
    if 'H2' in spec_list:
        auto_ign_temp_H2 = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('H2')])
    else: auto_ign_temp_H2 = [0.0] * len(auto_ign_temp['axis0'])
    

    auto_ign_temp_HRR = list(auto_ign_temp['heat_release_rate'][:])
    X0 = ','.join(x_labels)    
    for j, temp in enumerate(auto_ign_temp_t):
        for k, d_time in enumerate(auto_ign_det['axis0'][:]):
            if d_time > tick + auto_ign_temp['axis0'][j] or k == len(auto_ign_det['axis0'])-1:
                det_ind = k
                break
        ad_times.append(tick + auto_ign_temp_time[j])
        ad_temps.append(temp)
        ad_CH4.append(auto_ign_temp_CH4[j])
        ad_CH2O.append(auto_ign_temp_CH2O[j])
        ad_CO2.append(auto_ign_temp_CO2[j])
        ad_CO.append(auto_ign_temp_CO[j])
        ad_OH.append(auto_ign_temp_OH[j])
        ad_O.append(auto_ign_temp_O[j])
        ad_H.append(auto_ign_temp_H[j])
        ad_H2O.append(auto_ign_temp_H2O[j])
        ad_CH3.append(auto_ign_temp_CH3[j])
        ad_CH.append(auto_ign_temp_CH[j])
        ad_HRR.append(auto_ign_temp_HRR[j])
        
        # ad_temp_err.append((abs(auto_ign_temp_t[j]-auto_ign_det['temperature'][det_ind])/auto_ign_det['temperature'][det_ind])*100)
        # ad_CH4_err.append((abs(auto_ign_temp_CH4[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH4')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH4')]+div_0_prev))*100)
        # ad_CO2_err.append((abs(auto_ign_temp_CO2[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CO2')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CO2')]+div_0_prev))*100)
        # ad_CO_err.append((abs(auto_ign_temp_CO[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CO')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CO')]+div_0_prev))*100)
        # ad_OH_err.append((abs(auto_ign_temp_OH[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('OH')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('OH')]+div_0_prev))*100)
        # ad_O_err.append((abs(auto_ign_temp_O[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('O')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('O')]+div_0_prev))*100)
        # ad_H_err.append((abs(auto_ign_temp_H[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('H')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('H')]+div_0_prev))*100)
        # ad_H2O_err.append((abs(auto_ign_temp_H2O[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('H2O')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('H2O')]+div_0_prev))*100)
        # ad_CH3_err.append((abs(auto_ign_temp_CH3[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH3')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH3')]+div_0_prev))*100)
        # ad_CH_err.append((abs(auto_ign_temp_CH[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH')]+div_0_prev))*100)
        # ad_HRR_err.append((abs(auto_ign_temp_HRR[j]-auto_ign_det['heat_release_rate'][det_ind])/abs(auto_ign_det['heat_release_rate'][det_ind]))*100)
    ad_specs.append(spec_list)
    sorted_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)
    real_sorted_hubs = []
    for hub in sorted_hubs:
        real_sorted_hubs.append(hub[0])
    for hub in real_sorted_hubs[:4]:
        if hub not in tot_ad_hubs:
            tot_ad_hubs.append(hub)
    ad_hubs.append(real_sorted_hubs[:4])
    T0 = auto_ign_temp_t[-1]
    tick = ad_times[-1]
    i+=1


# f = soln_in.reaction
# print(f)

T0 = T0_in;
X0 = X0_in;
ad_pred_times = []
ad_pred_temps = []
ad_pred_CH4 = []
ad_pred_CH2O = []
ad_pred_CO2 = []
ad_pred_CO = []
ad_pred_OH = []
ad_pred_O = []
ad_pred_H = []
ad_pred_H2O = []
ad_pred_CH3 = []
ad_pred_CH = []
ad_pred_HRR = []

ad_pred_temp_err = []
ad_pred_CH4_err = []
ad_pred_CH2O_err = []
ad_pred_CO2_err = []
ad_pred_CO_err = []
ad_pred_OH_err = []
ad_pred_O_err = []
ad_pred_H_err = []
ad_pred_H2O_err = []
ad_pred_CH3_err = []
ad_pred_CH_err = []
ad_pred_HRR_err = []

t_pred_steps = []
n_ad_pred_reactions = []
n_ad_pred_species = []
tick = 0.0
Dt = norm_Dt
ign = False
ign_count = 0
i=0

count = 0
pred_specs = []
while tick < t_end:
#    if os.path.exists('ign/mole_fraction.csv'):
#        os.remove('ign/mole_fraction.csv')
#    if os.path.exists('ign/net_reaction_rate.csv'):
#        os.remove('ign/net_reaction_rate.csv')
#    if os.path.exists('ign/pressure.csv'):
#        os.remove('ign/pressure.csv')
#    if os.path.exists('ign/reaction_list.csv'):
#        os.remove('ign/reaction_list.csv')
#    if os.path.exists('ign/species_list.csv'):
#        os.remove('ign/species_list.csv')
#    if os.path.exists('ign/temperature.csv'):
#        os.remove('ign/temperature.csv')
#    if os.path.exists('ign/time.csv'):
#        os.remove('ign/time.csv')     
#    t0 = time.perf_counter()
    check = 0
    while True:
        soln_in.TPX= T0, atm, X0
        pred_input = [T0,atm]
        for spec in tracked_specs:
            if spec in soln_in.mole_fraction_dict().keys():
                pred_input.append(soln_in.mole_fraction_dict()[spec])
            else:
                pred_input.append(0.0)
        pred_input = np.array([pred_input])
        min_max_scaler = load(open('Min-Max Scalers/Sandia_100sims.pkl', 'rb'))
        pred_input_proc = min_max_scaler.transform(pred_input)
        prediction = model.predict(pred_input_proc)
        pred_kept_specs = always_specs.copy()
        for j in range(len(prediction[0, :])):
            if prediction[0, j] > 0.5:
                pred_kept_specs.append(var_specs[j])
        # if tick==0.0:
        #     pred_kept_specs.append('CH2OH')
        #     pred_kept_specs.append('C2H6')
        #     pred_kept_specs.append('CH3OH')
        # if count < len(adapt_all_specs):    
        #     pred_kept_specs = adapt_all_specs[count]
        write_sk_inp(pred_kept_specs, 'de_mech', 'pred_mech', notes=[])
        sk_mech = ct.Solution(os.path.join('pred_mech', 'chem.cti'))
        
        n=0
        if i > 0.5:
            new_spec_list = []
            for spec in sk_mech.species():
                new_spec_list.append(str(spec)[9:-1])
            x_remove = []
            for spec in spec_list:
                if spec not in new_spec_list:
                    x_remove.append(x_labels[spec_list.index(spec)])
            new_labels = x_labels.copy()
            for x in x_remove:
                new_labels.remove(x)
            X0 = ','.join(new_labels)

    #    t0 = time.perf_counter()
        # if count > 0.1:
        #     Dt = ign_Dt
        auto_ign_temp = auto_ign_build(sk_mech, T0, atm, X0, 0, Dt, norm_dt, ign_dt, t_ign=t_ign, true_start=tick, dir_raw = 'ign')  
        if check > 0.5: 
            break
        elif ign_count > 0.5 and ign_count <= norm_Dt//ign_Dt+1:
            ign_count+=1
            break
        elif auto_ign_temp['heat_release_rate'][-1] < ign_threshold and not ign:
            break
        elif auto_ign_temp['heat_release_rate'][-1] >= ign_threshold and not ign:
            Dt = ign_Dt
            ign = True
            ign_count=1
            check += 1            
        elif auto_ign_temp['heat_release_rate'][0] >= ign_threshold and ign:
            break
        else:
            Dt = norm_Dt
            ign = False
            check += 1
            
#    adapt_sim_time += time.perf_counter() - t0
    auto_ign_temp_time = auto_ign_temp['axis0'][:]
    auto_ign_temp_t = auto_ign_temp['temperature'][:]
    t_pred_steps.append(tick)
    n_ad_pred_reactions.append(len(list(sk_mech.reactions())))
    n_ad_pred_species.append(len(list(sk_mech.species())))
    pred_specs.append([pred_kept_specs])
    x_labels = []
    spec_list = []
    n = 0
    for spec in sk_mech.species():
        x_labels.append(str(spec)[9:-1] + ':' + str(auto_ign_temp['mole_fraction'].tolist()[-1][n]))
        spec_list.append(str(spec)[9:-1])
        n += 1
    if 'CH4' in spec_list:
        auto_ign_temp_CH4 = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CH4')])
    else: auto_ign_temp_CH4 = [0.0] * len(auto_ign_temp['axis0'])
    if 'CH2O' in spec_list:
        auto_ign_temp_CH2O = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CH2O')])
    else: auto_ign_temp_CH2O = [0.0] * len(auto_ign_temp['axis0'])
    if 'CO2' in spec_list:
        auto_ign_temp_CO2 = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CO2')])
    else: auto_ign_temp_CO2 = [0.0] * len(auto_ign_temp['axis0'])
    if 'CO' in spec_list:
        auto_ign_temp_CO = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CO')])
    else: auto_ign_temp_CO = [0.0] * len(auto_ign_temp['axis0'])
    if 'OH' in spec_list:
        auto_ign_temp_OH = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('OH')])
    else: auto_ign_temp_OH = [0.0] * len(auto_ign_temp['axis0'])
    if 'O' in spec_list:
        auto_ign_temp_O = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('O')])
    else: auto_ign_temp_O = [0.0] * len(auto_ign_temp['axis0'])
    if 'H' in spec_list:
        auto_ign_temp_H = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('H')])
    else: auto_ign_temp_H = [0.0] * len(auto_ign_temp['axis0'])
    if 'H2O' in spec_list:
        auto_ign_temp_H2O = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('H2O')])
    else: auto_ign_temp_H2O = [0.0] * len(auto_ign_temp['axis0'])
    if 'CH3' in spec_list:
        auto_ign_temp_CH3 = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CH3')])
    else: auto_ign_temp_CH3 = [0.0] * len(auto_ign_temp['axis0'])
    if 'CH' in spec_list:
        auto_ign_temp_CH = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('CH')])
    else: auto_ign_temp_CH = [0.0] * len(auto_ign_temp['axis0'])
    if 'H2' in spec_list:
        auto_ign_temp_H2 = list(np.array(auto_ign_temp['mole_fraction'].tolist())[:, spec_list.index('H2')])
    else: auto_ign_temp_H2 = [0.0] * len(auto_ign_temp['axis0'])
    auto_ign_temp_HRR = list(auto_ign_temp['heat_release_rate'][:])
    X0 = ','.join(x_labels)    
    for j, temp in enumerate(auto_ign_temp_t):
        for k, d_time in enumerate(auto_ign_det['axis0'][:]):
            if d_time > tick + auto_ign_temp['axis0'][j] or k == len(auto_ign_det['axis0'])-1:
                det_ind = k
                break
        ad_pred_times.append(tick + auto_ign_temp_time[j])
        ad_pred_temps.append(temp)
        ad_pred_CH4.append(auto_ign_temp_CH4[j])
        ad_pred_CH2O.append(auto_ign_temp_CH2O[j])
        ad_pred_CO2.append(auto_ign_temp_CO2[j])
        ad_pred_CO.append(auto_ign_temp_CO[j])
        ad_pred_OH.append(auto_ign_temp_OH[j])
        ad_pred_O.append(auto_ign_temp_O[j])
        ad_pred_H.append(auto_ign_temp_H[j])
        ad_pred_H2O.append(auto_ign_temp_H2O[j])
        ad_pred_CH3.append(auto_ign_temp_CH3[j])
        ad_pred_CH.append(auto_ign_temp_CH[j])
        ad_pred_HRR.append(auto_ign_temp_HRR[j])
        
        # g= flux_graph['O2']['OH']['flux']
        # print(g)
        # ad_temp_err.append((abs(auto_ign_temp_t[j]-auto_ign_det['temperature'][det_ind])/auto_ign_det['temperature'][det_ind])*100)
        # ad_CH4_err.append((abs(auto_ign_temp_CH4[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH4')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH4')]+div_0_prev))*100)
        # ad_CO2_err.append((abs(auto_ign_temp_CO2[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CO2')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CO2')]+div_0_prev))*100)
        # ad_CO_err.append((abs(auto_ign_temp_CO[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CO')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CO')]+div_0_prev))*100)
        # ad_OH_err.append((abs(auto_ign_temp_OH[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('OH')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('OH')]+div_0_prev))*100)
        # ad_O_err.append((abs(auto_ign_temp_O[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('O')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('O')]+div_0_prev))*100)
        # ad_H_err.append((abs(auto_ign_temp_H[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('H')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('H')]+div_0_prev))*100)
        # ad_H2O_err.append((abs(auto_ign_temp_H2O[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('H2O')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('H2O')]+div_0_prev))*100)
        # ad_CH3_err.append((abs(auto_ign_temp_CH3[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH3')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH3')]+div_0_prev))*100)
        # ad_CH_err.append((abs(auto_ign_temp_CH[j]-auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH')])/(auto_ign_det['mole_fraction'].tolist()[det_ind][det_spec_strs.index('CH')]+div_0_prev))*100)
        # ad_HRR_err.append((abs(auto_ign_temp_HRR[j]-auto_ign_det['heat_release_rate'][det_ind])/abs(auto_ign_det['heat_release_rate'][det_ind]))*100)
    T0 = auto_ign_temp_t[-1]
    tick = ad_pred_times[-1]
    i+=1
    count += 1
ad_start, ad_end = findIgnInterval(ad_HRR, ign_threshold)
ad_pred_start, ad_pred_end = findIgnInterval(ad_pred_HRR, ign_threshold)
if 'CH' in const_spec_strs:
    const_CH = list(np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('CH')])
else:
    const_CH = None
sup_results = {'det_times':auto_ign_det['axis0'],
               'det_temps':auto_ign_det['temperature'],
               'det_CH4': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('CH4')],
               'det_CH2O': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('CH2O')],
               'det_CO2': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('CO2')],
               'det_CO': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('CO')],
               'det_OH': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('OH')],
               'det_CH': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('CH')],
               'det_O': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('O')],
               'det_H': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('H')],
               'det_CH3': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('CH3')],
               'det_H2O': np.array(auto_ign_det['mole_fraction'].tolist())[:, det_spec_strs.index('H2O')],
               'det_HRR': list(auto_ign_det['heat_release_rate']),
               'const_times':auto_ign_test['axis0'],
               'const_temps':auto_ign_test['temperature'],
               'const_CH4': np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('CH4')],
               'const_CH2O': np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('CH2O')],
               'const_CO2': np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('CO2')],
               'const_CO': np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('CO')],
               'const_OH': np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('OH')],
               'const_CH': const_CH,
               'const_O': np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('O')],
               'const_H': np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('H')],
               'const_CH3': np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('CH3')],
               'const_H2O': np.array(auto_ign_test['mole_fraction'].tolist())[:, const_spec_strs.index('H2O')],
               'const_HRR': list(auto_ign_test['heat_release_rate']),
               'ad_times': ad_times,
               'ad_temps': ad_temps,
               'ad_CH4': ad_CH4,
               'ad_CH2O': ad_CH2O,
               'ad_CO2': ad_CO2,
               'ad_CO': ad_CO,
               'ad_OH': ad_OH,
               'ad_CH': ad_CH,
               'ad_O': ad_O,
               'ad_H': ad_H,
               'ad_CH3': ad_CH3,
               'ad_H2O': ad_H2O,
               'ad_HRR': ad_HRR,
               'ad_pred_times': ad_pred_times,
               'ad_pred_temps': ad_pred_temps,
               'ad_pred_CH4': ad_pred_CH4,
               'ad_pred_CH2O': ad_pred_CH2O,
               'ad_pred_CO2': ad_pred_CO2,
               'ad_pred_CO': ad_pred_CO,
               'ad_pred_OH': ad_pred_OH,
               'ad_pred_CH': ad_pred_CH,
               'ad_pred_O': ad_pred_O,
               'ad_pred_H': ad_pred_H,
               'ad_pred_CH3': ad_pred_CH3,
               'ad_pred_H2O': ad_pred_H2O,
               'ad_pred_HRR': ad_pred_HRR,
               'det_start': det_start,
               'det_end': det_end,
               'const_start': const_start,
               'const_end': const_end,
               'ad_start': ad_start,
               'ad_end': ad_end,
               'ad_pred_start': ad_pred_start,
               'ad_pred_end': ad_pred_end,
               't_steps': t_steps,
               't_pred_steps': t_pred_steps,
               'const_n_spec': const_n_spec,
               'const_n_rxn': const_n_rxn,
               'n_ad_species': n_ad_species,
               'n_ad_reactions': n_ad_reactions,
               'n_ad_pred_species': n_ad_pred_species,
               'n_ad_pred_reactions': n_ad_pred_reactions,
               'det_specs': det_spec_strs,
               'const_specs': const_specs,
               'always_specs': always_specs,
               'var_specs': var_specs,
               'pred_specs': pred_specs}

with open('results_data/t1310_e0.65_h1.1_a0.001', 'wb') as results_file:
    pickle.dump(sup_results, results_file)

plt.rcParams['font.size'] = '18'
plt.rcParams["figure.figsize"] = (30,25)
# plt.plot(auto_ign_det['axis0'], auto_ign_det['temperature'], 'r-')
f, ((ax1, ax2, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11), (ax12, ax13, ax14)) = plt.subplots(4, 3, sharex=True)

ax1.plot(auto_ign_det['axis0'][det_start:det_end], auto_ign_det['temperature'][det_start:det_end], 'r-')
ax1.plot(auto_ign_test['axis0'][const_start:const_end], auto_ign_test['temperature'][const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax1.plot(ad_times[ad_start:ad_end], ad_temps[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax1.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_temps[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax1.set(xlabel='Time (s)', ylabel = 'Temperature (K)', title='Temperature vs Time')
ax1.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 12})


ax2.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CH4')]), 'r-')
ax2.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CH4')]), 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax2.plot(ad_times[ad_start:ad_end], ad_CH4[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax2.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH4[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax2.set(xlabel='Time (s)', ylabel=r'$CH_4$ Mole Fraction', title=r'$CH_4$ Mole Fraction vs Time')
ax2.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 12})


# ax3.plot(t_steps, n_ad_species, 'bo-')
# ax3.plot(t_steps, [const_n_spec]*len(t_steps), 'go--')
# ax3.plot(t_pred_steps, n_ad_pred_species, 'mo-')
# ax3.set(xlabel='Time (s)', ylabel='Number of Species', title='Number of Species Used vs. Time')
# ax3.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': 12})

# ax4.plot(t_steps, n_ad_reactions, 'bo-')
# ax4.plot(t_steps, [const_n_rxn]*len(t_steps), 'go--')
# ax4.plot(t_pred_steps, n_ad_pred_reactions, 'mo-')
# ax4.set(xlabel='Time (s)', ylabel='Number of Reactions', title='Number of Reactions Used vs. Time')
# ax4.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': 12})


ax14.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CO2')]), 'r-')
ax14.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CO2')]), 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax14.plot(ad_times[ad_start:ad_end], ad_CO2[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax14.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CO2[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax14.set(xlabel='Time (s)', ylabel=r'$CO_2$ Mole Fraction', title=r'$CO_2$ Mole Fraction vs Time')
ax14.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 14})


ax6.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CO')]), 'r-')
ax6.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CO')]), 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax6.plot(ad_times[ad_start:ad_end], ad_CO[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax6.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CO[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax6.set(xlabel='Time (s)', ylabel=r'$CO$ Mole Fraction', title=r'$CO$ Mole Fraction vs Time')
ax6.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 14})


ax7.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('OH')]), 'r-')
ax7.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('OH')]), 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax7.plot(ad_times[ad_start:ad_end], ad_OH[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax7.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_OH[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax7.set(xlabel='Time (s)', ylabel=r'$OH$ Mole Fraction', title=r'$OH$ Mole Fraction vs Time')
ax7.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 14})


ax8.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('O')]), 'r-')
ax8.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('O')]), 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax8.plot(ad_times[ad_start:ad_end], ad_O[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax8.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_O[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax8.set(xlabel='Time (s)', ylabel=r'$O$ Mole Fraction', title=r'$O$ Mole Fraction vs Time')
ax8.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 14})


ax9.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('H')]), 'r-')
ax9.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('H')]), 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax9.plot(ad_times[ad_start:ad_end], ad_H[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax9.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_H[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax9.set(xlabel='Time (s)', ylabel=r'$H$ Mole Fraction', title=r'$H$ Mole Fraction vs Time')
ax9.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 14})


ax10.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('H2O')]), 'r-')
ax10.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('H2O')]), 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax10.plot(ad_times[ad_start:ad_end], ad_H2O[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax10.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_H2O[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax10.set(xlabel='Time (s)', ylabel=r'$H_2O$ Mole Fraction', title=r'$H_2O$ Mole Fraction vs Time')
ax10.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 14})


ax11.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CH3')]), 'r-')
ax11.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CH3')]), 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax11.plot(ad_times[ad_start:ad_end], ad_CH3[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax11.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH3[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax11.set(xlabel='Time (s)', ylabel=r'$CH_3$ Mole Fraction', title=r'$CH_3$ Mole Fraction vs Time')
ax11.legend(['Detailed',  'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 14})


ax12.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CH')]), 'r-')
if 'CH' in const_spec_strs:
    ax12.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CH')]), 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax12.plot(ad_times[ad_start:ad_end], ad_CH[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax12.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax12.set(xlabel='Time (s)', ylabel=r'$CH$ Mole Fraction', title=r'$CH$ Mole Fraction vs Time')
ax12.legend(['Detailed', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 12})


ax13.plot(auto_ign_det['axis0'][det_start:det_end], list(auto_ign_det['heat_release_rate'])[det_start:det_end], 'r-')
ax13.plot(auto_ign_test['axis0'][const_start:const_end], list(auto_ign_test['heat_release_rate'])[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax13.plot(ad_times[ad_start:ad_end], ad_HRR[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax13.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_HRR[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax13.set(xlabel='Time (s)', ylabel=r'Heat Release Rate ($J/m^3*s$)', title='Heat Release Rate vs Time')
ax13.set_yscale('log')
ax13.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 14})
f.delaxes(ax5)
f.show()

plt.rcParams["figure.figsize"] = (10,8)
g, (ax3, ax4) = plt.subplots(2, 1)

ax3.plot(t_steps, n_ad_species, 'bo-')
ax3.plot(t_steps, [const_n_spec]*len(t_steps), 'go--')
ax3.plot(t_pred_steps, n_ad_pred_species, 'mo-')
ax3.set(xlabel='Time (s)', ylabel='Number of Species', title='Number of Species Used vs. Time')
ax3.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': 14})

ax4.plot(t_steps, n_ad_reactions, 'bo-')
ax4.plot(t_steps, [const_n_rxn]*len(t_steps), 'go--')
ax4.plot(t_pred_steps, n_ad_pred_reactions, 'mo-')
ax4.set(xlabel='Time (s)', ylabel='Number of Reactions', title='Number of Reactions Used vs. Time')
ax4.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': 14})
g.tight_layout()
g.show()

hub_data={'tot_ad_hubs':tot_ad_hubs, 'ad_hubs':ad_hubs, 'tot_const_hubs':tot_const_hubs, 'const_hubs':const_hubs, 't_steps':t_steps}

with open('results_data/t1310_e0.65_h1.1_a0.001_hubs', 'wb') as results_file:
    pickle.dump(hub_data, results_file)

hub_plot_lists={}
for hub in tot_ad_hubs:
    hub_plot_lists[hub] = []
    for ranks in ad_hubs:
        if hub in ranks:
            hub_plot_lists[hub].append(ranks.index(hub)+1)
        else:
            hub_plot_lists[hub].append(0)

verts = [
   (0., -4.),  # left, bottom
   (0., 4.),  # left, top
   (0.9, 4.),  # right, top
   (0.9, -4.),  # right, bottom
   (0., -4.),  # back to left, bottom
]

codes = [
    Path.MOVETO, #begin drawing
    Path.LINETO, #straight line
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY, #close shape. This is not required for this shape but is "good form"
]

path = Path(verts, codes)

h, (ax5) = plt.subplots(1)
for hub in tot_ad_hubs:
    ax5.scatter(list(np.array(t_steps)*1000), hub_plot_lists[hub], marker=path, s=11500)
ax5.set(xlabel='Time (ms)', ylabel='Rank', title='Ranked Hub Species vs. Time')
ax5.set_ylim([0.5, 4.5])
ax5.invert_yaxis()
ax5.set_yticks([1,2,3,4])
ax5.legend(convertChemNames(tot_ad_hubs), prop = {'size': 18}, bbox_to_anchor = (1., 1.05), markerscale = 0.15)

verts = [
   (-4, -5.),  # left, bottom
   (-4, 5.),  # left, top
   (4, 5.),  # right, top
   (4, -5.),  # right, bottom
   (-4, -5.),  # back to left, bottom
]

codes = [
    Path.MOVETO, #begin drawing
    Path.LINETO, #straight line
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY, #close shape. This is not required for this shape but is "good form"
]

c_path = Path(verts, codes)

colors = ['tab:orange', 'tab:blue', 'tab:gray', 'tab:purple']

plt.savefig(os.path.join('chem_plots', 'SLGPS_ranked.png'))

i, (ax6) = plt.subplots(1)
for i, hub in enumerate(tot_const_hubs):
    ax6.scatter(0, const_hubs.index(hub)+1, marker=c_path, s=11500, c=colors[i])
ax6.set(ylabel='Rank', xlabel='Constant Through Simulation', title='Ranked Constant Hub Species')
ax6.set_ylim([0.5, 4.5])
ax6.invert_yaxis()
ax6.set_yticks([1, 2, 3, 4])
ax6.set_xticks([])
ax6.legend(convertChemNames(tot_const_hubs), prop = {'size': 18}, markerscale = 0.1)
plt.savefig(os.path.join('chem_plots', 'classicGPS_ranked.png'))
# plt.plot(auto_ign_det['axis0'][det_start:det_end], auto_ign_det['temperature'][det_start:det_end], 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], auto_ign_test['temperature'][const_start:const_end], 'gs')
# plt.plot(ad_times[ad_start:ad_end], ad_temps[ad_start:ad_end], 'bo')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_temps[ad_pred_start:ad_pred_end], 'm^')
# plt.xlabel('Time (s)')
# plt.ylabel('Temperature (K)')
# plt.title('Temperature vs Time')
# plt.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'temp.png'))
# f.show()

# h = plt.figure(2)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CH4')]), 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CH4')]), 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_CH4[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH4[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$CH_4$ Mole Fraction')
# plt.title(r'$CH_4$ Mole Fraction vs Time')
# plt.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'CH4.png'))
# h.show()

# i = plt.figure(3)

# plt.plot(t_steps, n_ad_species, 'bo-')
# plt.plot(t_steps, [const_n_spec]*len(t_steps), 'go--')
# plt.plot(t_pred_steps, n_ad_pred_species, 'mo-')
# plt.xlabel('Time (s)')
# plt.ylabel('Number of Species')
# plt.title('Number of Species Used vs. Time')
# plt.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'spec.png'))
# i.show()

# j = plt.figure(4)
# plt.plot(t_steps, n_ad_reactions, 'bo-')
# plt.plot(t_steps, [const_n_rxn]*len(t_steps), 'go--')
# plt.plot(t_pred_steps, n_ad_pred_reactions, 'mo-')
# plt.xlabel('Time (s)')
# plt.ylabel('Number of Reactions')
# plt.title('Number of Reactions Used vs. Time')
# plt.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'rxn.png'))
# j.show()

# k = plt.figure(5)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CO2')]), 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CO2')]), 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_CO2[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CO2[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$CO_2$ Mole Fraction')
# plt.title(r'$CO_2$ Mole Fraction vs Time')
# plt.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'CO2.png'))
# k.show()

# l = plt.figure(6)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CO')]), 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CO')]), 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_CO[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CO[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$CO$ Mole Fraction')
# plt.title(r'$CO$ Mole Fraction vs Time')
# plt.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'CO.png'))
# l.show()

# m = plt.figure(7)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('OH')]), 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('OH')]), 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_OH[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_OH[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$OH$ Mole Fraction')
# plt.title(r'$OH$ Mole Fraction vs Time')
# plt.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'OH.png'))
# m.show()

# n = plt.figure(8)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('O')]), 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('O')]), 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_O[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_O[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$O$ Mole Fraction')
# plt.title(r'$O$ Mole Fraction vs Time')
# plt.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'O.png'))
# n.show()

# o = plt.figure(9)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('H')]), 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('H')]), 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_H[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_H[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$H$ Mole Fraction')
# plt.title(r'$H$ Mole Fraction vs Time')
# plt.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'H.png'))
# o.show()

# p = plt.figure(10)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('H2O')]), 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('H2O')]), 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_H2O[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_H2O[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$H_2O$ Mole Fraction')
# plt.title(r'$H_2O$ Mole Fraction vs Time')
# plt.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'H2O.png'))
# p.show()

# q = plt.figure(11)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CH3')]), 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CH3')]), 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_CH3[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH3[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$CH_3$ Mole Fraction')
# plt.title(r'$CH_3$ Mole Fraction vs Time')
# plt.legend(['Detailed',  'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'CH3.png'))
# q.show()

# r = plt.figure(12)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(np.array(auto_ign_det['mole_fraction'].tolist())[det_start:det_end, det_spec_strs.index('CH')]), 'r-')
# if 'CH' in const_spec_strs:
#     plt.plot(auto_ign_test['axis0'][const_start:const_end], list(np.array(auto_ign_test['mole_fraction'].tolist())[const_start:const_end, const_spec_strs.index('CH')]), 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_CH[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$CH$ Mole Fraction')
# plt.title(r'$CH$ Mole Fraction vs Time')
# plt.legend(['Detailed', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'CH.png'))
# r.show()

# s = plt.figure(13)

# plt.plot(auto_ign_det['axis0'][det_start:det_end], list(auto_ign_det['heat_release_rate'])[det_start:det_end], 'r-')
# plt.plot(auto_ign_test['axis0'][const_start:const_end], list(auto_ign_test['heat_release_rate'])[const_start:const_end], 'g--')
# plt.plot(ad_times[ad_start:ad_end], ad_HRR[ad_start:ad_end], 'b--')
# plt.plot(ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_HRR[ad_pred_start:ad_pred_end], 'm--')
# plt.xlabel('Time (s)')
# plt.ylabel(r'Heat Release Rate ($J/m^3*s$)')
# plt.yscale('log')
# plt.title('Heat Release Rate vs Time')
# plt.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': 10})
# plt.savefig(os.path.join('chem_plots', 'HRR.png'))
# s.show()

# plt.input()
