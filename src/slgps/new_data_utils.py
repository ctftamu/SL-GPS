# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:01:56 2022

@author: agnta
"""
import cantera as ct
from cantera import ck2yaml
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from src.ck.def_cheminp import skeletal
from src.ct.def_ct_tools import soln2raw
from src.ct.def_ct_tools import save_raw_npz
from src.core.def_GPS import GPS_algo
from src.core.def_build_graph import build_flux_graph
from tensorflow.keras.models import load_model
from pickle import load
#from drgep_red_mech import drgep_reduce
#from reduce_drgep_short import reduce_drgep_short
# from psr_funcs import psr_const_mech
# from sup_new_data import findIgnInterval, GPS_spec, findTimeIndex
# from sup_new_data import auto_ign_build

def sub_mech(mech_file, species_names):
    new_species = []
    spec = ct.Species.listFromFile(mech_file)
    for sp in spec:
        if sp.name in species_names:
            new_species.append(sp)
    ref_phase = ct.Solution(thermo='ideal-gas', kinetics='gas', species=spec)
    all_reactions = ct.Reaction.listFromFile(mech_file, ref_phase)
    reactions = []

    # print('\nReactions:')
    for R in all_reactions:
        if not all(reactant in species_names for reactant in R.reactants):
            continue

        if not all(product in species_names for product in R.products):
            continue

        reactions.append(R)
        # print(R.equation)
    # print('\n')

    return ct.Solution(thermo='ideal-gas', kinetics='gas',
                   species=new_species, reactions=reactions)

def write_sk_inp(species_kept, dir_mech_de, dir_mech_sk, notes):

    species_kept = list(species_kept)
    n_sp = len(species_kept)
    print('total: '+str(n_sp)) 
    notes.append('! number of species = '+str(n_sp))
    skeletal(dir_mech_de, dir_mech_sk, species_kept, notes=notes)
    new_t=time.perf_counter()
    # ck2cti.convertMech(dir_mech_sk+'/chem.inp', thermoFile=dir_mech_sk+'/therm.dat', transportFile=dir_mech_sk+'/tran.dat', outName=dir_mech_sk+'/chem.cti')
    ck2yaml.convert_mech(dir_mech_sk+'/chem.inp', thermo_file=dir_mech_sk+'/therm.dat', transport_file=dir_mech_sk+'/tran.dat', out_name=dir_mech_sk+'/chem.yaml', permissive=True)
    count=time.perf_counter()-new_t
    f = open(os.path.join(dir_mech_sk,'ns.txt'),'w')
    f.write(str(n_sp))
    f.close()
    return(count)

def GPS_spec(soln_in, raw, t_start, t_end, alpha):
    ind_start=None
    ind_end=None
    elements = ['C', 'H', 'O']
    sources = ['CH4', 'O2']
    targets = ['CO2', 'H2O']
    species_kept = set(sources) | set(targets) | set(['N2'])
    in_rng= False
    for i in range(len(raw['axis0'])):
        if raw['axis0'][i] >= t_start and not in_rng:
            ind_start = i
            in_rng = True
        elif raw['axis0'][i] > t_end or i == len(raw['axis0'])-1 and in_rng:
            ind_end = i
            break
    if not ind_start:
        ind_start=len(raw['axis0'])-1
    if not ind_end:
        ind_end=ind_start
    print(ind_start)
    print(ind_end)
    for i_pnt in range(ind_start, ind_end, ((ind_end-ind_start)//4)+1):
        print('doing another point')
        # if i_pnt == len(list(auto_ign['temperature']))-1 or auto_ign['temperature'][i_pnt+1]-auto_ign['temperature'][i_pnt] > 50: 
        for e in elements:
            flux_graph = build_flux_graph(soln_in, raw, e, path_save='flux_graph', overwrite=True, i0=i_pnt, i1='eq', constV=False)
            for s in sources:
                for t in targets:
                    if s in flux_graph.nodes() and t in flux_graph.nodes():
                        GPS_results = GPS_algo(soln_in, flux_graph, s, t, path_save=None, K=1, alpha=alpha, beta=0.5, normal='max', iso=None, overwrite=False, raw='unknown',notes=None, gamma=None)
                        new_species = GPS_results['species'].keys()
                        species_kept |= new_species
    return species_kept

def GPS_spec_data(soln_in, raw, alpha):
    elements = ['C', 'H', 'O']
    sources = ['CH4', 'O2']
    targets = ['CO2', 'H2O']
    species_kept = set(sources) | set(targets) | set(['N2'])
    for i_pnt in range(len(raw['axis0'])):
        print('doing another point')
        # if i_pnt == len(list(auto_ign['temperature']))-1 or auto_ign['temperature'][i_pnt+1]-auto_ign['temperature'][i_pnt] > 50: 
        for e in elements:
            flux_graph = build_flux_graph(soln_in, raw, e, path_save='flux_graph', overwrite=True, i0=i_pnt, i1='eq', constV=False)
            for s in sources:
                for t in targets:
                    if s in flux_graph.nodes() and t in flux_graph.nodes():
                        GPS_results = GPS_algo(soln_in, flux_graph, s, t, path_save=None, K=1, alpha=alpha, beta=0.5, normal='max', iso=None, overwrite=False, raw='unknown',notes=None, gamma=None)
                        new_species = GPS_results['species'].keys()
                        species_kept |= new_species
    return species_kept

def drgep_spec(soln_in, raw, t_start, t_end, threshold):
    ind_start=None
    ind_end=None
    target_species = ['CH4', 'HO2', 'CO']
    in_rng= False
    for i in range(len(raw['axis0'])):
        if raw['axis0'][i] >= t_start and not in_rng:
            ind_start = i
            in_rng = True
        elif raw['axis0'][i] > t_end or i == len(raw['axis0'])-1 and in_rng:
            ind_end = i
            break
    if not ind_start:
        ind_start=len(raw['axis0'])-1
    if not ind_end:
        ind_end=ind_start
    print(ind_start)
    print(ind_end)
    Ts = []
    Ps = []
    Xs = []
    for i_pnt in range(ind_start, ind_end, ((ind_end-ind_start)//4)+1):
        print('doing another point')
        Ts.append(raw['temperature'][i_pnt])
        Ps.append(raw['pressure'][i_pnt])
        Xs.append(raw['mole_fractions'][i_pnt])
        # if i_pnt == len(list(auto_ign['temperature']))-1 or auto_ign['temperature'][i_pnt+1]-auto_ign['temperature'][i_pnt] > 50: 
    species_kept = reduce_drgep_short(soln_in, Ts, Ps, Xs, target_species, threshold, [])
    return species_kept



def GPS_sim(alpha, soln_in, sources, targets, elements, auto_ign):
    species_kept = set(sources) | set(targets) | set(['N2'])
    hub_scores = {}
    for i_pnt in range(0, len(auto_ign['axis0']), len(auto_ign['axis0'])//50):
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
                            # for hub in list(GPS_results['hubs'].keys()):
                            #     if hub not in list(hub_scores.keys()):
                            #         hub_scores[hub] = GPS_results['hubs'][hub]['score']
                            #     else:
                            #         if GPS_results['hubs'][hub]['score'] > hub_scores[hub]:
                            #             hub_scores[hub] = GPS_results['hubs'][hub]['score']  
    # write_sk_inp(species_kept, 'de_mech', 'sk_mech', notes=[])
    # sk_mech = ct.Solution(os.path.join('sk_mech', 'chem.yaml'))
    sk_mech = sub_mech('gri30.yaml', species_kept)
    return sk_mech

def DRGEP_sim(threshold, soln_in, auto_ign):
    target_species = ['CH4', 'HO2', 'CO']
    Ts = []
    Ps = []
    Xs = []
    for i_pnt in range(0, len(auto_ign['axis0']), (len(auto_ign['axis0'])//4)+1):
        Ts.append(auto_ign['temperature'][i_pnt])
        Ps.append(auto_ign['pressure'][i_pnt])
        Xs.append([auto_ign['mole_fraction'][i_pnt]])
    drg_start = time.process_time()
    species_kept = reduce_drgep_short(soln_in, Ts, Ps, Xs, target_species, threshold, [])
    time_drg = time.process_time()-drg_start
    sk_mech = sub_mech('gri30.yaml', species_kept)
    return sk_mech, time_drg

def findIgnInterval(hrr, threshold):
    ign_count = 0
    ss_count = 0
    ign = False
    ign_end = len(hrr)-1
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
    return ign_start, ign_end

def find_ign_delay(times, temperature):
    thresh = (0.5*(max(temperature)-min(temperature)))
    # thresh = 400
    for i in range(len(times)):
        # print(temperature[i]-temperature[0])
        # print(thresh)
        if temperature[i]-temperature[0]>thresh:
            return times[i]
        # if thresh<20:
        #     raise Exception('Threshold too low')
        # thresh-=10

def findTimeIndex(times, time):
    for i in range(len(times)):
        if times[i] > time:
            return i


def auto_ign_build_new(soln, T0, atm, X0, end_threshold=2e3, end=5, dir_raw = None):
    raw_all = None
    soln.TPX = T0, atm*101400, X0
    reactor = ct.IdealGasConstPressureReactor(soln)
    network = ct.ReactorNet([reactor])
    t = 0.0
    t_prev = 0.0
    ign = False
    time1 = 0
    step_count = 0
    step_lens = [0]
    start1 = time.process_time()
    while t <= end:
        start1 = time.process_time()
        t = network.step()
        time1+=time.process_time()-start1
        step_lens.append(step_lens[-1]+(t-t_prev))
        t_prev=t
        step_count+=1
        raw_all = soln2raw(t, 'time', soln, raw_all)
        # network.advance(t)
        if raw_all['heat_release_rate'][-1] > end_threshold and not ign:
            ign=True
        elif raw_all['heat_release_rate'][-1] < end_threshold and ign:
            break
    raw = save_raw_npz(raw_all, 'ign')
    return raw, time1, step_count

def auto_ign_build_data(soln, T0, pres, X0, steps, dir_raw = None):
    raw_all = None
    soln.TPX = T0, pres, X0
    reactor = ct.IdealGasConstPressureReactor(soln)
    network = ct.ReactorNet([reactor])
    t = 0.0
    t_prev = 0.0
    ign = False
    # for i in range(steps):
        # raw_all = soln2raw(t, 'time', soln, raw_all)
        # t = network.step()
        # t_prev=t
    raw_all = soln2raw(t, 'time', soln, raw_all)
    raw = save_raw_npz(raw_all, 'ign')
    return raw

def auto_ign_build_sup(soln_in, tracked_specs, norm_Dt, 
                 ign_Dt, T0_in, X0_in, atm, t_end, scaler_loc, model_loc, data_loc, 
                 ign_threshold):
    spec_counts = []
    time_ann = 0
    time_step = 0
    step_count = 0
    step_lens = []
    var_specs = pd.read_csv(os.path.join(data_loc, 'var_spec_nums.csv')).columns.to_list()[:-1]
    always_specs = pd.read_csv(os.path.join(data_loc, 'always_spec_nums.csv')).columns.to_list()[:-1]
    if var_specs[0][:2] == '# ':
        var_specs[0] = var_specs[0][2:]
    if always_specs[0][:2] == '# ':
        always_specs[0] = always_specs[0][2:]   
    model=load_model(model_loc)
    
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
    
    t_pred_steps = []
    n_ad_pred_reactions = []
    n_ad_pred_species = []
    tick = 0.0
    Dt = norm_Dt
    ign = False
    ign_count = 0
    j=0
    
    count = 0
    pred_specs = []
    time0=time.perf_counter()
    time02=time.process_time()
    write_time=0
    write_time2=0
    T0=T0_in
    X0=X0_in
    t_count=0
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
        soln_in.TPX= T0, atm, X0
        pred_input = [T0, atm]
        for spec in tracked_specs:
            if spec in soln_in.mole_fraction_dict().keys():
                pred_input.append(soln_in.mole_fraction_dict()[spec])
            else:
                pred_input.append(0.0)
        pred_input = np.array([pred_input])
        min_max_scaler = load(open(scaler_loc, 'rb'))
        pred_input_proc = min_max_scaler.transform(pred_input)
        ann_start = time.process_time()
        prediction = model.predict(pred_input_proc)
        time_ann += time.process_time()-ann_start
        
        pred_kept_specs = always_specs.copy()
        for k in range(len(prediction[0, :])):
            if prediction[0, k] > 0.5:
                pred_kept_specs.append(var_specs[k])
                
        spec_counts.append(pred_kept_specs)
        # if tick==0.0:
        #     pred_kept_specs.append('CH2OH')
        #     pred_kept_specs.append('C2H6')
        #     pred_kept_specs.append('CH3OH')
        # if count < len(adapt_all_specs):    
        #     pred_kept_specs = adapt_all_specs[count]
        # if 'c_'+str(n_t_cases)+'_l_'+n_layers+dt_name not in max_specs.keys():
        #     max_specs['c_'+str(n_t_cases)+'_l_'+n_layers+dt_name] = n_cases*[-1]

        # if len(pred_kept_specs) > max_specs['c_'+str(n_t_cases)+'_l_'+n_layers+dt_name][i]:
        #     max_specs['c_'+str(n_t_cases)+'_l_'+n_layers+dt_name][i] = len(pred_kept_specs)
        
        
        time00=time.perf_counter()
        time002=time.process_time()
        # t_count+=write_sk_inp(pred_kept_specs, 'de_mech', '.', notes=[])
        # sk_mech = ct.Solution(os.path.join('.', 'chem.yaml'))
        sk_mech = sub_mech('gri30.yaml', pred_kept_specs)
        write_time += time.perf_counter()-time00
        write_time2 += time.process_time()-time002
        
        
        n=0
        if j > 0.5:
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
    
            # t0 = time.perf_counter()
            # if count > 0.1:
            #     Dt = ign_Dt
        while True:
            auto_ign_temp, time_meas, temp_count = auto_ign_build_new(sk_mech, T0, atm, X0, -100, end=Dt, dir_raw = 'ign')  
            time_step +=time_meas
            step_count +=temp_count
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
                
        # auto_ign_temp = auto_ign_build_new(sk_mech, T0, atm, X0, -100, end=Dt, dir_raw = 'ign')  
        # time_sup +=time_meas
        # step_count += temp_count
        # step_lens += step_lens_temp
        # step_lens.append(count)
        # if auto_ign_temp['heat_release_rate'][-1] >= ign_threshold and not ign:
        #     Dt = ign_Dt
        #     ign = True
        # elif auto_ign_temp['heat_release_rate'][-1] < ign_threshold and ign:
        #     Dt = norm_Dt
        #     ign = False
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
        auto_ign_temp_HRR = list(auto_ign_temp['heat_release_rate'][:])
        X0 = ','.join(x_labels)    
        for k, temp in enumerate(auto_ign_temp_t):
            # for l, d_time in enumerate(auto_ign_det['axis0'][:]):
            #     if d_time > tick + auto_ign_temp['axis0'][k] or l == len(auto_ign_det['axis0'])-1:
            #         det_ind = l
                    # break
            ad_pred_times.append(tick + auto_ign_temp_time[k])
            ad_pred_temps.append(temp)
            ad_pred_CH4.append(auto_ign_temp_CH4[k])
            ad_pred_CH2O.append(auto_ign_temp_CH2O[k])
            ad_pred_CO2.append(auto_ign_temp_CO2[k])
            ad_pred_CO.append(auto_ign_temp_CO[k])
            ad_pred_OH.append(auto_ign_temp_OH[k])
            ad_pred_O.append(auto_ign_temp_O[k])
            ad_pred_H.append(auto_ign_temp_H[k])
            ad_pred_H2O.append(auto_ign_temp_H2O[k])
            ad_pred_CH3.append(auto_ign_temp_CH3[k])
            ad_pred_CH.append(auto_ign_temp_CH[k])
            ad_pred_HRR.append(auto_ign_temp_HRR[k])

        T0 = auto_ign_temp_t[-1]
        tick = ad_pred_times[-1]
        j+=1
        count += 1
        
        
        
    result_dict={'axis0': ad_pred_times, 'temperature': ad_pred_temps, 'heat_release_rate':ad_pred_HRR}
    return result_dict, time_step, time_ann, step_count, spec_counts

def auto_ign_build_ad(soln_in, norm_Dt, ign_Dt, t_end, T0,  X0, atm, ign_threshold, drg_threshold):
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
    
    
    spec_counts = []
    time_drg = 0
    time_comp_drg = 0
    time_step = 0
    step_count = 0
    t_steps = []
    n_ad_reactions = []
    n_ad_species = []
    tick = 0.0
    Dt = norm_Dt
    ign = False
    ign_count = 0
    i=0
    
    count = 0
    pred_specs = []
    time0=time.perf_counter()
    time02=time.process_time()
    write_time=0
    write_time2=0
    t_count=0
    
    
    
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
            auto_ign, time_meas_n, temp_count_n = auto_ign_build_new(soln_in, T0, atm, X0, -100, end=Dt, dir_raw = 'ign') 
            sk_mech, step_drg = DRGEP_sim(drg_threshold, soln_in, auto_ign)
            time_comp_drg += step_drg
            
            spec_counts.append(sk_mech.species_names)
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
            auto_ign_temp, time_meas, temp_count = auto_ign_build_new(sk_mech, T0, atm, X0, -100, end=Dt, dir_raw = 'ign')  
            step_count +=temp_count
            time_step +=time_meas
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
        time_drg += step_drg
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
        auto_ign_temp_HRR = list(auto_ign_temp['heat_release_rate'][:])
        X0 = ','.join(x_labels)    
        for j, temp in enumerate(auto_ign_temp_t):
            # for k, d_time in enumerate(auto_ign_det['axis0'][:]):
            #     if d_time > tick + auto_ign_temp['axis0'][j] or k == len(auto_ign_det['axis0'])-1:
            #         det_ind = k
            #         break
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
        # sorted_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)
        # real_sorted_hubs = []
        # for hub in sorted_hubs:
        #     real_sorted_hubs.append(hub[0])
        # for hub in real_sorted_hubs[:4]:
        #     if hub not in tot_ad_hubs:
        #         tot_ad_hubs.append(hub)
        # ad_hubs.append(real_sorted_hubs[:4])
        T0 = auto_ign_temp_t[-1]
        tick = ad_times[-1]
        i+=1

    result_dict={'axis0': ad_times, 'temperature': ad_temps, 'heat_release_rate':ad_HRR}
    return result_dict, time_step, time_drg, time_comp_drg, step_count, spec_counts

def auto_ign_build_const(soln_in, T0, atm, X0, alpha, end_threshold, t_end, sources, targets, elements, auto_ign):
    sk_mech = GPS_sim(alpha, soln_in, sources, targets, elements, auto_ign)
    spec_count= sk_mech.n_species
    reac_count = sk_mech.n_reactions
    auto_ign_const, time1, step_count = auto_ign_build_new(sk_mech, T0, atm, X0, end_threshold, t_end)
    return auto_ign_const, time1, step_count, spec_count, reac_count

def auto_ign_build_drgep(mech_file, T0, atm, X0, phi, fuel, oxid, error_limit, end_threshold, t_end, safe_specs):
    sk_mech = drgep_reduce(mech_file, T0, atm, phi, fuel, oxid, error_limit, spec_safe=safe_specs)
    spec_count= sk_mech.n_species
    reac_count = sk_mech.n_reactions
    auto_ign, time1, step_count = auto_ign_build_new(sk_mech, T0, atm, X0, end_threshold, t_end)
    return auto_ign, time1, step_count, spec_count, reac_count

# soln =ct.Solution('gri30.cti')
# auto_ign_det = auto_ign_build(soln=soln, T0=1300, atm=10, X0={'CH4': 1.6, 'O2': 2.0, 'N2':7.52}, end_threshold=2e5)
# plt.plot(auto_ign_det['axis0'], auto_ign_det['temperature'])
