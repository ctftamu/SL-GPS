


import cantera as ct
import pandas as pd
import time
import numpy as np
import os
from GPS.src.ct.def_ct_tools import soln2raw
from GPS.src.ct.def_ct_tools import save_raw_npz
from GPS.src.core.def_GPS import GPS_algo
from GPS.src.core.def_build_graph import build_flux_graph
from tensorflow.keras.models import load_model
from pickle import load



#returns new Cantera solution using reduced set of species
def sub_mech(mech_file, species_names):
    
    new_species = []
    spec = ct.Species.listFromFile(mech_file)
    for sp in spec:
        if sp.name in species_names:
            new_species.append(sp)
    ref_phase = ct.Solution(thermo='ideal-gas', kinetics='gas', species=spec)
    all_reactions = ct.Reaction.listFromFile(mech_file, ref_phase)
    reactions = []


    #remove all reactions involving excluded species
    for R in all_reactions:
        if not all(reactant in species_names for reactant in R.reactants):
            continue

        if not all(product in species_names for product in R.products):
            continue

        reactions.append(R)

    return ct.Solution(thermo='ideal-gas', kinetics='gas',
                   species=new_species, reactions=reactions)


#returns list of reduced species returned by GPS run on an interval of autoignition data
def GPS_spec(soln_in, fuel, raw, t_start, t_end, alpha, GPS_per_interval):
    ind_start=None
    ind_end=None
    elements = ['C', 'H', 'O']
    sources = [fuel, 'O2']
    targets = ['CO2', 'H2O']
    species_kept = set(sources) | set(targets) | set(['N2'])
    in_rng= False
    
    
    #get index range of raw data from time range
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
        
    #run GPS on GPS_per_interval data points across interval
    for i_pnt in range(ind_start, ind_end, ((ind_end-ind_start)//GPS_per_interval)+1):
        #build flux graph for each element
        for e in elements:
            flux_graph = build_flux_graph(soln_in, raw, e, path_save='flux_graph', overwrite=True, i0=i_pnt, i1='eq', constV=False)
            #find species important to each global pathway from each source to each target
            for s in sources:
                for t in targets:
                    if s in flux_graph.nodes() and t in flux_graph.nodes():
                        GPS_results = GPS_algo(soln_in, flux_graph, s, t, path_save=None, K=1, alpha=alpha, beta=0.5, normal='max', iso=None, overwrite=False, raw='unknown',notes=None, gamma=None)
                        new_species = GPS_results['species'].keys()
                        #include union of all selected species in reduced mechanism
                        species_kept |= new_species
    return species_kept



#find the index range of ignition given heat release rate data and a threshold
def findIgnInterval(hrr, threshold):
    ign_start = 0
    ign_end = len(hrr)-1
    ign = False
    for i in range(len(hrr)):
        if hrr[i] > threshold and not ign:
            ign_start = i
            ign = True
        elif hrr[i] < threshold and ign:
            ign_end = i
            ign = False
    return ign_start, ign_end



#find ignition delay from temperature data
def find_ign_delay(times, temperature):
    thresh = (0.5*(max(temperature)-min(temperature)))
    for i in range(len(times)):
        if temperature[i]-temperature[0]>thresh:
            return times[i]



#find nearest index for a time in a list of times
def findTimeIndex(times, time):
    for i in range(len(times)):
        if times[i] > time:
            return i



#run auto_ignition simulation for given initial conditions (incl. mole fractions X0) until heat release rate drops below threshold
def auto_ign_build_X0(soln, T0, atm, X0, end_threshold=2e3, end=5, dir_raw = None):

    raw_all = None
    soln.TPX = T0, atm*101400, X0
    reactor = ct.IdealGasConstPressureReactor(soln)
    network = ct.ReactorNet([reactor])
    t = 0
    ign = False
    time_exec = 0
    step_count = 0
    
    #begin time loop for simulation
    while t <= end:
        start_exec = time.process_time()
        t = network.step()
        
        time_exec+=time.process_time()-start_exec
        step_count+=1
        raw_all = soln2raw(t, 'time', soln, raw_all)
        
        
        #end simulation once hrr exceeds end threshold then drops below again
        if end_threshold:
            if raw_all['heat_release_rate'][-1] > end_threshold and not ign:
                ign=True
            elif raw_all['heat_release_rate'][-1] < end_threshold and ign:
                break
    raw = save_raw_npz(raw_all, 'ign')
    
    #return results, execution time, and number of timesteps
    return raw, time_exec, step_count

#run auto_ignition simulation for given initial conditions (incl. equiv. ratio phi) until heat release rate drops below threshold
def auto_ign_build_phi(soln, T0, atm, phi, fuel, end_threshold=2e3, end=5, dir_raw = None):
    raw_all = None
    soln.TP = T0, atm*101400
    soln.set_equivalence_ratio(phi, fuel+':1.0', 'O2:1.0, N2:3.76', basis='mole')
    reactor = ct.IdealGasConstPressureReactor(soln)
    network = ct.ReactorNet([reactor])
    t = 0
    ign = False
    time_exec = 0
    step_count = 0
    
    #begin time loop for simulation
    while t <= end:
        start_exec = time.process_time()
        t = network.step()
        time_exec+=time.process_time()-start_exec
        step_count+=1
        raw_all = soln2raw(t, 'time', soln, raw_all)


        #end simulation once hrr exceeds end threshold then drops below again
        if end_threshold:
            if raw_all['heat_release_rate'][-1] > end_threshold and not ign:
                ign=True
            elif raw_all['heat_release_rate'][-1] < end_threshold and ign:
                break
    raw = save_raw_npz(raw_all, 'ign')
    
    #return results, execution time, and number of timesteps
    return raw, time_exec, step_count


#run simulation using SL_GPS with trained ANN
def auto_ign_build_SL(fuel, mech_file, input_specs, norm_Dt, 
                 ign_Dt, T0_in, phi, atm, t_end, scaler_loc, model_loc, data_loc, 
                 ign_threshold):
    soln_in = ct.Solution(mech_file)
    det_spec_lst = soln_in.species_names.copy()
    det_rxn_lst = soln_in.reaction_equations().copy()
    
    time_chem = 0
    step_count = 0
    time_ann = 0
    spec_list = []
    mech_times = []
    
    #load data for 'variable' and 'always' species
    var_specs = pd.read_csv(os.path.join(data_loc, 'var_spec_nums.csv')).columns.to_list()[:-1]
    always_specs = pd.read_csv(os.path.join(data_loc, 'always_spec_nums.csv')).columns.to_list()[:-1]
    if var_specs[0][:2] == '# ':
        var_specs[0] = var_specs[0][2:]
    if always_specs[0][:2] == '# ':
        always_specs[0] = always_specs[0][2:]   
    
    #load ann from h5 file
    model=load_model(model_loc)
    
    
    
    n_ad_pred_reactions = []
    n_ad_pred_species = []
    tick = 0.0
    Dt = norm_Dt
    ign = False
    ign_count = 0
    j=0
    
    count = 0
    T0=T0_in
    soln_in.TP = T0, atm*101325
    soln_in.set_equivalence_ratio(phi, fuel+':1.0', 'O2:1.0, N2:3.76')
    spec_list = soln_in.species_names.copy()
    X0 = dict(zip(spec_list.copy(), soln_in.X.copy()))
    
    
    #begin loop of running ANN to reduce mechanism for each interval
    while tick < t_end:
        mech_times.append(tick)
        check = 0
        
        #set up state vector to input to ANN
        pred_input = [T0, atm]
        for spec in input_specs:
            if spec in soln_in.mole_fraction_dict().keys():
                pred_input.append(soln_in.mole_fraction_dict()[spec])
            else:
                pred_input.append(0.0)
        pred_input = np.array([pred_input])
        
        #normalize input data
        min_max_scaler = load(open(scaler_loc, 'rb'))
        pred_input_proc = min_max_scaler.transform(pred_input)
        
        #produce predictions from ANN and track associated execution time
        ann_start = time.process_time()
        prediction = model.predict(pred_input_proc)
        time_ann += time.process_time()-ann_start
        
        #produce reduced mechanism based on ANN output
        pred_kept_specs = always_specs.copy()
        for k in range(len(prediction[0, :])):
            if prediction[0, k] > 0.5:
                pred_kept_specs.append(var_specs[k])
        sk_mech = sub_mech(mech_file, pred_kept_specs)
        rxn_list = sk_mech.reaction_equations()
        
        #ensure initial mole fractions do not include species that were excluded from reduced mechanism
        new_spec_list = sk_mech.species_names
        x_remove = []
        for spec in spec_list:
            if spec not in new_spec_list:
                x_remove.append(spec)
        for spec in new_spec_list:
            if spec not in spec_list:
                X0[spec] = 0.0
        for x in x_remove:
            X0.pop(x)
        spec_list = new_spec_list.copy()
        
        #determine GPS interval based on ignition HRR threshold
        while True:
            auto_ign_temp, time_meas, temp_count = auto_ign_build_X0(sk_mech, T0, atm, X0, None, end=Dt, dir_raw = 'ign')  
            time_chem +=time_meas
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
                
        #append results of currect interval to overall simulation results        
        if tick == 0.0:
            auto_ign_complete = dict.fromkeys(auto_ign_temp.keys())
            for key in auto_ign_complete:
                if key == 'mole_fraction':
                    auto_ign_complete[key] = np.zeros((np.shape(auto_ign_temp['mole_fraction'])[0], len(det_spec_lst)))
                    for spec in det_spec_lst:
                        if spec in spec_list:
                            auto_ign_complete[key][:, det_spec_lst.index(spec)] = np.array(auto_ign_temp['mole_fraction'])[:, spec_list.index(spec)]
                elif key == 'net_reaction_rate':
                    auto_ign_complete[key] = np.zeros((np.shape(auto_ign_temp['net_reaction_rate'])[0], len(det_rxn_lst)))
                    for rxn in det_rxn_lst:
                        if rxn in rxn_list:
                            auto_ign_complete[key][:, det_rxn_lst.index(rxn)] = np.array(auto_ign_temp['net_reaction_rate'])[:, rxn_list.index(rxn)]
                elif str(type(auto_ign_temp[key])) == '<class \'list\'>':
                    auto_ign_complete[key] = auto_ign_temp[key]
                elif str(type(auto_ign_temp[key])) == '<class \'str\'>':
                    auto_ign_complete[key] = [auto_ign_temp[key]]
        else:
            for key in auto_ign_complete:
                if key == 'mole_fraction':
                    new_array = np.zeros((np.shape(auto_ign_temp['mole_fraction'])[0], len(det_spec_lst)))
                    for spec in det_spec_lst:
                        if spec in spec_list:
                            new_array[:, det_spec_lst.index(spec)] = np.array(auto_ign_temp['mole_fraction'])[:, spec_list.index(spec)]
                    auto_ign_complete[key] = np.append(auto_ign_complete[key], new_array, axis=0)
                    
                elif key == 'net_reaction_rate':
                    new_array = np.zeros((np.shape(auto_ign_temp['net_reaction_rate'])[0], len(det_rxn_lst)))
                    for rxn in det_rxn_lst:
                        if rxn in rxn_list:
                            new_array[:, det_rxn_lst.index(rxn)] = np.array(auto_ign_temp['net_reaction_rate'])[:, rxn_list.index(rxn)]
                    auto_ign_complete[key] = np.append(auto_ign_complete[key], new_array, axis=0)
                elif str(type(auto_ign_temp[key])) == '<class \'list\'>':
                    if key == 'axis0':
                        for i in range(len(auto_ign_temp[key])):
                            auto_ign_complete[key].append(auto_ign_temp[key][i]+tick)
                    else:
                        for i in range(len(auto_ign_temp[key])):
                            auto_ign_complete[key].append(auto_ign_temp[key][i])
                elif str(type(auto_ign_temp[key])) == '<class \'str\'>':
                    auto_ign_complete[key].append(auto_ign_temp[key])
                else:
                    for i in range(len(auto_ign_temp[key].tolist())):
                        auto_ign_complete[key].append(auto_ign_temp[key].tolist()[i])
            
        #track number of species and reactions included in each interval
        n_ad_pred_reactions.append(len(list(sk_mech.reactions())))
        n_ad_pred_species.append(len(list(sk_mech.species())))
        
        #set up next initial conditions with final data from current interval
        X0 = dict(zip(spec_list, auto_ign_temp['mole_fraction'][-1].tolist()[0]))
        T0 = auto_ign_temp['temperature'][-1]
        tick = auto_ign_complete['axis0'][-1]
        j+=1
        count += 1
    
        
        
    result_dict = auto_ign_complete
    result_dict['mechanism'] = mech_file
    print('Simulation completed with ' + "{0:.4g}".format(time_chem) + ' seconds spent on solving chemistry and ' + "{0:.4g}".format(time_ann) + ' seconds spent on evaluating the ANN')
    
    return result_dict, mech_times, n_ad_pred_reactions, n_ad_pred_species