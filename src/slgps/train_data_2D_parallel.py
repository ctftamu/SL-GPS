# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 16:20:45 2022

@author: agnta
"""
import cantera as ct
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from new_data_utils import findIgnInterval, GPS_spec_data, findTimeIndex, auto_ign_build_data, auto_ign_build_new
import os
import pandas
import multiprocessing

# def rand_train_cases(mech_file, n_cases, t_rng, p_rng, phi_rng, alpha):
    
    

def process_simulation(sim_data):
  try:    

    T0_in, mol_fracs_in, det_spec_strs, mech_file, alpha, i, j = sim_data

    print('Simulation #:', i, j);

    # Initialize arrays
    data_array = []
    bin_array = []
     
    # Initialize lists to store data
    bin_list = []
    data_list = []

#    sim_count += 1
#    count += 1
#    print('Performing Simulation: '+str(sim_count))
    T0 = T0_in
    pres = 1.0
    X0 = mol_fracs_in
    
    soln_in = ct.Solution(mech_file)
    
 
    
    auto_ign_det = auto_ign_build_data(soln_in, T0, pres, X0, 0, dir_raw = 'ign')
    spec_strs = GPS_spec_data(soln_in, auto_ign_det, alpha = alpha)
    
    bin_spec_list = np.zeros((1, len(det_spec_strs)), dtype=int)
    for spec in spec_strs:
        if spec in det_spec_strs:
            bin_spec_list[0, det_spec_strs.index(spec)] = 1
    print('Total Species: '+str(sum(sum(bin_spec_list))))

 
 #   if count == 1:
 #       bin_array = bin_spec_list
 #       data_array = np.array([[auto_ign_det['temperature'][0], pres, auto_ign_det['mole_fraction'].tolist()[0]]])
 #   else:
 #       bin_array = np.append(bin_array, bin_spec_list, axis=0)
 #       data_array = np.append(data_array, [[auto_ign_det['temperature'][0], pres, auto_ign_det['mole_fraction'].tolist()[0]]], axis=0)

 

 
    # Append species and state vector data to the data_list
    temperature = auto_ign_det['temperature'][0]
    mole_fraction = auto_ign_det['mole_fraction'].tolist()[0]
    data_list.append([temperature, pres, mole_fraction])
        #print('time:', t)

    # Convert the lists to NumPy arrays
    bin_array = np.array(bin_list)
    data_array = np.array(data_list, dtype=object)
   

    # Reformat array of state vectors
    data_proc = np.zeros((np.shape(data_array)[0], 2 + len(det_spec_strs)))
    for i in range(np.shape(data_array)[0]):
     for j in range(2):
        data_proc[i, j] = data_array[i, j]
        for k in range(len(det_spec_strs)):
           data_proc[i, 2+k] = data_array[i, 2][det_spec_strs.index(det_spec_strs[k])]
            # for i in range(np.shape(data_proc)[1]):
                #     data_proc[:, i] -= min(data_proc[:, i])
                #     data_proc[:, i] /= (max(data_proc[:, i]) - min(data_proc[:, i]))

    return data_proc, bin_array
  except Exception as e:
        # Print the traceback in case of an exception
    traceback.print_exc()
    return None





    
mech_file='gri30.cti'
alpha=0.00000000000075
    
# temp = np.array(pandas.read_csv('reactingFOAM_TDAC_Temp.csv'))[:, 1]
# pressures = np.array(pandas.read_csv('reactingFOAM_TDAC_Pres.csv'))[:, 1]
# conc = np.array(pandas.read_csv('reactingFOAM_TDAC_Conc.csv'))[:, 1:54]
extra_H2O = 0.0090 #mass fraction
extra_CO2 = 0.0600 #mass_fraction
soln_in = ct.Solution(mech_file)
CH4_W = soln_in.molecular_weights[soln_in.species_names.index('CH4')]
O2_W = soln_in.molecular_weights[soln_in.species_names.index('O2')]
N2_W = soln_in.molecular_weights[soln_in.species_names.index('N2')]
CO2_W = soln_in.molecular_weights[soln_in.species_names.index('CO2')]
H2O_W = soln_in.molecular_weights[soln_in.species_names.index('H2O')]

O2_N = 2.0
N2_N = 7.52


n_mol_fracs = 3 #300
n_temps = 2 #20
soln_in = ct.Solution(mech_file)
soln2 = ct.Solution(mech_file)
T0 = 1500
pres = 1.0
end_threshold = 2e5
t_end=0.005
phis = np.linspace(0.1, 3.2, 6)
# phis = [1.0]
mol_fracs = np.zeros((n_mol_fracs*len(phis)*2, 53))
k=0
for extra_prod in [True, False]:
    for phi in phis:
        if extra_prod:
            prod_fracs = np.linalg.solve(np.array([[1.0-extra_CO2, -extra_CO2*H2O_W/CO2_W], [-extra_H2O*CO2_W/H2O_W, 1.0-extra_H2O]]), np.array([[extra_CO2*(phi*CH4_W+O2_N*O2_W+N2_N*N2_W)/CO2_W], [extra_H2O*(phi*CH4_W+O2_N*O2_W+N2_N*N2_W)/H2O_W]]))
            X0 = {'CO2':prod_fracs[0], 'H2O':prod_fracs[1], 'CH4':phi, 'O2':2, 'N2':7.52}
            # soln2.X = X0
            # print(soln2.Y[soln_in.species_names.index('H2O')])
        else:
            X0 = {'CH4':phi, 'O2':2, 'N2':7.52}
            soln2.X = X0
        auto_ign_det_first = auto_ign_build_new(soln_in, T0, pres, X0, end_threshold, t_end, dir_raw = 'ign')[0]
        # plt.plot(auto_ign_det_first['axis0'], auto_ign_det_first['temperature'])
        # plt.legend(['0.1', '0.7', '1.3', '1.9', '2.5', '3.1', '1.0'])
        for i in range(n_mol_fracs):
            for j in range(53):
                mol_fracs[i+n_mol_fracs*k, j] = auto_ign_det_first['mole_fraction'][i*(len(auto_ign_det_first['axis0'])//n_mol_fracs), j]
        k+=1
temp = np.linspace(290, 2050, n_temps)
# print(max(mol_fracs[:, 5]))
# print(max(mol_fracs[:, 15]))
# print(min(auto_ign_det_first['heat_release_rate']))

# inputs = [13, 5, 4, 1, 14, 3, 15, 2, 12, 9]




soln_in = ct.Solution(mech_file)
count=0
sim_count = 0
ign = False
det_spec_list = []
for spec in soln_in.species():
    det_spec_list.append(spec)
    det_spec_strs = []  
for spec in det_spec_list:
    det_spec_strs.append(str(spec)[9:-1])


# Create a list of simulation data
#sim_data_list = [(temps[i], phis[i], pressures[i], 5.0, ign_GPS_resolution, norm_GPS_resolution, ign_HRR_threshold_div, det_spec_strs, mech_file, fuel, end_threshold, alpha, GPS_per_interval,i) for i in range(n_cases)] # add second loop



sim_data_list = [
    (temp[i], mol_fracs[j], det_spec_strs, mech_file, alpha, i, j)
    for i in range(len(temp))
    for j in range(n_mol_fracs*len(phis)*2)
]



num_processors = multiprocessing.cpu_count()
print('Number of processors:', num_processors)


    # Create a multiprocessing pool with the number of CPU cores
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Map the process_simulation function to the sim_data_list
results = pool.map(process_simulation, sim_data_list)

    # Close the pool and wait for the work to finish
pool.close()
pool.join()

concatenated_data_proc = None
concatenated_bin_array = None

# Process the results
for result in results:
    data_proc, bin_array = result
    # Do further processing with the data_proc and bin_array


    if concatenated_data_proc is None:
   	        concatenated_data_proc = data_proc
    else:
            concatenated_data_proc = np.concatenate((concatenated_data_proc, data_proc), axis=0)

    if concatenated_bin_array is None:
            concatenated_bin_array = bin_array
    else:
           	concatenated_bin_array = np.concatenate((concatenated_bin_array, bin_array), axis=0)
concatenated_bin_array = np.squeeze(concatenated_bin_array)
#print(concatenated_bin_array)            
#Identify species included above and below frequency thresholds to designate
#"always" and "never" species, to reduce complexity of output layer of ANN
always_arr = np.array([], dtype=int)
never_arr = np.array([], dtype=int)
i=0


print('shape of array:',np.shape(concatenated_bin_array))
for i in range(np.shape(concatenated_bin_array)[1]):
    if sum(concatenated_bin_array[:, i]) >= np.shape(concatenated_bin_array)[0]*0.999: #always_threshold:
        always_arr = np.append(always_arr, np.array([int(i)]), 0)
    elif sum(concatenated_bin_array[:, i]) <= np.shape(concatenated_bin_array)[0]*0.001: #never_threshold:
        never_arr = np.append(never_arr, np.array([int(i)]), 0)


delete_arr = np.append(always_arr, never_arr,0)
#print(delete_arr)
new_bin_array = np.delete(concatenated_bin_array, delete_arr, 1)
#print(new_bin_array)
det_spec_nums = np.arange(0, 53, dtype=int)
det_spec_nums = np.delete(det_spec_nums, delete_arr) 



# input : temp[i], mol_fracs[j],
# output : bin_array, data_array
'''
for i in range(len(temp)):
    for j in range(n_mol_fracs*len(phis)*2):
        sim_count += 1
        count += 1
        print('Performing Simulation: '+str(sim_count))
        T0 = temp[i]
        pres = 1.0
        X0 = mol_fracs[j]
        
        auto_ign_det = auto_ign_build_data(soln_in, T0, pres, X0, 0, dir_raw = 'ign')
        spec_strs = GPS_spec_data(soln_in, auto_ign_det, alpha = alpha)
        
        bin_spec_list = np.zeros((1, len(det_spec_strs)), dtype=int)
        for spec in spec_strs:
            if spec in det_spec_strs:
                bin_spec_list[0, det_spec_strs.index(spec)] = 1
        print('Total Species: '+str(sum(sum(bin_spec_list))))
        if count == 1:
            bin_array = bin_spec_list
            data_array = np.array([[auto_ign_det['temperature'][0], pres, auto_ign_det['mole_fraction'].tolist()[0]]])
        else:
            bin_array = np.append(bin_array, bin_spec_list, axis=0)
            data_array = np.append(data_array, [[auto_ign_det['temperature'][0], pres, auto_ign_det['mole_fraction'].tolist()[0]]], axis=0)
    
'''    




    
tracked_specs = ['CH4', 'H2O', 'OH', 'H', 'CO', 'O2', 'CO2', 'O', 'CH3', 'CH']

                  

var_spec_header = ''
all_spec_header = ''
for i in range(len(det_spec_strs)):
    all_spec_header += det_spec_strs[i] + ','
    if i not in delete_arr:
        var_spec_header += det_spec_strs[i] + ','
data_header = 'Temperature,Atmospheres,'
for spec in det_spec_strs:
    data_header += str(spec) + ','
never_specs = ''
always_specs = ''
var_specs = ''
for i in range(np.shape(never_arr)[0]):
    never_specs += det_spec_strs[never_arr[i]] + ','
for i in range(np.shape(always_arr)[0]):
    always_specs += det_spec_strs[always_arr[i]] + ','
for i in range(np.shape(det_spec_nums)[0]):
    var_specs += det_spec_strs[det_spec_nums[i]] + ','
print('Success')
np.savetxt(os.path.join('train_data_EXTRA_PROD_a_0.00000000000075','never_spec_nums.csv'), np.array([]), delimiter=',', header=never_specs)
np.savetxt(os.path.join('train_data_EXTRA_PROD_a_0.00000000000075','always_spec_nums.csv'), np.array([]), delimiter=',', header=always_specs)
np.savetxt(os.path.join('train_data_EXTRA_PROD_a_0.00000000000075','var_spec_nums.csv'), np.array([]), delimiter=',', header=var_specs)
np.savetxt(os.path.join('train_data_EXTRA_PROD_a_0.00000000000075','data.csv'), data_proc, delimiter=',', header=data_header)
np.savetxt(os.path.join('train_data_EXTRA_PROD_a_0.00000000000075','species.csv'), new_bin_array, delimiter=',', header=var_spec_header)
np.savetxt(os.path.join('train_data_EXTRA_PROD_a_0.00000000000075','all_species.csv'), bin_array, delimiter=',', header=all_spec_header)
# return never_specs, always_specs, var_specs, data_proc, data_header, new_bin_array, var_spec_header
    
# never_specs, always_specs, var_specs, data_proc, data_header, new_bin_array, var_spec_header = rand_train_cases('gri30.cti', n_cases=100, t_rng=[1300,1900], p_rng=[-1, 2], phi_rng=[0.6, 1.4], alpha=0.01)
            
