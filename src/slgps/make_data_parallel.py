


import cantera as ct
import numpy as np
import random
from slgps.utils import findIgnInterval, GPS_spec, findTimeIndex, auto_ign_build_X0
import os
import multiprocessing

def process_simulation(sim_data):
  try:    

    T0, X0_value, pres, t_end, ign_GPS_resolution, norm_GPS_resolution, ign_HRR_threshold_div, det_spec_strs, mech_file, fuel, end_threshold, alpha, GPS_per_interval,i = sim_data

    print('Simulation #:', i);

    # Initialize arrays
    data_array = []
    bin_array = []
     
    # Initialize lists to store data
    bin_list = []
    data_list = []

  
    soln_in = ct.Solution(mech_file)
    X0 = X0_value #'CH4:0.689, H2:0.111, N2:0.767, O2:0.233, CO2:0.06305, H2O:0.05335, NH3:0.15'
    #X0 = {'CH4':0.689, 'H2':0.111, 'N2':0.767, 'O2':0.233, 'CO2':0.06305, 'H2O':0.05335, 'NH3':0.15}
    #soln, T0, atm, X0, end_threshold=2e3, end=5, dir_raw = None
    auto_ign_det = auto_ign_build_X0(soln_in, T0, pres, X0, end_threshold, t_end, dir_raw = 'ign')
    #Run detailed simulation with current initial conditions
 #   auto_ign_det = auto_ign_build_phi(soln_in, T0, pres, phi, fuel, end_threshold, t_end, dir_raw = 'ign')
    ign_Dt = auto_ign_det[0]['axis0'][-1]/ign_GPS_resolution
    norm_Dt = auto_ign_det[0]['axis0'][-1]/norm_GPS_resolution
    Dt = norm_Dt

    ign = False
    ign_start, ign_end = findIgnInterval(auto_ign_det[0]['heat_release_rate'], max(auto_ign_det[0]['heat_release_rate'])/ign_HRR_threshold_div)
    t = 0.0
   
    count = 0
    while t + Dt < auto_ign_det[0]['axis0'][-1]:
        count += 1

        if t + Dt > auto_ign_det[0]['axis0'][ign_start] and not ign:
            ign = True
            Dt = ign_Dt
        elif t > auto_ign_det[0]['axis0'][ign_end]:
            ign = False
            Dt = norm_Dt

        spec_strs = GPS_spec(soln_in, fuel, auto_ign_det[0], t, t+Dt, alpha, GPS_per_interval) # here 0.1 is the increased alpha for adaptive GPS
        bin_spec_list = np.zeros((1, len(det_spec_strs)), dtype=int)
        for spec in spec_strs:
            if spec in det_spec_strs:
                bin_spec_list[0, det_spec_strs.index(spec)] = 1
        
        # Append current binary list to the bin_list
        bin_list.append(bin_spec_list)

        # Append species and state vector data to the data_list
        temperature = auto_ign_det[0]['temperature'][findTimeIndex(auto_ign_det[0]['axis0'], t)]
        mole_fraction = auto_ign_det[0]['mole_fraction'].tolist()[findTimeIndex(auto_ign_det[0]['axis0'], t)]
        data_list.append([temperature, pres, mole_fraction])
        #print('time:', t)
        t += Dt

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

    return data_proc, bin_array
  except Exception as e:
        # Print the traceback in case of an exception
    traceback.print_exc()
    return None









def make_data_parallel(fuel, mech_file, end_threshold, ign_HRR_threshold_div, ign_GPS_resolution, 
              norm_GPS_resolution, GPS_per_interval, n_cases, t_rng, p_rng, phi_rng, 
              alpha, always_threshold, never_threshold, pathname):
    #initialize lists of random initial conditions
    temps = []
    pressures = []
    phis = []
    #Get random initial conditions within ranges
    for i in range(n_cases):
        #Uniform distribution
        temps.append(random.uniform(t_rng[0], t_rng[1]))
        #Log uniform distribution
        pressures.append(10**(random.uniform(p_rng[0], p_rng[1])))
        #Uniform distribution
        phis.append(random.uniform(phi_rng[0], phi_rng[1]))
    
    
    
    #Initialize Cantera solution
    soln_in = ct.Solution(mech_file)
    


    # Define the species and their allowable range
    species_ranges = {
    'CH4': (0.0, 1.0),
    'N2': (0, 0.8),
    'O2': (0.0, 0.4),
    'CO2': (0.0, 0.005),
    'H2O': (0.0,  0.1),
    'OH': (0.00, 1e-3)
    }

    # Number of cases
    #n_cases = 10  # Change this to your desired number of cases

    # Create a list to store the randomized strings
    X0_values = []

    # Generate random values for each case
    for _ in range(n_cases):
        species_values = {}

        # Generate random values within the specified range for each species
        for species, value_range in species_ranges.items():
            random_value = random.uniform(value_range[0], value_range[1])
            species_values[species] = random_value

        # Normalize the values to ensure they sum up to 1
        total_value = sum(species_values.values())
        for species in species_values.keys():
            species_values[species] /= total_value

        # Convert the values to the desired string format and append to the list
        X0 = ', '.join([f'{species}:{value:.5f}' for species, value in species_values.items()])
        X0_values.append(X0)

    # Print or use the generated values as needed
    for X0 in X0_values:
        print(X0)

# Print the generated strings for each case
#for i, X0 in enumerate(X0_values):
#    print(f'Case {i+1}: {X0}')

   
    
    #Get full mechanism species list
    det_spec_strs = []
    for spec in soln_in.species_names:
        det_spec_strs.append(spec)


       
    
    # Create a list of simulation data
    sim_data_list = [(temps[i], X0_values[i], 1, 5.0, ign_GPS_resolution, norm_GPS_resolution, ign_HRR_threshold_div, det_spec_strs, mech_file, fuel, end_threshold, alpha, GPS_per_interval,i) for i in range(n_cases)]

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
    
    for i in range(np.shape(concatenated_bin_array)[1]):
        if sum(concatenated_bin_array[:, i]) >= np.shape(concatenated_bin_array)[0]*always_threshold:
            always_arr = np.append(always_arr, np.array([int(i)]), 0)
        elif sum(concatenated_bin_array[:, i]) <= np.shape(concatenated_bin_array)[0]*never_threshold:
            never_arr = np.append(never_arr, np.array([int(i)]), 0)

  
    delete_arr = np.append(always_arr, never_arr,0)
    #print(delete_arr)
    new_bin_array = np.delete(concatenated_bin_array, delete_arr, 1)
    #print(new_bin_array)
    #Mechanism contains 128 species and 957 reactions (MILD combustion)
    det_spec_nums = np.arange(0, 53, dtype=int)
    #det_spec_nums = np.arange(0, 128, dtype=int)
    det_spec_nums = np.delete(det_spec_nums, delete_arr) 

    #Create headers for binary list csv
    var_spec_header = ''
    for i in range(len(det_spec_strs)):
        if i not in delete_arr:
            var_spec_header += det_spec_strs[i] + ','
    
    #Create headers for state vector csv
    data_header = 'Temperature,Atmospheres,'
    for spec in det_spec_strs:
        data_header += str(spec) + ','
        
    #Create lists of "always," "never," and "variable" species
    never_specs = ''
    always_specs = ''
    var_specs = ''
    for i in range(np.shape(never_arr)[0]):
        never_specs += det_spec_strs[never_arr[i]] + ','
    for i in range(np.shape(always_arr)[0]):
        always_specs += det_spec_strs[always_arr[i]] + ','
    for i in range(np.shape(det_spec_nums)[0]):
        var_specs += det_spec_strs[det_spec_nums[i]] + ','
    # Reshape new_bin_array to 2D
 #  new_bin_array_2d = new_bin_array.reshape(new_bin_array.shape[0], -1)

    # Save empty arrays with headers as CSV files if they don't exist
    if not os.path.exists(os.path.join(pathname, 'never_spec_nums.csv')):
        with open(os.path.join(pathname, 'never_spec_nums.csv'), 'w') as f:
            f.write(never_specs + '\n')
    
    if not os.path.exists(os.path.join(pathname, 'always_spec_nums.csv')):
        with open(os.path.join(pathname, 'always_spec_nums.csv'), 'w') as f:
            f.write(always_specs + '\n')
    
    if not os.path.exists(os.path.join(pathname, 'var_spec_nums.csv')):
        with open(os.path.join(pathname, 'var_spec_nums.csv'), 'w') as f:
            f.write(var_specs + '\n')
    
    # Save concatenated_data_proc with header as CSV file if it doesn't exist
    if not os.path.exists(os.path.join(pathname, 'data.csv')):
        np.savetxt(os.path.join(pathname, 'data.csv'), concatenated_data_proc, delimiter=',', header=data_header)
    
    # Save new_bin_array with header as CSV file if it doesn't exist
    if not os.path.exists(os.path.join(pathname, 'species.csv')):
        np.savetxt(os.path.join(pathname, 'species.csv'), new_bin_array, delimiter=',', header=var_spec_header)           
