


import cantera as ct
import numpy as np
import random
from utils import findIgnInterval, GPS_spec, findTimeIndex, auto_ign_build_phi
import os
import concurrent.futures

def make_data_parallel(fuel, mech_file, end_threshold, ign_HRR_threshold_div, ign_GPS_resolution,
              norm_GPS_resolution, GPS_per_interval, n_cases, t_rng, p_rng, phi_rng,
              alpha, input_specs, always_threshold, never_threshold, pathname):
    # Initialize lists of random initial conditions
    temps = []
    pressures = []
    phis = []

    # Get random initial conditions within ranges
    for i in range(n_cases):
        temps.append(random.uniform(t_rng[0], t_rng[1]))  # Uniform distribution
        pressures.append(10**(random.uniform(p_rng[0], p_rng[1])))  # Log uniform distribution
        phis.append(random.uniform(phi_rng[0], phi_rng[1]))  # Uniform distribution

    # Initialize Cantera solution
    soln_in = ct.Solution(mech_file)

    count = 0
    sim_count = 0
    ign = False

    # Get full mechanism species list
    det_spec_strs = soln_in.species_names

    def process_simulation(i):
        nonlocal count, ign, sim_count

        sim_count += 1
        print('Performing Simulation: ' + str(sim_count))

        T0 = temps[i]
        phi = phis[i]
        pres = pressures[i]
        t_end = 5.0

        # Run detailed simulation with current initial conditions
        auto_ign_det = auto_ign_build_phi(soln_in, T0, pres, phi, fuel, end_threshold, t_end, dir_raw='ign')
        ign_Dt = auto_ign_det[0]['axis0'][-1] / ign_GPS_resolution
        norm_Dt = auto_ign_det[0]['axis0'][-1] / norm_GPS_resolution
        Dt = norm_Dt

        # Determine interval of ignition over current simulation to determine frequency of mechanism collection
        ign_start, ign_end = findIgnInterval(auto_ign_det[0]['heat_release_rate'],
                                             max(auto_ign_det[0]['heat_release_rate']) / ign_HRR_threshold_div)
        t = 0.0

        # Begin collecting GPS reduced mechanisms
        while t + Dt < auto_ign_det[0]['axis0'][-1]:
            count += 1

            # Decide which GPS interval to use based on ignition interval
            if t + Dt > auto_ign_det[0]['axis0'][ign_start] and not ign:
                ign = True
                Dt = ign_Dt
            elif t > auto_ign_det[0]['axis0'][ign_end]:
                ign = False
                Dt = norm_Dt

            # Run GPS on data from the current interval and store as a binary list
            spec_strs = GPS_spec(soln_in, fuel, auto_ign_det[0], t, t + Dt, alpha, GPS_per_interval)
            bin_spec_list = np.zeros((1, len(det_spec_strs)), dtype=int)
            for spec in spec_strs:
                if spec in det_spec_strs:
                    bin_spec_list[0, det_spec_strs.index(spec)] = 1

            # Initialize or grow arrays of species and state vector data
            if count == 1:
                bin_array = bin_spec_list
                data_array = np.array([[auto_ign_det[0]['temperature'][findTimeIndex(auto_ign_det[0]['axis0'], t)],
                                        pres, auto_ign_det[0]['mole_fraction'].tolist()[findTimeIndex(auto_ign_det[0]['axis0'], t)]]], dtype=object)
            else:
                bin_array = np.append(bin_array, bin_spec_list, axis=0)
                data_array = np.append(data_array, [[auto_ign_det[0]['temperature'][findTimeIndex(auto_ign_det[0]['axis0'], t)],
                                                     pres, auto_ign_det[0]['mole_fraction'].tolist()[findTimeIndex(auto_ign_det[0]['axis0'], t)]]], axis=0)
            t += Dt

        # Reformat array of state vectors
        data_proc = np.zeros((np.shape(data_array)[0], 2 + len(det_spec_strs)))
        for i in range(np.shape(data_array)[0]):
            for j in range(2):
                data_proc[i, j] = data_array[i, j]
            for k in range(len(det_spec_strs)):
                data_proc[i, 2+k] = data_array[i, 2][det_spec_strs.index(det_spec_strs[k])]

    # Identify species included above and below frequency thresholds to designate
    # "always" and "never" species, to reduce complexity of the output layer of ANN
    always_arr = np.array([], dtype=int)
    never_arr = np.array([], dtype=int)
    for i in range(np.shape(bin_array)[1]):
        if sum(bin_array[:, i]) >= np.shape(bin_array)[0] * always_threshold:
            always_arr = np.append(always_arr, np.array([int(i)]), 0)
        elif sum(bin_array[:, i]) <= np.shape(bin_array)[0] * never_threshold:
            never_arr = np.append(never_arr, np.array([int(i)]), 0)
    delete_arr = np.append(always_arr, never_arr, 0)
    new_bin_array = np.delete(bin_array, delete_arr, 1)
    det_spec_nums = np.arange(0, 53, dtype=int)
    det_spec_nums = np.delete(det_spec_nums, delete_arr, 0)

    # Create headers for binary list csv
    var_spec_header = ''
    for i in range(len(det_spec_strs)):
        if i not in delete_arr:
            var_spec_header += det_spec_strs[i] + ','

    # Create headers for state vector csv
    data_header = 'Temperature,Atmospheres,'
    for spec in det_spec_strs:
        data_header += str(spec) + ','

    # Create lists of "always," "never," and "variable" species
    never_specs = ''
    always_specs = ''
    var_specs = ''
    for i in range(np.shape(never_arr)[0]):
        never_specs += det_spec_strs[never_arr[i]] + ','
    for i in range(np.shape(always_arr)[0]):
        always_specs += det_spec_strs[always_arr[i]] + ','
    for i in range(np.shape(det_spec_nums)[0]):
        var_specs += det_spec_strs[det_spec_nums[i]] + ','

    # Save species, state vector data, and always/never/variable species lists as CSV files for training
    os.mkdir(os.path.join('.', pathname))
    np.savetxt(os.path.join(pathname, 'never_spec_nums.csv'), np.array([]), delimiter=',', header=never_specs)
    np.savetxt(os.path.join(pathname, 'always_spec_nums.csv'), np.array([]), delimiter=',', header=always_specs)
    np.savetxt(os.path.join(pathname, 'var_spec_nums.csv'), np.array([]), delimiter=',', header=var_specs)
    np.savetxt(os.path.join(pathname, 'data.csv'), data_proc, delimiter=',', header=data_header)
    np.savetxt(os.path.join(pathname, 'species.csv'), new_bin_array, delimiter=',', header=var_spec_header)

