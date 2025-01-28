


import matplotlib.pyplot as plt
import cantera as ct
import numpy as np
import pickle

# ------------EDITABLE INPUT VALUES FOR SIMULATION RESULTS DISPLAY-------------

specs = ['CH4', 'OH', 'H2O', 'CO2']

results_path = 'SL_GPS Simulation Data/results_c_100_a_0.001_n_16.pkl'





# ------------------------------END OF INPUTS---------------------------------


def display_sim_data(specs, results_path):
    result_file = open(results_path, 'rb')
    results = pickle.load(result_file)
    results_dict = results[0]
    mech_times = results[1]
    rxn_nums = results[2]
    spec_nums = results[3]
    
    
    soln = ct.Solution(results_dict['mechanism'])
    fig, axs = plt.subplots(5, 1, sharex=True)
    fig.set_figheight(12)
    fig.set_figwidth(8)
    
    
    
    axs[0].plot(results_dict['axis0'], results_dict['temperature'])
    axs[0].set_ylabel('Temperature (K)')
    
    axs[1].plot(results_dict['axis0'], results_dict['heat_release_rate'])
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Heat Release Rate ' + r'$J/(s*m^2)$')
    
    legend_lst = []
    for i, spec in enumerate(specs):
        axs[2].plot(results_dict['axis0'], results_dict['mole_fraction'][:, soln.species_names.index(spec)])
        legend_lst.append(spec)
    axs[2].legend(legend_lst)
    axs[2].set_ylabel('Mole Fraction (-)')
    
    axs[3].plot(mech_times, spec_nums, 'g-o')
    axs[3].set_ylabel('# of Species')
    
    axs[4].plot(mech_times, rxn_nums, 'b-o')
    axs[4].set_ylabel('# of Reactions')
    axs[4].set_xlabel('Time (s)')
    
    plt.show()
    
    
if __name__ == '__main__':
    display_sim_data(specs, results_path)
    
    