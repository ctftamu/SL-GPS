


from utils import auto_ign_build_SL
import pickle

# -----------------EDITABLE INPUT VALUES FOR SL_GPS Simulation-----------------

#Name of fuel species
fuel = 'CH4'

#name of detailed mechanism file, either stored in cantera or in custom path
mech_file='gri30.cti'

#Input species whose mole fractions are used as input to ANN
input_specs = ['CH4', 'H2O', 'OH', 'H', 'CO', 'O2', 'CO2', 'O', 'CH3', 'CH']

#Max HHR used as threshold to decide ignition interval
ign_threshold = 9e7

#Time interval between ANN calls before and after ignition
norm_Dt = 0.0002

#Time interval between ANN calls during ignition
ign_Dt = 0.00005

#Initial temperature in Kelvin
T0_in = 1500

#Equivalence ratio
phi = 1.0

#Pressure in atmospheres
atm = 1.0

#End time of the simulation
t_end = 0.002

#Location of file containing min-max scaler specific to ANN for normalizing input
scaler_path = 'Min-Max Scalers/scaler_c_100_a_0.001.pkl'

#Location of h5 file containing trained ANN
model_path = 'Artificial Neural Networks/model_c_100_a_0.001_n_16.h5'

#Location of file containing training data (necessary for indicating which are 
#always or never to be included, as opposed to species considered by ANN)
data_path = 'Training Data/train_data_c_100_a_0.001'

#Name of pkl file to save simulation results to
results_path = 'SL_GPS Simulation Data/results_c_100_a_0.001_n_16.pkl'


# ------------------------------END OF INPUTS---------------------------------

if __name__ == '__main__':
    auto_ign_SL = auto_ign_build_SL(fuel, mech_file, input_specs, norm_Dt, 
                 ign_Dt, T0_in, phi, atm, t_end, scaler_path, model_path, data_path, ign_threshold)
    file = open(results_path,"wb")
    pickle.dump(auto_ign_SL, file)
    
    