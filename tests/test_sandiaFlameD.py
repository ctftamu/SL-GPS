
from slgps.make_data_parallel import make_data_parallel
from slgps.mech_train import make_model
import os

# -----------------EDITABLE INPUT VALUES FOR SIMULATION SET-------------------

#fuel species name
fuel = 'CH4'

#name of detailed mechanism file, either stored in cantera or in custom path
mech_file='gri30.yaml'

#HHR threshold, below which simulation ends
end_threshold = 2e5

#Division of Max HHR used as threshold to decide ignition interval
ign_HRR_threshold_div = 3e2

#Division of detailed simulation time over which a single GPS mechanism is produced, during ignition
ign_GPS_resolution = 200

#Division of detailed simulation time over which a single GPS mechanism is produced, not during ignition
norm_GPS_resolution = 40

#Number of time steps (evenly spaced) within each interval on which to perform GPS
GPS_per_interval = 4

#Number of training simulations
n_cases= 100 #1000

#training range for initial temperature (K)
t_rng=[800,2300]

#training range for log(pressure (atm))
p_rng=[2.1, 2.5]

#training range for equivalance ratios (-)
phi_rng=[0.6, 1.4] # Not utilized since we are using species_ranges (still need to specify any dummy range to run the code)

#alpha (error control for GPS)
alpha=0.001

# Define the species and their allowable range
species_ranges = {
'CH4': (0.0, 1.0),
'N2': (0, 0.8),
'O2': (0.0, 0.4),
'CO2': (0.0, 0.005),
'H2O': (0.0,  0.1),
'OH': (0.00, 1e-3)
}

#Frequency threshold for inclusion of a species above which a species is always
#to be included in a reduced mechanism, and not considered in ANN
always_threshold = 0.99

#Frequency threshold for inclusion of a species below which a species is never
#to be included in a reduced mechanism, and not considered in ANN
never_threshold = 0.01

#Name of path to store new species and state vector data or to retrieve data from
data_path = 'TrainingData/Sandia_100sims' #Sandia_train_data_c_100_a_0.001'

#Species whose mole fractions are to be used as the inputs of the neural network
input_specs = ['CH4', 'H2O', 'OH', 'H', 'CO', 'O2', 'CO2', 'O', 'CH3', 'CH', 'H2']

#Path to min max scaler for normalizing input data to neural network
scaler_path = 'Min-Max Scalers/Sandia_100sims' #Sandia_scaler_c_100_a_0.001.pkl'

#Path to h5 file containing trained neural network
model_path = 'Artificial Neural Networks/Sandia_100sims' #Sandia_model_c_100_a_0.001_n_16.h5'

    
# ------------------------------END OF INPUTS---------------------------------


#produce autoignition data and train model according to the above parameters
if __name__ == '__main__':
    if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
        # Create the directory if it doesn't exist or if it's empty
        os.makedirs(data_path)

        make_data_parallel(fuel, mech_file, end_threshold, ign_HRR_threshold_div, ign_GPS_resolution,\
        norm_GPS_resolution, GPS_per_interval, n_cases, t_rng, p_rng, phi_rng, alpha, always_threshold,\
        never_threshold, data_path, species_ranges)

    make_model(input_specs, data_path, scaler_path, model_path)