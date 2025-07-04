NOTE: This readme is for an older version of SL-GPS.
# SL-GPS
This repository contains the means to create a neural network architecture for dynamic chemistry reduction based on reduction results from Global Pathway Selection. The basic procedure is to first run adaptive GPS for 0D auto-ignition simulation so as to create a dataset. This dataset is later used for training the Artificial Neural Network (ANN). You can reach out to us at rmishra@tamu.edu (Rohit Mishra) or aaronnelson@tamu.edu (Aaron Nelson) for code issues, suggestions and/or pull requests. The step-by-step procedure to use this repository is as follows:

## Creating an ANN
* To produce a new ANN as an h5 file, run the main.py file. If the input parameter 'data_path' specified at the top of the file points to a pre-existing directory containing the state vector and reduced species data, an ANN will be trained to select a reduced set of species given a thermochemical state as input. If the data does not exist, a series of autoignition simulations will be run across a random set of initial conditions, and classic GPS will be run on the results to produce a set of data that is stored at the specified directory. The nature of these simulations is controlled by the other input parameters in main.py, including the fuel, detailed mechanism, ranges of initial temperatures, pressures, and equivalence ratios considered, the number of cases run, and the tolerance of GPS in including species ("alpha").

* The architecture of the neural network defaults to one hidden layer with 16 neurons, but this can be changed in mech_train.py in the indicated section of code. Other hyperparameters of the training process can also be edited in this file, which uses the Python Keras API in Tensorflow 2. By default, a sigmoid activation function and binary crossentropy loss are used. An 80-20 training test split is performed and the model is trained until validation loss fails to improve for 20 epochs. The trained ANN is then stored as an h5 file in another directory specified in main.py. 

* Please keep in mind that the ANN predicts the inclusion of only a subset of the detailed mechanism's species, while the remaining species are marked to be always or never included. These groups are indicated by the "var_spec_nums.csv," "always_spec_nums.csv," and "never_spec_nums.csv" respectively, all stored in the ANN's corresponding training data directory.

## Testing the ANN
* This repository also allows the testing of an ANN by adaptively reducing a mechanism using the SL-GPS method over the course of another autoignition simulation. This simulation can be run from the SL_GPS.py file. Another set of input parameters at the top of this file control the initial conditions, simulation duration, and frequency of mechanism updates over the course of the simulation. The results, including temperature, heat release rate, mole fractions, and net reaction rates are stored in a pkl file in a directory given to the input parameters.

* To plot the results, display_sim_data.py can be run with the path to the pkl file in the input parameters. Using matplotlib, this will display the time profiles of temperature, heat release rate, the mole fractions of a set of species given in the input parameters, and the numbers of species and reactions in the reduced mechanisms selected by the ANN over the course of the simulation.
## About
This code was developed entirely in Python 3. Dependent packages include Cantera 2, Tensorflow 2, pandas, sklearn, numpy, pickle, and networkx. Code for GPS has been copied from https://github.com/golsun/GPS and modified to work in Python 3. 
## How to Cite
- Mishra, R., Nelson, A., Jarrahbashi, D., "Adaptive global pathway selection using artificial neural networks: A-priori study", **Combustion and Flame**, 244 (2022) 112279 [[link](https://doi.org/10.1016/j.combustflame.2022.112279)]
## Related Publications
- X. Gao, S. Yang, W. Sun, "A global pathway selection algorithm for the reduction of detailed chemical kinetic mechanisms", **Combustion and Flame**, 167 (2016) 238-247 [[link](https://doi.org/10.1016/j.combustflame.2016.02.007)]
