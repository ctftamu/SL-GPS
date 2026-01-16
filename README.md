---
title: SL-GPS
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Simple steps to get started

1. Install basic libraries
   ```
   pip install matplotlib tensorflow
   ```
2. Install cantera libraries
   ```
   pip install --no-cache-dir "cantera==2.6.0"
   ```
3. Install numpy
   ```
   pip install "numpy==1.26.4"
   ```
4. Install networkx (for parallel compute)
   ```
   pip install networkx
   ```
5. Install scikit
   ```
   pip install scikit-learn
   ```
6. Install the SL-GPS library
   ```
   pip install "git+https://github.com/ctftamu/SL-GPS.git"
   ```
7. Test your installation by running any of the files in the [tests folder](tests/)
8. The trained neural network is stored as .h5 file which can be accessed and utilized to produce reduced mechanisms for any given composition, temperature and pressure.
9. Next, convert the generated .h5 file to .pb (frozen graph) to be used in OpenFOAM using the script converth5ToPb.ipynb in [tests folder](tests/).

NOTE: The default neural network architecture is 16x8 (2 hidden layers). To change the neural network architecture, go to the [file](/src/slgps/mech_train.py) and edit the function spec_train according to your needs.

For questions and discussions please join : https://discord.com/channels/1333609076726431798/1333610748424880128

Please feel free to ask any questions related to SL-GPS there. 

# SL-GPS
This repository contains the means to create a neural network architecture for dynamic chemistry reduction based on reduction results from Global Pathway Selection. The basic procedure is to first run adaptive GPS for 0D auto-ignition simulation so as to create a dataset. This dataset is later used for training the Artificial Neural Network (ANN). You can reach out to us at rmishra@tamu.edu (Rohit Mishra) or aaronnelson@tamu.edu (Aaron Nelson) for code issues, suggestions and/or pull requests. 

## About
This code was developed entirely in Python 3. Dependent packages include Cantera 2, Tensorflow 2, pandas, sklearn, numpy, pickle, and networkx. Code for GPS has been copied from https://github.com/golsun/GPS and modified to work in Python 3. 
## How to Cite
- Mishra, R., Nelson, A., Jarrahbashi, D., "Adaptive global pathway selection using artificial neural networks: A-priori study", **Combustion and Flame**, 244 (2022) 112279 [[link](https://doi.org/10.1016/j.combustflame.2022.112279)]
## Related Publications
- X. Gao, S. Yang, W. Sun, "A global pathway selection algorithm for the reduction of detailed chemical kinetic mechanisms", **Combustion and Flame**, 167 (2016) 238-247 [[link](https://doi.org/10.1016/j.combustflame.2016.02.007)]
