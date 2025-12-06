# Usage Workflow

[cite_start]The repository facilitates a two-stage workflow: **Creating an ANN** (Training) and **Testing the ANN** (Simulation)[cite: 186550, 186638].

## 1. Creating an ANN (Training)

To produce a new ANN as an `.h5` file, run the `main.py` file.

### Data Generation
[cite_start]If the input parameter `'data_path'` specified in `main.py` does **not** point to a pre-existing directory containing state vector and reduced species data, the script will perform the following[cite: 186550]:
1.  [cite_start]Run a series of **autoignition simulations** across a random set of initial conditions (T, P, $\phi$)[cite: 186550].
2.  [cite_start]Run **classic GPS** on the simulation results to produce the training dataset[cite: 186550].
3.  [cite_start]The simulation parameters are controlled by variables in `main.py`, including the `fuel`, `detailed mechanism` (`mech_file`), ranges of initial temperature (`t_rng`), pressure (`p_rng`), and equivalence ratio (`phi_rng`), the `number of cases` (`n_cases`), and the **tolerance of GPS** in including species (`alpha`)[cite: 186550].

### ANN Training
[cite_start]Once data exists at the specified directory, an ANN will be trained[cite: 186550].

* [cite_start]The trained ANN is stored as an **.h5 file** in a directory specified in `main.py` (variable: `model_path`)[cite: 186550].
* The ANN predicts the inclusion of a subset of species (variable species). The full species list is partitioned into three groups based on training data frequency thresholds (`always_threshold`, `never_threshold`):
    * **Variable Species** (determined by ANN)
    * [cite_start]**Always Included** species (`always_spec_nums.csv`) [cite: 186550]
    * [cite_start]**Never Included** species (`never_spec_nums.csv`) [cite: 186550]

## 2. Testing the ANN (Simulation)

[cite_start]To test a trained ANN by adaptively reducing a mechanism using the **SL-GPS method** over the course of an autoignition simulation, run `SL_GPS.py`[cite: 186550].

[cite_start]Parameters within `SL_GPS.py` control the following[cite: 186550, 186638]:
* Initial conditions (`T0_in`, `phi`, `atm`)
* Simulation duration (`t_end`)
* Frequency of mechanism updates (`norm_Dt`, `ign_Dt`)
* Paths to the trained ANN and scaler (`model_path`, `scaler_path`)

[cite_start]The result of this simulation, including temperature, heat release rate, mole fractions, and net reaction rates, is stored in a **.pkl file**[cite: 186550, 186638].

## 3. Plotting Results

To visualize the simulation results, run `display_sim_data.py` with the path to the `.pkl` file in the input parameters. [cite_start]It uses **matplotlib** to display time profiles of[cite: 186550, 186638]:
* Temperature
* Heat release rate
* Mole fractions for selected species
* The number of species and reactions in the reduced mechanisms selected by the ANN over time