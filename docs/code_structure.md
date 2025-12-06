# Code Structure and Customization

The core functionality of SL-GPS is modular, allowing key components to be customized.

## Key Files for Customization

| File | Description | Customization Notes |
| :--- | :--- | :--- |
| `main.py` | Main script for generating training data and triggering model training. | [cite_start]Adjust input parameters like `t_rng`, `p_rng`, `n_cases`, `alpha`, and `species_ranges` to control the data generation process[cite: 186550]. |
| `mech_train.py` | Contains the logic for the Neural Network training. | The **architecture of the neural network** can be modified in the indicated section of code. The default is typically one hidden layer with 16 neurons. [cite_start]Other hyperparameters (e.g., stopping criteria) can also be edited[cite: 186550]. |
| `SL_GPS.py` | Main script for running the adaptive reduction simulation. | [cite_start]Adjust simulation controls like initial conditions, `t_end`, and the mechanism update frequency (`norm_Dt`, `ign_Dt`)[cite: 186550, 186638]. |
| `tests/converth5ToPb.ipynb` | Jupyter notebook example. | [cite_start]Used to convert the trained `.h5` model to a `.pb` (frozen graph) format for use in other platforms like OpenFOAM[cite: 186554]. |

## Default ANN Architecture

[cite_start]The default neural network architecture is specified in `mech_train.py`[cite: 186554].

* [cite_start]**Default Layers:** It usually defaults to a single hidden layer with 16 neurons[cite: 186550].
* [cite_start]**Activation:** Sigmoid activation function is used[cite: 186550].
* [cite_start]**Loss Function:** Binary crossentropy loss is used[cite: 186550].

[cite_start]_Note: One version of the repository notes a default of "16x8 (2 hidden layers)" in its README, suggesting the architecture may be subject to changes and should be verified in the source code if an older version is in use._ [cite: 186554]