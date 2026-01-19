import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pandas import read_csv
import matplotlib.pyplot as plt
import os
from pickle import dump
from joblib import Parallel, delayed

# train neural network with input data X and output data Y
def spec_train(X_train, Y_train, num_hidden_layers: int = 1, neurons_per_layer: int = 16):
    """Train a single Keras model with the given architecture.

    Parameters:
    - X_train: numpy array of input features
    - Y_train: numpy array of binary targets
    - num_hidden_layers: number of hidden Dense layers to add
    - neurons_per_layer: number of neurons in each hidden layer

    Returns:
    - model: trained Keras model
    - history: training history for the final fit on full dataset
    - train_test_history: training history from train/test split
    """
    train_test_model = tf.keras.Sequential()
    model = tf.keras.Sequential()

    # Build hidden layers according to requested architecture
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu', kernel_initializer='he_normal'))
        train_test_model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu', kernel_initializer='he_normal'))

    # add output layer
    model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid'))
    train_test_model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'binary_crossentropy'])
    train_test_model.compile(optimizer='adam', loss='binary_crossentropy',
                             metrics=['binary_accuracy', 'binary_crossentropy'])

    # first train with train-test split until validation loss fails to improve, then train for same number of epochs with full dataset
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
    train_test_history = train_test_model.fit(X_train, Y_train, epochs=200, batch_size=32, callbacks=[es],
                                              verbose=2, validation_split=0.2)
    history = model.fit(X_train, Y_train, epochs=len(train_test_history.history['loss']), batch_size=32, verbose=2)
    return model, history, train_test_history


# save model and input normalizer to h5 and pkl files, using temperature, pressure, and mole fractions of 'input_specs' as input
def make_model(input_specs, data_path, scaler_path, model_path, num_hidden_layers: int = 1, neurons_per_layer: int = 16, num_processes: int = 28):
    X = read_csv(os.path.join(data_path, 'data.csv'))
    Y = read_csv(os.path.join(data_path, 'species.csv'))
    X = X[['# Temperature', 'Atmospheres'] + input_specs]
    Y = Y.iloc[:, :-1]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_proc = min_max_scaler.fit_transform(X)

    # Use the requested number of parallel processes (default 28)
    try:
        num_processes = int(num_processes)
    except Exception:
        num_processes = 28

    if num_processes < 1:
        num_processes = 1

    # Create a list of inputs for parallel processing
    inputs = [(X_train_proc, Y, num_hidden_layers, neurons_per_layer)] * num_processes

    # Run the function in parallel
    results = Parallel(n_jobs=num_processes)(delayed(spec_train)(*input) for input in inputs)

    models = [result[0] for result in results]
    histories = [result[1] for result in results]
    train_test_histories = [result[2] for result in results]

    # Get the best model based on validation loss
    best_model_idx = min(range(len(train_test_histories)), key=lambda i: train_test_histories[i].history['loss'][-1])
    best_model = models[best_model_idx]

    # Create all missing directories if they don't exist
    os.makedirs(scaler_path, exist_ok=True)

    scaler_path_with_file = os.path.join(scaler_path, 'model.pkl')

    # Now you can check for the file
    if not os.path.exists(scaler_path_with_file ):
        # Create the file or process it as needed
        dump(min_max_scaler, open(scaler_path_with_file , 'wb'))



    # Create all missing directories if they don't exist
    os.makedirs(model_path, exist_ok=True)

    model_path_with_file = os.path.join(model_path, 'model.h5')

    # Now you can check for the file
    if not os.path.exists(model_path_with_file):
        # Create the file or process it as needed
        best_model.save(model_path_with_file, save_format='h5')

