import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pandas import read_csv
import matplotlib.pyplot as plt
import os
from pickle import dump
from joblib import Parallel, delayed

# train neural network with input data X and output data Y
def spec_train(X_train, Y_train):
    train_test_model = tf.keras.Sequential()
    model = tf.keras.Sequential()

    # --------------------------------HERE ADD DESIRED NUMBER OF LAYERS AND NEURONS-----------------------------

    #model.add(tf.keras.layers.Dense(8, activation='relu', kernel_initializer='he_normal'))
    #model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal')) 
    train_test_model.add(tf.keras.layers.Dense(8, activation='relu', kernel_initializer='he_normal'))

    # ------------------------------------------END OF EDITABLE ARCHITECTURE------------------------------------

    # add output layer
    model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid'))
    train_test_model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'binary_crossentropy'])
    train_test_model.compile(optimizer='adam', loss='binary_crossentropy',
                             metrics=['binary_accuracy', 'binary_crossentropy'])

    # first train with train-test split until validation loss fails to improve for 100 epochs, then train for same number of epochs with full dataset
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
    train_test_history = train_test_model.fit(X_train, Y_train, epochs=200, batch_size=32, callbacks=[es],
                                              verbose=2, validation_split=0.2)
    history = model.fit(X_train, Y_train, epochs=len(train_test_history.history['loss']), batch_size=32, verbose=2)
    return model, history, train_test_history


# save model and input normalizer to h5 and pkl files, using temperature, pressure, and mole fractions of 'input_specs' as input
def make_model(input_specs, data_path, scaler_path, model_path):
    X = read_csv(os.path.join(data_path, 'data.csv'))
    Y = read_csv(os.path.join(data_path, 'species.csv'))
    X = X[['# Temperature', 'Atmospheres'] + input_specs]
    Y = Y.iloc[:, :-1]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_proc = min_max_scaler.fit_transform(X)

    # Define the number of parallel processes
    num_processes = 28

    # Create a list of inputs for parallel processing
    inputs = [(X_train_proc, Y)] * num_processes

    # Run the function in parallel
    results = Parallel(n_jobs=num_processes)(delayed(spec_train)(*input) for input in inputs)

    models = [result[0] for result in results]
    histories = [result[1] for result in results]
    train_test_histories = [result[2] for result in results]

    # Get the best model based on validation loss
    best_model_idx = min(range(len(train_test_histories)), key=lambda i: train_test_histories[i].history['loss'][-1])
    best_model = models[best_model_idx]

    dump(min_max_scaler, open(scaler_path, 'wb'))
    best_model.save(model_path)

