


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pandas import read_csv
import matplotlib.pyplot as plt
import os
from pickle import dump


#train neural network with input data X and output data Y
def spec_train(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    train_test_model = tf.keras.Sequential()
    model = tf.keras.Sequential()
    
    
    
    #--------------------------------HERE ADD DESIRED NUMBER OF LAYERS AND NEURONS-----------------------------
    
    model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'))
    train_test_model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'))
    
    #------------------------------------------END OF EDITABLE ARCHITECTURE------------------------------------
    

    #add output layer
    model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid'))
    train_test_model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid'))
    
    
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'binary_crossentropy'])
    train_test_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'binary_crossentropy'])
    
    #first train with train-test split until validation loss fails to improve for 100 epochs, then train for same number of epochs with full dataset
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
    train_test_history = train_test_model.fit(X_train, Y_train, epochs=10000, batch_size=32, callbacks=[es], verbose=2, validation_data=(X_test, Y_test))
    history = model.fit(X, Y, epochs=len(train_test_history.history['loss']), batch_size=32, verbose=2)
    return model, history, train_test_history


#save model and input normalizer to h5 and pkl files, using temperature, pressure, and mole fractions of 'input_specs' as input
def make_model(input_specs, data_path, scaler_path, model_path):
    
    X = read_csv(os.path.join(data_path, 'data.csv'))
    Y = read_csv(os.path.join(data_path, 'species.csv'))
    X = X[['# Temperature', 'Atmospheres'] + input_specs]
    Y = Y.iloc[: , :-1]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_proc = min_max_scaler.fit_transform(X)
    model, history, train = spec_train(X_train_proc, Y)
    dump(min_max_scaler, open(scaler_path, 'wb'))
    model.save(model_path)




