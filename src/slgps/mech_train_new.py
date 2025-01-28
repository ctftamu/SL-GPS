
from math import *
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pandas import read_csv
import matplotlib.pyplot as plt
import os
from pickle import dump

def spec_train(X, Y, path_name):
    # tf.keras.utils.normalize(X_train)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # n_features = X_train.shape[1]
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    # model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    # model.add(tf.keras.layers.Dense(24, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'))
    # model.add(tf.keras.layers.Dense(12, activation='relu', kernel_initializer='he_normal'))
    # model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'))
    # model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    # model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'binary_crossentropy'])
    # es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
    # history = model.fit(X_train, Y_train, epochs=10000, batch_size=32, callbacks=[es], verbose=2, validation_data=(X_test, Y_test))
    history = model.fit(X, Y, epochs=640, batch_size=32, verbose=2)
    # history = model.fit(X_train, Y_train, epochs=150, batch_size=32, verbose=2)
    # plt.plot(history.history['binary_accuracy'], label='train')
    # plt.legend()
    # plt.show()
    model.save(path_name)
    return model, history

# tracked_specs = ['# Temperature', 'Atmospheres', 'C2H5OH', 'H2O', 'OH', 'H', 'CO', 'O2', 'CO2', 'O', 'CH3']
tracked_specs = ['# Temperature', 'Atmospheres', 'CH4', 'H2O', 'OH', 'H', 'CO', 'O2', 'CO2', 'O', 'CH3', 'CH']
# X = read_csv(os.path.join('train_data_rand_c_500_a_0.01_dt_200_pcrl', 'data.csv'))
# Y = read_csv(os.path.join('train_data_rand_c_500_a_0.01_dt_200_pcrl', 'species.csv'))
X = read_csv(os.path.join('train_data_EXTRA_PROD_a_0.00000000000075', 'data.csv'))
Y = read_csv(os.path.join('train_data_EXTRA_PROD_a_0.00000000000075', 'species.csv'))

X = X[tracked_specs]
Y = Y.iloc[: , :-1]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_proc = min_max_scaler.fit_transform(X)
# dump(min_max_scaler, open('Sensitivity Models/scaler_c_500_a_0.01_l_16_8_dt_200_pcrl.pkl', 'wb'))
# model, history = spec_train(X_train_proc, Y_train, 'Sensitivity Models/model_c_500_a_0.01_l_16_8_dt_200_pcrl.h5')
model, history = spec_train(X_train_proc, Y, 'Sensitivity Models/model_EXTRA_PROD_a_0.00000000000075_l_16.h5')
dump(min_max_scaler, open('Sensitivity Models/model_EXTRA_PROD_a_0.00000000000075_l_16.pkl', 'wb'))
# dump(min_max_scaler, open('scaler.pkl', 'wb'))
# model, history = spec_train(X_train_proc, Y_train, 'sup_model.h5')



# X_test_proc = min_max_scaler.transform(X_test)
# f = model.evaluate(X_test_proc, Y_test, verbose=0)
# print(model.evaluate(X_test_proc, Y_test, verbose=0))

# plt.rcParams["figure.figsize"] = (30,8)
# f, axs = plt.subplots(1, 3, sharex=False)
# axs[0].plot(history.history['val_loss'])
# axs[0].set(xlabel='Number of Epochs Trained', ylabel = 'Binary Accuracy', title='Binary Accuracy of ANN vs. Epoch')
# axs[1].plot(history.history['val_binary_accuracy'])
# axs[1].set(xlabel='Number of Epochs Trained', ylabel = 'Binary Crossentropy', title='BCE of ANN vs. Epoch')


# epoch_plot_data = {'acc': history.history['binary_accuracy'], 'bce': history.history['binary_crossentropy']}
# with open('epoch_plot_data', 'wb') as results_file:
#     dump(epoch_plot_data, results_file)



# print('Loss: %.3f, Accuracy: %.3f' % (loss, acc))