import tensorflow as tf
from keras import backend as K
from tensorflow import keras #, py_func
from sklearn.metrics import r2_score
print(tf.__version__)
import h5py
def r2_2(y_true, y_pred):
    res = py_func(r2_score, [y_true, y_pred], tf.float64)
    return res
filename = 'model_EXTRA_PROD_a_0.00000000000075_l_16.h5'
#str.encode(filename) #.decode()

# Create, compile and train model...n
HeNormal = tf.keras.initializers.he_normal()
#model =  tf.keras.models.load_model('/usr/model_EXTRA_PROD_a_0.00000000000075_l_16.h5',custom_objects={'HeNormal': HeNormal, 'r2_2': r2_2},compile=None)
model =  tf.keras.models.load_model('/usr/sandiaD_model_c_100_a_0.001_n_16.h5',custom_objects={'HeNormal': HeNormal, 'r2_2': r2_2},compile=None)

