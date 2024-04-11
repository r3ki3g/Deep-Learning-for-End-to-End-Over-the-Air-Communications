#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    10-Apr-2024 18:05:26

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    rxin = keras.Input(shape=(4,))
    fc_3 = layers.Dense(4, name="fc_3_")(rxin)
    relu_2 = layers.ReLU()(fc_3)
    fc_4 = layers.Dense(4, name="fc_4_")(relu_2)
    softmax = layers.Softmax()(fc_4)
    classoutput = softmax

    model = keras.Model(inputs=[rxin], outputs=[classoutput])
    return model
