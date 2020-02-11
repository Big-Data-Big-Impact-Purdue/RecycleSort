import tensorflow as tf
import os
import h5py

def convert_to_tflite():
    converter = tf.lite.TFLiteConverter.from_keras_model_file('object_model.h5')
    print ("Here")
    tfmodel = converter.convert()
    open ("model.tflite" , "wb").write(tfmodel)

convert_to_tflite()
