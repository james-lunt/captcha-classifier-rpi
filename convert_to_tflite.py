#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--model-output', help='Model name to use for classification', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.model_output is None:
        print("Please specify the tflite model to output")
        exit(1)

    json_file = open(args.model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(args.model_name+'.h5')
    model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                    metrics=['accuracy'])

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(args.model_output, 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    main()