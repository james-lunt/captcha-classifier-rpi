#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy
import string
import random
import argparse
import tflite_runtime.interpreter as tflite
import cv2

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    captcha = ''.join([characters[x] for x in y if x != len(characters)])
    print(captcha)
    return captcha

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

   # with tflite.device('/cpu:0'): 
    with open(args.output, 'w') as output_file:
        interpreter = tflite.Interpreter(model_path=args.model_name)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        classified_captchas=[]
        for x in os.listdir(args.captcha_dir):
            # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = numpy.array(rgb_data) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])
            input_data = numpy.array(image, dtype=numpy.float32)
            
            #Make prediction
            interpreter.set_tensor(input_details[0]['index'],input_data) 
            interpreter.invoke()

            #Decode prediction to symbols
            prediction = [])
            prediction.append(interpreter.get_tensor(output_details[3]['index']))
            prediction.append(interpreter.get_tensor(output_details[5]['index']))
            prediction.append(interpreter.get_tensor(output_details[0]['index']))
            prediction.append(interpreter.get_tensor(output_details[4]['index']))
            prediction.append(interpreter.get_tensor(output_details[2]['index']))
            prediction.append(interpreter.get_tensor(output_details[1]['index']))
            
            classified_captchas.append(x + "," + decode(captcha_symbols, prediction) + "\n")
            print('Classified ' + x)
            
            
        classified_captchas.sort()
        for classified_captcha in classified_captchas:
            output_file.write(classified_captcha)

if __name__ == '__main__':
    main()
