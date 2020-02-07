import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
import cv2
import csv
from dataloader import TestDataLoader

class Evaluate():
    def __init__(self):
        self.testdata = TestDataLoader()
        #pass


    def predictall(self, testdatafolder, classfile):
        test_images = self.testdata.load_images(testdatafolder)
        model = load_model("BDBI/RecycleSort/weights/recyclesort_weights.h5")
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])
        classes = [line.rstrip('\n') for line in open("BDBI/RecycleSort/weights/recyclesort_classes.txt", "r")]
        for filename, image in test_images.items():
            pred = model.predict(image)
            pred = pred[0]
            max_index = np.argmax(pred)
            print("Actual:", filename, "Predicted", classes[max_index])



os.chdir('../')
#test = TestDataLoader()
print (os.getcwd())
eval = Evaluate()
print (eval.predictall('BDBI/all_dataset/MainDataFolder/Test/Test_images/', 'BDBI/all_dataset/MainDataFolder/Test/image_labels.csv'))
#filename = 'image_labels.csv'
#print (test.load_classes(filename))
