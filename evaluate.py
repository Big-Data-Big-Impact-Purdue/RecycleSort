import keras
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
import cv2
import csv

class TestDataLoader():
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def load_classes(self, image_labels_file):
        image_classes = {}
        f = open(image_labels_file, "r")
        rows = f.readlines()
        for row in rows:
            list_of_rowtext = row.split()
            image_classes[list_of_rowtext[0]] = list_of_rowtext[1]

        f.close()
        return image_classes


    def load_single_image(self, img_path, show=False):
            img = image.load_img(img_path, self.target_size)
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.

            if show:
                plt.imshow(img_tensor[0])
                plt.axis('off')
                plt.show()

            return img_tensor

    def load_images(self, image_folder, valid_images = [".jpeg"]):
        images = {}
        for f in os.listdir(image_folder):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            image_tensor = self.load_single_image(os.path.join(image_folder, f))
            images[f] = image_tensor

        return images

class Evaluate():
    def __init__(self):
        self.testdata = TestDataLoader()
        #pass


    def predictall(self, testdatafolder, classfile):
        test_images = self.testdata.load_images(testdatafolder)
        model = load_model("RecycleSort/weights/recyclesort_weights.h5")
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])
        classes = [line.rstrip('\n') for line in open("RecycleSort/weights/recyclesort_classes.txt", "r")]
        for filename, image in test_images.items():
            pred = model.predict(image)
            pred = pred[0]
            max_index = np.argmax(pred)
            print("Actual:", filename, "Predicted", classes[max_index])



os.chdir('../../')
#test = TestDataLoader()
eval = Evaluate()
print (eval.predictall('all_dataset/MainDataFolder/Test/Test_images/', 'all_dataset/MainDataFolder/Test/image_labels.csv'))
#filename = 'image_labels.csv'
#print (test.load_classes(filename))
