import os
import shutil
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
import pandas as pd
import random
import split_folders
import boto3


def create_directories(path):

    try:
        os.mkdir("input")
    except:
        pass

    try:
        os.mkdir("output")
    except:
        pass

    path1 = path + "/input/PlasticBottles"
    try:
        os.mkdir(path1)
    except:
        pass

    path2 = path + "/input/MetalCans"
    try:
        os.mkdir(path2)
    except:
        pass



def split_data(data_dir, csv_file):
    ImageLabels = pd.read_csv(data_dir + '/' + csv_file, header=None)
    ImageDir = {}
    path = data_dir + '/'
    output_dir = path + "output/"
    dictImageLabels = {}
    filenames = []
    create_directories(path)


    for row in ImageLabels.index:
        dictImageLabels[ImageLabels[0][row]] = ImageLabels[1][row]

    for f in filenames:
        if dictImageLabels[f] == 'PlasticBottles':
            shutil.copyfile(path + 'data/' + f, 'input/PlasticBottles/'+f)
        elif dictImageLabels[f] == 'MetalCans':
            shutil.copyfile(path + 'data/' + f, 'input/MetalCans/'+f)


    split_folders.ratio(path + 'input/', output=output_dir, seed = 10, ratio=(.8, .1, .1))
    
'''
def download_from_S3(prefix, local, bucket, client=s3_client):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)
'''

def DataLoader(path):

    pass


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

print (os.getcwd())

#s3_client = boto3.client('s3')
#download_from_S3(s3_client)

#os.chdir("../")
#split_data(os.getcwd(), "image_labels.csv")
