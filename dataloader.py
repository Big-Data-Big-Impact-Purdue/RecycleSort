import os
import shutil
import matplotlib.pyplot as plt
#from keras.preprocessing import image
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



def split_data(data_dir, csv_file, client):
    ImageLabels = pd.read_csv(csv_file, header=None)
    path = data_dir + '/'
    output_dir = path + "output/"
    dictImageLabels = {}
    create_directories(path)
    count_dict = {}

    for row in ImageLabels.index:
        dictImageLabels[ImageLabels[0][row]] = ImageLabels[1][row]
        if ImageLabels[1][row] not in count_dict:
            count_dict[ImageLabels[1][row]] = 1
        else:
            count_dict[ImageLabels[1][row]] += 1
    print (count_dict)
    bucket = 'firstpythonbucket70889483-b408-41b1-b2d4-d677152cedb0'
    min_count = min(count_dict["PlasticBottles"], count_dict["MetalCans"])
    count_bottle = 0
    count_can = 0
    for f in dictImageLabels:
        df = path + f
        if dictImageLabels[f] == 'PlasticBottles' and count_bottle < min_count:
            client.Bucket(bucket).download_file(df, df)
            count_bottle += 1
        elif dictImageLabels[f] == 'MetalCans' and count_can < min_count:
            count_can += 1
            client.Bucket(bucket).download_file(df, df)



    #split_folders.ratio(path + 'input/', output=output_dir, seed = 10, ratio=(.8, .1, .1))


def download_from_S3(labels_file, prefix, local_folder, bucket, client):


    client.Bucket(bucket).download_file(labels_file, labels_file)
    try:
        os.mkdir('Test_images')
    except FileExistsError:
        pass
    split_data('Test_images', labels_file, client)
    '''
    
    bucket_obj = client.Bucket(bucket)
    for obj in bucket_obj.objects.all():
        try:
            client.Object(bucket, obj.key).download_file(obj.key)
        except FileExistsError:
            pass
    '''


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


s3_client = boto3.resource('s3')
download_from_S3('image_labels.csv', 'Test_images', '', 'firstpythonbucket70889483-b408-41b1-b2d4-d677152cedb0', s3_client)

#os.chdir("../")
#split_data(os.getcwd(), "image_labels.csv")
