# RecycleSort
Neural Network Model


Commands:

Run:
python Dataloader.py
 - This should 1) Download dataset and labels file from AWS
               2) Seperate the images into train, test, validation. 
               3) The train, test, validation folders have folders named after the labels (eg, plasticbottles, metalcans, bananapeels etc.)

python train.py
  - This should 1) Tune training hyperparameters (epochs, learning rate, batch size)
                2) Start training MobileNet-v2 model on the training (and validation, it its enabled) dataset. 
                3) Output a training accuracy and loss for each epoch
                4) Save a trained model in .h5 format and labels file in .txt format
                5) Convert .h5 model to .tflite if enabled
                
                
python evaluate.py
  - This should 1) Evaluate the model on the test dataset
                2) Output a test accuracy
                3) Check outputted test accuracy against a pre-set threshold (85%) to determine if model should be deployed
