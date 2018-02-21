# Capon-Vs-Capoff

This project is a predictive model using Mxnet which can detect wheather a bottle has a Cap or not. 

Steps to run the code
=====================

1) Save the images of the bottle into their corresponding folders(Capon or Capoff) in the Dataset directory.
2) Run the annotate.py script. This will generate the csv files of image name and their labels.
3) Run the train_test_csv_creation.py script. This will generate the csv files of train and test data.
4) Run the train_test_data_creation.py script. This will create test and train data directories.
5) Run the train.py script to train the model.
6) Run the fine_tuning.py script for reloading the saved model for fine tuning change the prefix and epoch number in the script with the model you want to load.
7) Run inference.py file for inferencing.
