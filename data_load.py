import numpy as np 
import cv2 
import pandas as pd
import os
import matplotlib.pyplot as plt 

data_dir = os.getcwd() + "/Dataset/"

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir,'test.csv'))

width = 32
height = 32
channel = 3

def train_load():
	temp = []
	temp1 = []
	for image_name in train.image:
		image_path = os.path.join(data_dir,"train",image_name)
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image,(width,height))
		temp.append(image)

	for label in train.label:
		temp1.append(label)


	train_data = np.stack(temp).reshape([-1,channel,width,height])
	train_label = np.stack(temp1)

	return train_data, train_label


def test_load():
	temp = []
	temp1 = []
	for image_name in test.image:
		image_path = os.path.join(data_dir,"test",image_name)
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image,(width,height))
		temp.append(image)

	for label in test.label:
		temp1.append(label)


	test_data = np.stack(temp).reshape([-1,channel,width,height])
	test_label = np.stack(temp1)

	return test_data, test_label

if __name__ == '__main__':

	train_data ,train_label = train_load()
	test_data, test_label = test_load()
	print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)
	cv2.imshow("image",train_data[704].reshape(300,300,1))
	cv2.waitKey(0)
	plt.show()



