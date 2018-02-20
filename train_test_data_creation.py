import pandas as pd 
import os
from subprocess import call


path = os.getcwd() + "/Dataset/"
capon = path + "CapON/"
capoff = path + "Capoff/"
train_data = path + "train"
test_data = path + "test"



train = pd.read_csv(os.path.join(path,"train.csv"))
test = pd.read_csv(os.path.join(path,"test.csv"))

print(train.ix[0,1])




for i in range(train.shape[0]):

	if train.ix[i,1] == 1:
		file = os.path.join(capon,str(train.ix[i,0]))
		call([ "cp",file,train_data])

	else:
		file = os.path.join(capoff,str(train.ix[i,0]))
		call([ "cp",file,train_data])
	


for i in range(test.shape[0]):

	if test.ix[i,1] == 1:
		file = os.path.join(capon,str(test.ix[i,0]))
		call([ "cp",file,test_data])

	
	else:
		file = os.path.join(capoff,str(test.ix[i,0]))
		call([ "cp",file,test_data])
	