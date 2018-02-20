'''
This code runs inference model on CIFAR-10 dataset using mxnet
'''
###################################################################################################
import os
import data_load
import mxnet as mx
import logging
import time
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
################################### Load data #####################################################

# load the data in NDArray
test_data, test_labels = data_load.test_load()
train_data, train_labels = data_load.train_load()
print ('test_data shape: ', end = '')
print (test_data.shape)
print ('test_labels shape: ', end = '')
print (test_labels.shape)
# generate iterator
batch_size = 20
test_iter = mx.io.NDArrayIter(test_data, test_labels, batch_size)
batch_size = 369
train_iter = mx.io.NDArrayIter(data = train_data, label = train_labels, batch_size = batch_size, shuffle = True)
#################################### Load the model ###############################################
tic = time.clock()
saved_model_path = os.getcwd() + "/saved_model/"
prefix = saved_model_path + 'mymodel-new-ft'
epoch = 130
net, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
net_model = mx.mod.Module(symbol = net, context = mx.cpu())
net_model.bind(for_training = False, data_shapes = test_iter.provide_data, label_shapes = test_iter.provide_label)
#print (net_model._label_shapes)
net_model.set_params(arg_params, aux_params, allow_missing = True)
print ("Time to load the model: " + str(time.clock() - tic))
#################################### Predict ######################################################
tic = time.clock()
predicted = net_model.predict(test_iter, 1)
acc = mx.metric.Accuracy()
net_model.score(test_iter, acc)
print ("Time to predict: " + str(time.clock() - tic))
print(acc)

output = []
for list in predicted:
	if list[0] > list[1]:
		output.append("Capoff")
	else:
		output.append("Capon")

print(predicted)
print(output)
