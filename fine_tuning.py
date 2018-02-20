'''
This code runs inference model on CIFAR-10 dataset using mxnet
'''
###################################################################################################

import data_load
import os
import mxnet as mx
import logging
import time
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

################################### Load data #####################################################

train_data, train_labels = data_load.train_load()
test_data, test_labels = data_load.test_load()
print ('train_data shape: ', end = '')
print (train_data.shape)
print ('train_labels shape: ', end = '')
print (train_labels.shape)
print ('test_data shape: ', end = '')
print (test_data.shape)
print ('test_labels shape: ', end = '')
print (test_labels.shape)
################################### Prepare data for mxnet ########################################
batch_size = 100
train_iter = mx.io.NDArrayIter(data = train_data, label = train_labels, batch_size = batch_size, shuffle = True)
test_iter = mx.io.NDArrayIter(data = test_data, label = test_labels, batch_size = 20)
#################################### Load the model ###############################################
tic = time.clock()
saved_model_path = os.getcwd() + "/saved_model/"
prefix = saved_model_path + 'mymodel-new-ft'
epoch = 130
net, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
net_model = mx.mod.Module(symbol = net, context = mx.cpu())
net_model.bind(for_training = True, data_shapes = train_iter.provide_data, label_shapes = train_iter.provide_label)
#print (net_model._label_shapes)
net_model.set_params(arg_params, aux_params, allow_missing = True)
print ("Time to load the model: " + str(time.clock() - tic))
#################################### Predict ######################################################
tic = time.clock()
prefix = saved_model_path + 'mymodel-new-data-ft'
net_model.fit(train_iter,
			  eval_data = test_iter,
			  optimizer='adam',
			  optimizer_params = {'learning_rate': 0.01},
			  eval_metric = 'acc',
			  epoch_end_callback = mx.callback.do_checkpoint(prefix,60),
			  batch_end_callback = mx.callback.Speedometer(batch_size,100),
			  num_epoch = 1000,
			  validation_metric = 'acc' )
predicted = net_model.predict(test_iter, 1)
acc = mx.metric.Accuracy()
net_model.score(test_iter, acc)
print(acc)

output = []
for list in predicted:
    if list[0] == max(list[0],list[1]):
        output.append('CapOFF')
    elif list[1] == max(list[0],list[1]):
        output.append('CapON')
    else:
        output.append('Nothing')

print(predicted)
