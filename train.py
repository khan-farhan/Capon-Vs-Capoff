

###################################################################################################
import data_load
import mxnet as mx
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout


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
################################### Generating graph for computaion ###############################
data = mx.sym.var('data')
# First convolution lyer
conv1 = mx.sym.Convolution(data = data, kernel = (3,3), num_filter = 32, stride = (1,1))
bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=0.9)
relu1 = mx.sym.Activation(data = bn1, act_type = 'relu')
pool1 = mx.sym.Pooling(data = relu1, pool_type = 'max', kernel = (2,2), stride = (1,1))


# Second convolution layer
conv2 = mx.sym.Convolution(data = pool1, kernel = (3,3), num_filter = 64, stride = (1,1))
bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=0.9)
relu2 = mx.sym.Activation(data = bn2, act_type = 'relu')
pool2 = mx.sym.Pooling(data = relu2, pool_type = 'max', kernel = (2,2), stride = (1,1))


conv3 = mx.sym.Convolution(data = pool2, kernel = (2,2), num_filter = 64, stride = (1,1))
bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=0.9)
relu3 = mx.sym.Activation(data = bn3, act_type = 'relu')
pool3 = mx.sym.Pooling(data = relu3, pool_type = 'max', kernel = (2,2), stride = (1,1))

"""

conv4 = mx.sym.Convolution(data = pool3, kernel = (2,2), num_filter = 64, stride = (1,1))
bn4 = mx.sym.BatchNorm(data=conv4, fix_gamma=False, eps=2e-5, momentum=0.9)
relu4 = mx.sym.Activation(data = bn4, act_type = 'relu')
pool4 = mx.sym.Pooling(data = relu4, pool_type = 'max', kernel = (2,2), stride = (1,1))

"""
# First fully connected
flatten = mx.sym.Flatten(data = pool3)
fc1 = mx.sym.FullyConnected(data = flatten, num_hidden = 1024)
relu5 = mx.sym.Activation(data = fc1, act_type = 'relu')
# Second fully connected
fc2 = mx.sym.FullyConnected(data = relu5, num_hidden = 2)
# Softmax loss
net = mx.sym.SoftmaxOutput(data = fc2, name = 'softmax')
#################################### Creating Module from the network##############################
net_model = mx.mod.Module(symbol = net, context = mx.cpu())
saved_model_path = os.getcwd() + "/saved_model/"
prefix = saved_model_path + 'mymodel-new'

net_model.fit(train_iter,
			  eval_data = test_iter,
			  initializer = mx.initializer.Xavier(magnitude = 2.0),
			  optimizer='adam',
			  optimizer_params = {'learning_rate':0.01 },
			  eval_metric = 'acc',
			  epoch_end_callback = mx.callback.do_checkpoint(prefix,10),
			  batch_end_callback = mx.callback.Speedometer(batch_size,3),
			  num_epoch = 1000,
			  validation_metric = 'acc' )



##################################### Test the result #############################################
# predict accuracy for lenet
acc = mx.metric.Accuracy()
net_model.score(test_iter, acc)
print(acc)

