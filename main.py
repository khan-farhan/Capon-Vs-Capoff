import numpy as np
import cv2
import mxnet as mx
import argparse
import os
import time

start = time.time()
path = "/Users/p439/Desktop/CapOFF/"
saved_model_path = os.getcwd() + "/saved_model/"
prefix = saved_model_path + 'mymodel-new-ft'
ctx = mx.cpu()

batch_size = 1

epoch = 130

net, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

net_model = mx.mod.Module(symbol = net, context = ctx)




def handler(iter, _ctx):
    for img in iter:
        img = cv2.resize(img, (32,32))  # resize to 224*224 to fit model
        img = img.reshape([1,32,32])
        img = img[np.newaxis, :]  # extend to (n, c, h, w)
        img_iter = mx.io.NDArrayIter(img, None, batch_size)
        net_model.bind(for_training = False, data_shapes = img_iter.provide_data, label_shapes = None)
        net_model.set_params(arg_params, aux_params, allow_missing = True)
        predicted = net_model.predict(img_iter, 1)
        output = []

        for list in predicted:
            if list[0] == max(list[0],list[1]):
                output.append('CapOFF')
            elif list[1] == max(list[0],list[1]):
                output.append('CapON')
            else:
                output.append('Nothing')
        
        return output, predicted


if __name__ == '__main__':
    image = path + "img208.jpg"
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    print(handler([img],None))
    print(time.time() - start)

    
   
