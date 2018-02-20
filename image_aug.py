from keras.preprocessing.image import ImageDataGenerator
import pandas as pd 
from matplotlib import pyplot
import data_load
from keras import backend as K
K.set_image_dim_ordering('th')

train_data, train_labels = data_load.train_load()
test_data, test_labels = data_load.test_load()
shift = 0.2
datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)

datagen.fit(train_data)

datagen.flow(train_data, train_labels, batch_size=9, save_to_dir='images', save_prefix='aug', save_format='jpg')
	