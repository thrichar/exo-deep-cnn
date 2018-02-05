import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras import regularizers as regs
from keras import backend as K
K.set_image_dim_ordering('th')

def train_model(binary):
	if binary == True:
		num_classes = 2
	else:
		num_classes = 6

	init = 'glorot_uniform'
	act = 'relu'
	pad = 'same'

	model = Sequential()

	# Block 1
	model.add(Convolution2D(16,(3,5),kernel_initializer=init,activation=act,padding=pad,input_shape=(1,76,1000),name='block1_conv1'))
	model.add(MaxPooling2D((2,4),name='block1_pool'))

	# Block 2
	model.add(Convolution2D(32,(3,5),kernel_initializer=init,activation=act,padding=pad,name='block2_conv1'))
	model.add(MaxPooling2D((2,4),name='block2_pool'))

	# Block 3
	model.add(Convolution2D(64,(3,3),kernel_initializer=init,activation=act,padding=pad,name='block3_conv1'))
	model.add(MaxPooling2D((2,2),name='block3_pool'))

	# Block 4
#	model.add(Convolution2D(128,(3,3),kernel_initializer=init,activation=act,padding=pad,name='block4_conv1'))
#	model.add(MaxPooling2D((2,2),name='block4_pool'))

	# Block 5
#	model.add(Convolution2D(256,(3,3),kernel_initializer=init,activation=act,padding=pad,name='block5_conv1'))
#	model.add(MaxPooling2D((2,2),name='block5_pool'))

	L2 = 1.e-2
	# Regression
	model.add(Flatten(name='flatten'))

	model.add(Dense(256,kernel_initializer=init,activation=act,kernel_regularizer=regs.l2(L2),name='fc1'))
	model.add(Dropout(0.5,name='dropout1'))

	model.add(Dense(128,kernel_initializer=init,activation=act,kernel_regularizer=regs.l2(L2),name='fc2'))
	model.add(Dense(64,kernel_initializer=init,activation=act,kernel_regularizer=regs.l2(L2),name='fc3'))
	model.add(Dropout(0.5,name='dropout2'))

	model.add(Dense(32,kernel_initializer=init,activation=act,kernel_regularizer=regs.l2(L2),name='fc4'))
	model.add(Dense(16,kernel_initializer=init,activation=act,kernel_regularizer=regs.l2(L2),name='fc5'))

	model.add(Dense(num_classes,kernel_initializer=init,activation='softmax',name='predictions'))

	return model
