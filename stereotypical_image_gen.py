import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
from numpy import unravel_index
import time
from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('th')
from keras import regularizers as regs
#import plot_wires
from scipy.ndimage.filters import gaussian_filter
from datetime import date

date = "2018-01-23"

img_height = 1000
img_width = 76

#json_file = open("nets/train_model_new.json","r")
json_file = open("nets/binary_classifier_%s.json"%date,"r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#model.load_weights("nets/train_model_new.h5")
model.load_weights("nets/binary_classifier_%s.h5"%date)
print("Loaded model from disk.")

model.summary()

input_img = model.layers[0].input
layer_output = model.layers[-1].output

learning_rate = 4.e1

def reg(image,step):
#	image = gaussian_filter(image, sigma=1)
#	rms = np.sqrt(np.mean(np.square(image)))
#	for i in range(76):
#		for j in range(1000):
#			pixel = image[0][0][i][j]
#			if abs(pixel) < 0.7*rms:
#				image[0][0][i][j] = 0.0
	return image

final_image = np.zeros((img_width, img_height))
for class_index in [0,1]:
	print('Processing class %d' % class_index)
	start_time = time.time()

	loss1 = layer_output[0, class_index]
	l2 = 1.e-9*K.sum(K.square(K.abs(input_img)))

	if class_index == 0:
		other_class_index = 1
	if class_index == 1:
		other_class_index = 0

	loss2 = layer_output[0, other_class_index]

	# regularization function penalizes activation
	# by minimizing activation of other class,
	# L2 regularization
#	reg = loss2 + l2
	loss = loss1 - loss2 - l2
	# but trying to turn regularization into operator
	grads = K.gradients(loss, input_img)[0]
	grads /= (K.sqrt(K.mean(K.square(grads)))+1e-5)
	iterate = K.function([input_img, K.learning_phase()], [loss, grads])

	input_img_data = np.random.normal(loc=0.0,scale=100.0,size=(1,1,img_width,img_height))
#	input_img_data = np.zeros((1,1,img_width,img_height),np.float32)
	if class_index == 0:
		save_name = 'signal'
	if class_index == 1:
		save_name = 'background'

	for i in range(3000):
		loss_value, grads_value = iterate([input_img_data, 0])

		input_img_data = reg(input_img_data+learning_rate*grads_value, i)

#		input_img_data += grads_value * learning_rate
		print('Current loss value:', loss_value)
#		if i % 10 == 0:
#		input_img_data = gaussian_filter(input_img_data, sigma=2)
#		print 'Current grads value:', grads_value[0][0][0][0]

#		img = input_img_data[0]
#		final_image[0: img_width, 0: img_height] = img
#		plt.style.use('classic')
#		plt.imshow(final_image, aspect='auto')
#		plt.savefig('visualizations/stereotypes/maximized_%s_step_%02d.png'%(save_name, i))

	img = input_img_data[0]
	final_image[0: img_width, 0: img_height] = img

#	plot_wires.plot_wf(final_image,class_index)

	plt.style.use('classic')
	plt.imshow(final_image, aspect='auto')
	plt.savefig('images/stereotypes/maximized_%s.png'%save_name)

	max_element = unravel_index(img.argmax(), img.shape)
	plt.xlim(max_element[2]-40,max_element[2]+40)
	plt.ylim(max_element[1]+15,max_element[1]-15)
	plt.imshow(final_image, aspect='auto')
	plt.savefig('images/stereotypes/maximized_%s_zoomed.png'%save_name)

	plt.close()
	plt.plot(final_image[max_element[1]])
	plt.savefig('images/stereotypes/maximized_%s_wire.png'%save_name)
	plt.close()
	print('Saved image of class %d'%class_index)
	plt.cla()

	end_time = time.time()
	print('Filter %d processed in %d s' % (class_index, end_time-start_time))
