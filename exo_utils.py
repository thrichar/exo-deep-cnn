import numpy as np
np.random.seed(2)
import h5py
from keras.utils import np_utils

def init_arrays(binary):
	x_images = np.empty((0, 76, 1000), np.float32)
	if binary == False:
		y_labels = np.empty((0, 6), np.float32)
	else:
		y_labels = np.empty((0, 2), np.float32)
	y_recon = np.empty((0), np.float32)

	return (x_images, y_labels, y_recon)

def normalize(x_images):
	# Not using this function normally.
	# Events are already zero-centered, so this next line can remain commented.
#	x_images -= np.mean(x_images, axis=0)

	x_images /= np.std(x_images, axis=0)

	return x_images

def get_type(filename, binary):
	# Find type of image based on filename
	if binary == False:
		if "bb0n" in filename:
			ntype = 0
		elif "bb2n" in filename:
			ntype = 1
		elif "Xe137" in filename:
			ntype = 2
		elif "U238" in filename or "Ra226" in filename:
			ntype = 3
		elif "Th232" in filename or "Th228" in filename:
			ntype = 4
		elif "Co60" in filename or "Co060" in filename:
			ntype = 5
		else:
			raise ValueError("MC type for file not recognized.")

	else:
		if "bb0n" in filename:
			ntype = 0
		elif "bb2n" in filename:
			ntype = 1
		elif "Xe137" in filename:
			ntype = 1
		elif "U238" in filename or "Ra226" in filename:
			ntype = 1
		elif "Th232" in filename or "Th228" in filename:
			ntype = 1
		elif "Co60" in filename or "Co060" in filename:
			ntype = 1
		else:
			raise ValueError("MC type for file not recognized.")

	return ntype

def read_images(filename, onehot, binary):
	with h5py.File(filename, "r") as h5file:
#		print h5file.keys()

		# Data now uses same scheme
		x_images = np.asarray(h5file.get("wfimages"))
		y_recon = np.asarray(h5file.get("recon_energy"))
#		y_recon = np.asarray(h5file.get("true_energy"))

	# Create simple (not one hot) array for labels
	ntype = get_type(filename, binary)
	y_labels = np.array([ntype for i in x_images], np.int)

	# Use the Keras function to make one-hot vectors.
	# The confusion matrix library doesn't use one-hot vectors,
	# so it's better to keep this optional.		
	if onehot == True:
		if binary == False:
			y_labels = np_utils.to_categorical(y_labels, 6)
		else:
			y_labels = np_utils.to_categorical(y_labels, 2)

	return (x_images, y_labels, y_recon)

def energy_cut(x_images, y_labels, y_recon, cut, full):
	# Data events had to be cut, but already done now.
	# Currently using 2100 keV
	nevent = 0
	for i in range(len(y_recon)):
		if y_recon[i] >= float(cut):
			if full:
				x_images[nevent] = x_images[i]
			y_recon[nevent] = y_recon[i]
			nevent += 1

	if full:
		x_images = x_images[:nevent]
		y_labels = y_labels[:nevent]
	y_recon = y_recon[:nevent]

	if full:
		x_images = x_images.reshape(x_images.shape[0], 1, 76, 1000)
		x_images = x_images.astype("float32")

	if full:
		return (x_images, y_labels, y_recon)
	else:
		return y_recon

def get_nevents(files):
	nevents = 0
	for filename in files:
		with h5py.File(filename, "r") as h5file:
			n = len(np.asarray(h5file.get("recon_energy")))
		nevents += n
	return nevents

class generator_object(object):
	# It's a class since I have to use this index variable.
	def __init__(self):
		self.index = 0

	def generate_image(self, files, onehot, binary):	
		nfiles = 12
		while True:
			if self.index + nfiles > len(files):
				self.index = 0

			x_images, y_labels, y_recon = init_arrays(binary)

			for filename in files[self.index*nfiles:(self.index+1)*nfiles]:
				x_images_temp, y_labels_temp, y_recon_temp = read_images(filename, onehot, binary)
				x_images = np.concatenate((x_images_temp, x_images), axis=0)
				y_labels = np.concatenate((y_labels_temp, y_labels), axis=0)
				y_recon = np.concatenate((y_recon_temp, y_recon), axis=0)
#				print("\nIndex: %d\n%s\n%s\n"%(self.index, str(y_labels_temp[0]), filename))
			self.index += 1

			# Shuffle order of events
			# Network doesn't train otherwise
			nlist = range(len(y_recon))
			np.random.shuffle(nlist)
			for i in nlist:
				x_images_ = x_images[i]
				y_labels_ = y_labels[i]
				y_recon_ = y_recon[i]
				# Give one image at a time to the batch generator
				yield (x_images_, y_labels_, y_recon_)

def generate_batch(generator, batch_size, binary, d):
	while True:
		x_images, y_labels, y_recon = init_arrays(binary)
		for i in range(batch_size):
			# Take one image at a time from the image generator.
			# Concatenate images to make a batch.
			temp = generator.next()
			temp = np.asarray(temp)
			temp[0] = temp[0].reshape(1, 76, 1000)
			if binary == False:
				temp[1] = temp[1].reshape(1, 6)
			else:
				temp[1] = temp[1].reshape(1, 2)
			temp[2] = temp[2].reshape(1)
			x_images = np.concatenate((temp[0], x_images), axis=0)
			y_labels = np.concatenate((temp[1], y_labels), axis=0)
			y_recon = np.concatenate((temp[2], y_recon), axis=0)
	
		if d == True:
			# Explicitly declare color dimension
			x_images = x_images.reshape(x_images.shape[0], 1, 76, 1000)

		x_images = x_images.astype("float32")

		yield (x_images, y_labels)

def get_val_data(files, binary):
	x_images, y_labels, y_recon = init_arrays(binary)
	# Just load it all at once
	for filename in files:
		x_images_temp, y_labels_temp, y_recon_temp = read_images(filename, True, binary)
		x_images = np.concatenate((x_images_temp, x_images), axis=0)
		y_labels = np.concatenate((y_labels_temp, y_labels), axis=0)
		y_recon = np.concatenate((y_recon_temp, y_recon), axis=0)
		print filename

	# Explicitly declare color dimension
	x_images = x_images.reshape(x_images.shape[0], 1, 76, 1000)
	x_images = x_images.astype("float32")

	return (x_images, y_labels)

def compute_class_weights(y_labels):
	nevent = []
	ntotal = 0

	# Compute how many events of each class
	for i in range(len(y_labels)):
		maxelem = np.argmax(y_labels[i])
		nevent[maxelem] += 1

	for i in range(len(nevent)):
		ntotal += nevent[i]

	# Compute weights based on sklearn's class weights function.
	# I would use that directly, but it wasn't working for some reason.
	wevent = []
	for i in range(len(nevent)):
		wevent[i] = float(ntotal) / ( 6. * float(nevent[i]) )
		
	return wevent
