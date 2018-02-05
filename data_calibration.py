import numpy as np
from numpy import genfromtxt
np.random.seed(123)
from keras.models import model_from_json
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import h5py
from exo_utils import init_arrays, read_images, energy_cut

binary = True

net = "nets/binary_classifier_2018-01-23"

try:
	json_file = open("%s.json"%net,"r")
	loaded_model_json = json_file.read()
	json_file.close()
	train_model = model_from_json(loaded_model_json)
	train_model.load_weights("%s.h5"%net)
	print("Loaded model from disk.")
except:
	print("Error loading model.")
	exit()

datatypes = ["Th228", "Ra226", "Co60"]

i = 0
for datatype in datatypes:
	files[i] = "/home/shared_dir/MCImages_Ecut_UWiresOnly_GainCorrected_Smeared/%sWFs_Data_SS+MS_S5_Final_Cut/%s.hdf5"%(datatype, datatype)
	i += 1

i = 0
for filename in files:
	print("Loading %s"%filename)
	x_images, y_labels, y_recon = read_images(filename, True, binary)

	datatype = datatypes[i]
	i += 1

	train_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
	score = train_model.evaluate(x_images, y_labels, verbose=0)
	print("%s calibration loss: "%datatype, score[0])
	print("%s calibration accuracy: "datatype, score[1])
