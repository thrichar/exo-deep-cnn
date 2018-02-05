import numpy as np

from datetime import date

from keras.models import model_from_json
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('th')

from exo_utils import get_nevents, generator_object, generate_batch, get_val_data, compute_class_weights
from file_utils import get_train_files, get_val_files
from convolutional_models import train_model

binary = True
batch_size = 100
nepochs = 10

# Actual events aren't loaded here, just filenames
print("Loading training files.")
train_files = get_train_files(binary)
nevents = get_nevents(train_files, True)
print("Training with %d events."%nevents)

# Validation events are loaded here, not just filenames
print("Loading validation files.")
val_files = get_val_files(binary)
x_val, y_val = get_val_data(val_files, binary)
print("Validating with %d events."%len(y_val))

# Generators are used for training data
generator = generator_object()
image_generator = generator.generate_image(train_files, True, binary)
batch_generator = generate_batch(image_generator, batch_size, binary, True)

# Classes are pretty imbalanced, so set weights
wevent = compute_class_weights(y_val) 
class_weights = {
		0: wevent[0],
		1: wevent[1],
		2: wevent[2],
		3: wevent[3],
		4: wevent[4],
		5: wevent[5]	}
print("Using class weights:")
print class_weights

today = date.today()
print("Saving files with the date %s."%today)

if binary == False:
	netname = "all_classifier_%s"%today
else:
	netname = "binary_classifier_%s"%today

csv_logger = CSVLogger("csvs/training_log_%s.csv"%today, separator=',', append=False)

# Save the net after each epoch, but only if val_loss is greater than all previous values
model_checkpoint = ModelCheckpoint("nets/tmp/%s.h5"%netname, verbose=1, save_best_only=True)

# Create new model
model = train_model(binary)
print("Creating new model.")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit_generator(generator = batch_generator,
										steps_per_epoch = nevents/batch_size,
										nb_epoch = nepochs,
										class_weight = class_weights,
										validation_data = (x_val, y_val),
										callbacks = [csv_logger, model_checkpoint],
										verbose = 1)

# Save trained network to disk
model_json = model.to_json()
with open("nets/%s.json"%netname,"w") as json_file:
	json_file.write(model_json)
model.save_weights("nets/%s.h5"%netname)
print("Saved model to disk.")
