import numpy as np
np.random.seed(123)

mdir = "/home/shared_dir/MCImages_Ecut_UWiresOnly_GainCorrected_Smeared/txts/"

def get_train_files(binary):
	bb0n_train_txts = mdir+"bb0n_train.txt"
	bb2n_train_txts = mdir+"bb2n_train.txt"
	Xe137_train_txts = mdir+"ActiveLXe_Xe137_train.txt"
	U238_train_txts = mdir+"AllVessel_U238_train.txt"
	Th232_train_txts = mdir+"AllVessel_Th232_train.txt"
	Th228_train_txts = mdir+"Th228_train.txt" # This one is data, preprocessed for training
	Co60_train_txts = mdir+"AllVessel_Co60_train.txt"

	# Go through all the MC types and compile lists of files
	# I don't know if it's necessary to shuffle them,
	# but it may help hide patterns for the network.
	# It can't hurt.

	with open(bb0n_train_txts) as f:
		bb0n_train_files = f.readlines()
	bb0n_train_files = [x.strip() for x in bb0n_train_files]
#	np.random.shuffle(bb0n_train_files)

	with open(bb2n_train_txts) as f:
		bb2n_train_files = f.readlines()
	bb2n_train_files = [x.strip() for x in bb2n_train_files]
#	np.random.shuffle(bb2n_train_files)

	with open(Xe137_train_txts) as f:
		Xe137_train_files = f.readlines()
	Xe137_train_files = [x.strip() for x in Xe137_train_files]
#	np.random.shuffle(Xe137_train_files)

	with open(U238_train_txts) as f:
		U238_train_files = f.readlines()
	U238_train_files = [x.strip() for x in U238_train_files]
#	np.random.shuffle(U238_train_files)

	with open(Th232_train_txts) as f:
		Th232_train_files = f.readlines()
	Th232_train_files = [x.strip() for x in Th232_train_files]
#	np.random.shuffle(Th232_train_files)

	with open(Th228_train_txts) as f:
		Th228_train_files = f.readlines()
	Th228_train_files = [x.strip() for x in Th228_train_files]
#	np.random.shuffle(Th232_train_files)

	with open(Co60_train_txts) as f:
		Co60_train_files = f.readlines()
	Co60_train_files = [x.strip() for x in Co60_train_files]
#	np.random.shuffle(Co60_train_files)

	# Find set with largest number of files
	nfiles = max(len(bb0n_train_files), len(bb2n_train_files), len(Xe137_train_files), len(U238_train_files), len(Th232_train_files), len(Th228_train_files), len(Co60_train_files)

	# Create list of all the files, alternating across all the types
	train_files = []

	nbb0n = 0
	nbb2n = 0
	nXe137 = 0
	nU238 = 0
	nTh232 = 0
	nTh228 = 0
	nCo60 = 0

	for i in range(nfiles):
		try:
			train_files.append(bb0n_train_files[nbb0n])
			nbb0n += 1
		except:
			# Runs out of files
			nbb0n = 0
			train_files.append(bb0n_train_files[nbb0n])

		if binary == False:
			try:
				train_files.append(bb2n_train_files[nbb2n])
				nbb2n += 1
			except:
				nbb2n = 0
				train_files.append(bb2n_train_files[nbb2n])

		if binary == False:
			try:
				train_files.append(Xe137_train_files[nXe137])
				nXe137 += 1
			except:
				nXe137 = 0
				train_files.append(Xe137_train_files[nXe137])

		try:
			train_files.append(U238_train_files[nU238])
			nU238 += 1
		except:
			nU238 = 0
			train_files.append(U238_train_files[nU238])

		try:
			train_files.append(Th232_train_files[nTh232])
			nTh232 += 1
		except:
			# Shouldn't run out of Th232 files since it has the most files
			# UPDATE: Can't be sure about that anymore.
#			raise ValueError("Ran out of Th232 files.")
			nTh232 = 0
			train_files.append(Th232_train_files[nTh232])

		try:
			train_files.append(Th228_train_files[nTh228])
			nTh228 += 1
		except:
			nTh228 = 0
			train_files.append(Th228_train_files[nTh228])

		if binary == False:
			try:
				train_files.append(Co60_train_files[nCo60])
				nCo60 += 1
			except:
				nCo60 = 0
				train_files.append(Co60_train_files[nCo60])

	return train_files

def get_val_files(binary):
	# Basically the same thing as the training files,
	# but the order of events and files doesn't matter.
	if binary == False:
		val_txts = [mdir+"bb0n_val.txt", mdir+"bb2n_val.txt",
		mdir+"ActiveLXe_Xe137_val.txt", mdir+"AllVessel_U238_val.txt",
		mdir+"AllVessel_Th232_val.txt", mdir+"Th228_data_val.txt",
		mdir+"AllVessel_Co60_val.txt"]

	else:
		val_txts = [mdir+"bb0n_val.txt", mdir+"AllVessel_U238_val.txt",
		mdir+"AllVessel_Th232_val.txt"]

	val_files = []
	for val_txt in val_txts:
		with open(val_txt) as f:
			temp_files = f.readlines()
		val_files.extend(temp_files)
	val_files = [x.strip() for x in val_files]

	return val_files

def get_data_files():
	# Basically the same thing as the training files,
	# but the order of events and files doesn't matter.
	data_txts = [mdir+"Ra226_data.txt",
	mdir+"Th228_data.txt", mdir+"Co60_data.txt"]

	data_files = []
	for data_txt in data_txts:
		with open(data_txt) as f:
			temp_files = f.readlines()
		data_files.extend(temp_files)
	data_files = [x.strip() for x in data_files]

	return data_files
