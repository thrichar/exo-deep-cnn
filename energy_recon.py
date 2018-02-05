import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exo_utils import read_images, normalize

x_co, y_co, r_co = read_images("/home/shared_dir/MCImages_Ecut_UWiresOnly_GainCorrected_Smeared/Th228WFs_Data_SS+MS_S5_Final_Cut/Th228_000.hdf5", True, True)
#x_co, y_co, r_co = read_images("/home/shared_dir/MCImages_Ecut_UWiresOnly_GainCorrected_Smeared/Co60WFs_Data_SS+MS_S5_Final_Cut/Co60.hdf5", True, True)

#x_co = normalize(x_co)

pulses0 = np.empty((len(y_co)*76), np.float32)
i = 0

for image in range(len(y_co)):
	energy = 0.
	for wire in range(76):
		current_wire = x_co[image][wire]
		pulse = current_wire.max()
		energy += pulse
		pulses0[i] = pulse
		i += 1
#	r_co[image] = energy

print("Loaded %d events."%len(y_co))
plt.hist(r_co, bins="auto", normed=True, color=(1., 0., 0., 0.5))

del x_co, y_co, r_co

x_mc = np.empty((0, 76, 1000), np.float32)
y_mc = np.empty((0), np.float32)
r_mc = np.empty((0), np.float32)

mctype = "AllVessel_Th232"

for i in range(20):
#	filename = "/home/shared_dir/MCImages_Ecut_UWiresOnly_GainCorrected_Smeared/%s/ED_Source_%s_%03d.hdf5"%(mctype, mctype, i)
	filename = "/home/shared_dir/MCImages_Ecut_UWiresOnly_GainCorrected_Smeared/test/ED_Source_%s_%03d.hdf5"%(mctype, i)
	try:
		x_mc_t, y_mc_t, r_mc_t = read_images(filename, True, True)
	except:
		continue
	x_mc = np.concatenate((x_mc_t, x_mc), axis=0)
	r_mc = np.concatenate((r_mc_t, r_mc), axis=0)
	print("%s contains %d events."%(filename, len(r_mc_t)))

nevents = len(r_mc)
print("Loaded %d events."%nevents)

#x_mc = normalize(x_mc)

pulses1 = np.empty((nevents*76), np.float32)
i = 0

for image in range(nevents):
	energy = 0.
	for wire in range(76):
		current_wire = x_mc[image][wire]
		pulse = current_wire.max()
		energy += pulse
		pulses1[i] = pulse
		i += 1
	print energy
#	r_mc[image] = energy

plt.hist(r_mc, bins="auto", normed=True, color=(0., 0., 1., 0.5))
plt.savefig("images/rudimentary_recon_dual.png")

plt.close()

plt.yscale("log", nonposy="clip")
plt.hist(pulses0, bins="auto", normed=True, color=(1., 0., 0., 0.5))
plt.hist(pulses1, bins="auto", normed=True, color=(0., 0., 1., 0.5))
plt.savefig("images/pulse_dual.png")

plt.close()

for i in range(10):
	plt.style.use("classic")
	plt.imshow(x_mc[i], aspect="auto")
	plt.savefig("images/events/event_%02d.png"%i)
	plt.close()
	print i
