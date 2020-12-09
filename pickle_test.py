import pickle
import tables
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# file_name = '/data/mbaxrap7/Heidelberg speech/shd_train.h5'
#
# h5data = tables.open_file(file_name, mode='r')
# units = h5data.root.spikes.units
# times = h5data.root.spikes.times
# labels = h5data.root.labels
#
# fig = plt.figure(figsize=(16, 4))
# idx = np.random.randint(len(times), size=3)
#
# spike_rates = []
# binned_spikes = []
# for time in times:
#     neuron_spikes = [0 for i in range(int(np.max(time)*1000)+1)]
#     for spike in time:
#         bin = int(spike * 1000)
#         neuron_spikes[bin] += 1
#     binned_spikes.append(neuron_spikes)
#     min = np.min(neuron_spikes)
#     max = np.max(neuron_spikes)
#     ave = np.average(neuron_spikes)
#     spike_rates.append([min, ave, max])
#
# for i, k in enumerate(idx):
#     ax = plt.subplot(1, 3, 1 + i)
#     ax.scatter(times[k], 700-units[k], color="k", alpha=0.33, s=2)
#     ax.set_title("Label %i"%labels[k])
#     # ax.axis("off")
#
# plt.show()

# directory = '/localhome/mbaxrap7/spinnicub_python3/SpiNNiCub/pickle_data/'
# file_name = 'color1.pickle'
noise_level = 0.1
directory = '/data/mbaxrap7/white_noise/noise/{}/{}_304x240thr_noise.pickle'.format(noise_level, noise_level)
file_name = ''
infile = open(directory+file_name, 'rb')
new_dict = pickle.load(infile)
infile.close()

# x = new_dict['data']['/cam0/events']['dvs']['x']
# y = new_dict['data']['/cam0/events']['dvs']['y']
# t = new_dict['data']['/cam0/events']['dvs']['ts']

x = new_dict['x']
y = new_dict['y']
t = new_dict['ts']

print("done")


# calib_circles no_obj obj 019 029 085 157 multi_objects_saccade1 object_clutter object_clutter2 objects_approaching objects_approaching_no_saccade paddle_moving_clutter