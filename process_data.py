

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import spynnaker8 as p
from ATIS.decode_events import *
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

warnings.filterwarnings("error")

def create_filter_boundaries(filter_width, filter_height, overlap=0.):
    list_of_corners = []
    i = filter_width + peripheral_x
    x = 0
    while i < x_res - peripheral_x:
        j = filter_height + peripheral_y
        y = 0
        while j < y_res - peripheral_y:
            list_of_corners.append([i, j, x, y])
            j += filter_height - (overlap * filter_height)
            y += 1
        i += filter_width - (overlap * filter_width)
        x += 1
    return list_of_corners

def convert_filter_xy_to_proto_centre(filter_x, filter_y, filter_width, filter_height, overlap):
    # list_of_corners = create_filter_boundaries(filter_width, filter_height, overlap)
    # for corner in list_of_corners:
    #     if corner[2] == filter_x and corner[3] == filter_y:
    #         x = corner[0] - (filter_width / 2)
    #         y = corner[1] - (filter_height / 2)
    #         break

    x = filter_x * (filter_width - (overlap * filter_width))
    x += (filter_width / 2) + peripheral_x
    y = filter_y * (filter_height - (overlap * filter_height))
    y += (filter_height / 2) + peripheral_y

    return x, y

x_res = 304
y_res = 240
fovea_x = 300
fovea_y = 236
peripheral_x = (x_res - fovea_x) / 2
peripheral_y = (y_res - fovea_y) / 2
horizontal_split = 1
veritcal_split = 1

# connection configurations:
filter_sizes = [100, 70, 46, 30]
list_of_filter_sizes = []
for filter_size in filter_sizes:
    list_of_filter_sizes.append([filter_size, filter_size])
filter_split = 4
overlap = 0.6
base_weight = 5.
boarder_percentage_fire_threshold = 0.2
segment_percentage_fire_threshold = 0.2
filter_percentage_fire_threshold = 0.9
inhib_percentage_fire_threshold = 1.
proto_scale = 0.75
inhib = False #[0]: +ve+ve, -ve-ve   [1]:+ve-ve, -ve+ve

label = "fs-{} ol-{} w-{} bft-{} sft-{} fft-{} ps-{} -ve{} {}".format(filter_split, overlap, base_weight,
                                                                      boarder_percentage_fire_threshold,
                                                                      segment_percentage_fire_threshold,
                                                                      filter_percentage_fire_threshold,
                                                                      proto_scale, inhib, filter_sizes)

file_location = "run_data"

extracted_spikes = []
cords_and_times = []

for filter_size in filter_sizes:
    proto_data = np.load("{}/proto object data {}-{}.npy".format(file_location, filter_size, label))

    for proto_object in proto_data:
        split_data = proto_object[1].split('-')
        x_filter = int(split_data[1])
        y_filter = int(split_data[2])
        spike_data = proto_object[0].segments[0].spiketrains
        spikes = 0
        for neuron in spike_data:
            spikes += neuron.size
        if spikes:
            spike_times = spike_data[0].magnitude
            for spike_time in spike_times:
                extracted_spikes.append([x_filter, y_filter, spike_time, filter_size])
                x, y = convert_filter_xy_to_proto_centre(x_filter, y_filter, filter_size, filter_size, overlap)
                cords_and_times.append([x, y, spike_time, filter_size])
            print proto_object[1], cords_and_times[-1]

np.save("spike_data/extracted spikes {}.npy".format(label), cords_and_times)

print "done"


