

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import spynnaker8 as p
from ATIS.decode_events import *
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import imageio
from scipy.stats import multivariate_normal
import sys
from connection_matrix import parse_ATIS, combine_parsed_ATIS

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
    x = filter_x * (filter_width - (overlap * filter_width))
    x += (filter_width / 2) + peripheral_x
    y = filter_y * (filter_height - (overlap * filter_height))
    y += (filter_height / 2) + peripheral_y
    return x, y

def extract_spikes():
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

def plot_spikes(file_location, file_name):
    spike_data = np.load("{}/all extracted proto spikes {}.npy".format(file_location, file_name))

    list_data = []
    x = []
    y = []
    z = []
    t = []
    for spike in spike_data:
        # if np.random.random() < 0.01:
        x.append(spike[0])
        y.append(spike[1])
        z.append(1)
        t.append(spike[2])
        list_data.append(spike.tolist())
    list_data.sort(key=lambda x: x[2])

    xlim = 304
    ylim = 240

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(x, y, t)
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    # ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    # ax1.scatter(x, y, z)
    # ax1.get_shared_x_axes().join(ax1, ax)
    # plt.xlim(0, xlim)
    # plt.ylim(0, ylim)
    plt.title('{}'.format(label))
    plt.show()

    print "plotting"

def create_gaussians(filter_size):
    x, y = np.mgrid[-1.0:1.0:complex(0, filter_size), -1.0:1.0:complex(0, filter_size)]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.0, 0.0])
    sigma = np.array([0.3, 0.3])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

    # Reshape back to a grid.
    z = z.reshape(x.shape)
    # Normalise
    z /= z.max()
    # z *= filter_size
    return z

def create_video(file_location, file_name, frame_rate, spikes=[]):
    if spikes:
        spike_data = spikes
    else:
        spike_data = np.load("{}/all extracted proto spikes {}.npy".format(file_location, file_name))
    frame_duration = 1000. / frame_rate

    print "parsing data"
    list_data = []
    x = []
    y = []
    t = []
    max_time = 0.
    for spike in spike_data:
        x.append(spike[0])
        y.append(spike[1])
        t.append(spike[2])
        if spike[2] > max_time:
            max_time = spike[2]
        if not isinstance(spike, list):
            list_data.append(spike.tolist())
        else:
            list_data.append(spike)
    list_data.sort(key=lambda x: x[2])

    filter_gaussians = {}
    for filter_size in filter_sizes:
        filter_gaussians['{}'.format(filter_size)] = create_gaussians(filter_size)

    print "binning frames"
    binned_frames = [[[0 for y in range(y_res)] for x in range(x_res)] for i in range(int(np.ceil(max_time / frame_duration)))]

    # setup toolbar
    toolbar_width = 40
    print '[{}]'.format('-'*toolbar_width)
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    current_point = 0
    progression_count = 0
    for spike in list_data:
        x = int(spike[0])
        y = int(spike[1])
        t = spike[2]
        filter_size = int(spike[3])
        time_index = int(t / frame_duration)
        if filter_sizes[0] == 1:
            binned_frames[time_index][x][y] += 1
        else:
            gaussian = filter_gaussians['{}'.format(filter_size)]
            for i in range(len(gaussian)):
                for j in range(len(gaussian[0])):
                    new_x = x + i - (filter_size / 2)
                    new_y = y + j - (filter_size / 2)
                    if 0 <= new_x < x_res and 0 <= new_y < y_res:
                        binned_frames[time_index][new_x][new_y] += gaussian[i][j]
        progression_count += 1
        # print progression_count, '/', len(list_data)
        if current_point < int(round((float(progression_count) / float(len(list_data))) * float(toolbar_width))):
            current_point = int(round((float(progression_count) / float(len(list_data))) * float(toolbar_width)))
            sys.stdout.write("-")
            sys.stdout.flush()
    sys.stdout.write("]\n")

    print 'creating images'
    xlim = 304
    ylim = 240
    filenames = []
    # setup toolbar
    toolbar_width = 40
    print '[{}]'.format('-'*toolbar_width)
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    current_point = 0
    progression_count = 0
    for frame in binned_frames:
        plt.imshow(frame, cmap='hot', interpolation='nearest')
        title = '{} - {}'.format(file_name, binned_frames.index(frame))
        title += '.jpg'
        filenames.append(file_location+'/videos/'+title)
        plt.savefig(file_location+'/videos/'+title, format='jpeg', bbox_inches='tight')
        # plt.show()
        plt.clf()
        progression_count += 1
        if current_point < int(round((float(progression_count) / float(len(binned_frames))) * float(toolbar_width))):
            current_point = int(round((float(progression_count) / float(len(binned_frames))) * float(toolbar_width)))
            sys.stdout.write("-")
            sys.stdout.flush()
    sys.stdout.write("]\n")

    print "creating video"
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(file_location+'/videos/'+file_name+'.gif', images)

    # with imageio.get_writer('/path/to/movie.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

def parse_events_to_spike_times(events):
    global filter_sizes
    filter_sizes = [1]
    spikes = []
    for neuron in range(len(events)):
        for time in events[neuron]:
            x = neuron % x_res
            y = (neuron - x) / x_res
            spikes.append([x, y, time, 1])
    return spikes

if __name__ == '__main__':
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
    segment_percentage_fire_threshold = 0.04
    filter_percentage_fire_threshold = 0.8
    inhib_percentage_fire_threshold = 0.04
    inhib_connect_prob = 1.
    proto_scale = 0.75
    inhib = False  # [0]: +ve+ve, -ve-ve   [1]:+ve-ve, -ve+ve

    simulate = 'sim_dir'

    label = "{} fs-{} ol-{} w-{} bft-{} sft-{} fft-{} ift-{} icp-{} ps-{} in-{} {}".format(simulate, filter_split, overlap,
                                                                                           base_weight,
                                                                                           boarder_percentage_fire_threshold,
                                                                                           segment_percentage_fire_threshold,
                                                                                           filter_percentage_fire_threshold,
                                                                                           inhib_percentage_fire_threshold,
                                                                                           inhib_connect_prob, proto_scale,
                                                                                           inhib, filter_sizes)
   # plot_spikes("run_data", label)
    contrasts = ['high', 'medium', 'low']
    locations = ['RL', 'LR', 'BT', 'TB']
    combined_events = []
    for contrast in contrasts:
        print contrast, ':'
        for location in locations:
            print "\t", location
            combined_events.append(parse_ATIS('ATIS/{}/{}'.format(contrast, location), 'decoded_events.txt'))
    events = combine_parsed_ATIS(combined_events)
    spikes = parse_events_to_spike_times(events)
    create_video('run_data', 'input spikes', 2, spikes=spikes)
'''
values = [0.08, 0.06, 0.04, 0.02]

for value in values:
    print "current value:", value
    inhib_percentage_fire_threshold = value
    label = "{} fs-{} ol-{} w-{} bft-{} sft-{} fft-{} ift-{} icp-{} ps-{} in-{} {}".format(simulate, filter_split, overlap,
                                                                                           base_weight,
                                                                                           boarder_percentage_fire_threshold,
                                                                                           segment_percentage_fire_threshold,
                                                                                           filter_percentage_fire_threshold,
                                                                                           inhib_percentage_fire_threshold,
                                                                                           inhib_connect_prob, proto_scale,
                                                                                           inhib, filter_sizes)
    create_video("run_data", label, 2)

inhib_percentage_fire_threshold = 0.1
values = [0.08, 0.06, 0.04, 0.02]
for value in values:
    print "current value:", value
    segment_percentage_fire_threshold = value
    label = "{} fs-{} ol-{} w-{} bft-{} sft-{} fft-{} ift-{} icp-{} ps-{} in-{} {}".format(simulate, filter_split, overlap,
                                                                                           base_weight,
                                                                                           boarder_percentage_fire_threshold,
                                                                                           segment_percentage_fire_threshold,
                                                                                           filter_percentage_fire_threshold,
                                                                                           inhib_percentage_fire_threshold,
                                                                                           inhib_connect_prob, proto_scale,
                                                                                           inhib, filter_sizes)
    create_video("run_data", label, 2)
'''
print "done"

'''
proto spikes for testing inhibition
    filter_sizes = [100, 70, 46, 30]
    list_of_filter_sizes = []
    for filter_size in filter_sizes:
        list_of_filter_sizes.append([filter_size, filter_size])
    filter_split = 4
    overlap = 0.6
    base_weight = 5.
    boarder_percentage_fire_threshold = 0.2
    segment_percentage_fire_threshold = 0.2
    filter_percentage_fire_threshold = 0.8
    inhib_percentage_fire_threshold = 0.1, 0.15, 0.2, 0.3, 0.4
    inhib_connect_prob = 1.
    proto_scale = 0.55

label = "fs-{} ol-{} w-{} bft-{} sft-{} fft-{} ift-{} icp-{} ps-{} in-{} {}".format(filter_split, overlap, base_weight,
                                                                                    boarder_percentage_fire_threshold,
                                                                                    segment_percentage_fire_threshold,
                                                                                    filter_percentage_fire_threshold,
                                                                                    inhib_percentage_fire_threshold,
                                                                                    inhib_connect_prob,
                                                                                    proto_scale, inhib, filter_sizes)
    
good spikes for circle detection
    filter_sizes = [100, 70, 46, 30]
    list_of_filter_sizes = []
    for filter_size in filter_sizes:
        list_of_filter_sizes.append([filter_size, filter_size])
    filter_split = 4
    overlap = 0.6
    base_weight = 5.
    boarder_percentage_fire_threshold = 0.2
    segment_percentage_fire_threshold = 0.175, 0.2
    filter_percentage_fire_threshold = 0.8
    inhib_percentage_fire_threshold = 1.
    inhib_connect_prob = 0.1
    proto_scale = 0.75
'''

