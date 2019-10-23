#!usr/bin/python

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

'''
Components:
- create the filter
- downsample filter for various resolutions
- pass filter into a single neuron for part of proto-object
- combine parts of the proto-objects into a object/neuron
- inhibit between competing objects
'''

def zero_2pi_tan(x, y, theta=0):
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += np.pi * 2
    if angle > 6 * np.pi / 4:
        angle -= 6 * np.pi / 4
    else:
        angle += np.pi / 2
    return angle #+ theta

# Von Mises filter function, takes values between -1 and 1 and outputs the corresponding value
def VM(x, y, r=10, p=0.08, theta=0, threshold=0.75, filter_split=7):
    theta *= np.pi/4
    # rescaling of x and y for correct filter placement
    x = (x*15) + (0.8*r) * np.cos(theta)
    y = (y*15) - (0.8*r) * np.sin(theta)
    # slit the value for the filter
    angle = zero_2pi_tan(x, y, theta)
    if (angle >= np.pi - theta and angle <= (2*np.pi) - theta):
        split = -1
    else:
        split = int((angle + theta) / (np.pi / float(filter_split))) % filter_split
    numerator = np.exp(p * r * np.cos(np.arctan2(-y, x) - theta))
    denominator = np.i0(np.sqrt(np.square(x) + np.square(y)) - r)
    if denominator == 0:
        denominator = 0.01
    VM_output = numerator / denominator
    # threshold to get rid of small tail
    if VM_output < threshold:
        VM_output = 0
    else:
        VM_output = 1#-= threshold
    return VM_output, split

# depricated
# Creates a matrix of values for a particular resolution and rotation of the VM filter
def generate_matrix(resolution=46, rotation=0, plot=False):
    rotation_matrix = []
    banana_matrix = []
    xs = []
    ys = []
    for i in range(-resolution, resolution):
        x = []
        y = []
        banana_row = []
        rotation_row = []
        for j in range(-resolution, resolution):
            x.append(float(i) / float(resolution))
            y.append(float(j) / float(resolution))
            banan, split = VM(float(i)/float(resolution), float(j)/float(resolution), theta=rotation)
            banana_row.append(round(banan, 2))
            rotation_row.append(split)
        xs.append(x)
        ys.append(y)
        banana_matrix.append(banana_row)
        rotation_matrix.append(rotation_row)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(np.array(xs), np.array(ys), np.array(banana_matrix))
        ax.plot_wireframe(np.array(xs), np.array(ys), np.array(rotation_matrix))
        plt.show()
    return banana_matrix

def convert_pixel_to_id(x, y):
    return (y*304) + x

#depricated
def generate_visual_field(no_filter_x, no_filter_y, filter_split=4, rotation=0, plot=False):
    # create inputs of -1 to 1 for entire visual field, assuming no overlap for now
    filter_x_res = x_res / no_filter_x
    filter_y_res = y_res / no_filter_y
    connection_list = []
    visual_matrix = []
    xs = []
    ys = []
    for i in range(x_res):
        visual_row = []
        x = []
        y = []
        for j in range(y_res):
            x_value = (float(i % filter_x_res) / float(filter_x_res / 2)) - 1.
            y_value = (float(j % filter_y_res) / float(filter_y_res / 2)) - 1.
            pixel_value, split = VM(x_value, y_value, theta=rotation, filter_split=filter_split)
            if pixel_value > 0 and split >= 0:
                neuron_index = int(y_res / filter_y_res) + (int(x_res / filter_x_res) * no_filter_y) + filter_split
                filter_identifier = [int(y_res / filter_y_res), int(x_res / filter_x_res), filter_split, rotation]
                connection_list.append([convert_pixel_to_id(i, j), neuron_index, filter_identifier])
            x.append(i)
            y.append(j)
            visual_row.append(pixel_value)
        visual_matrix.append(visual_row)
        xs.append(x)
        ys.append(y)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(np.array(xs), np.array(ys), np.array(visual_matrix))
        plt.show()
    return visual_matrix

def create_filter_boundaries(filter_width, filter_height, overlap=0.):
    list_of_corners = []
    i = filter_width
    x = 0
    while i < x_res:
        j = filter_height
        y = 0
        while j < y_res:
            list_of_corners.append([i, j, x, y])
            j += filter_height - (overlap * filter_height)
            y += 1
        i += filter_width - (overlap * filter_width)
        x += 1
    return list_of_corners

def visual_field_with_overlap(filter_width, filter_height, overlap=0., filter_split=4, rotation=0,
                              base_weight=1., percentage_fire_threshold=0.5, plot=False):
    max_filters_x = int(x_res / (filter_width * (1-overlap)))
    max_filters_y = int(y_res / (filter_height * (1-overlap)))

    # create the filter which is to be copied in a grid like fashion around the visual field
    filter_split_matrix = []
    filter_matrix = []
    xs = []
    ys = []
    for i in range(filter_width):
        split_row = []
        filter_row = []
        x = []
        y = []
        for j in range(filter_height):
            x_value = ((float(i) / float(filter_width)) * 2) - 1
            y_value = ((float(j) / float(filter_height)) * 2) - 1
            pixel_value, split = VM(x_value, y_value, theta=rotation, filter_split=filter_split)
            x.append(x_value)
            y.append(y_value)
            filter_row.append(pixel_value)
            split_row.append(split)
        filter_split_matrix.append(split_row)
        filter_matrix.append(filter_row)
        xs.append(x)
        ys.append(y)

    # calculate all filter corners for later processing
    corners_list = create_filter_boundaries(filter_width, filter_height, overlap)

    # copy the filter dimensions around the board
    visual_matrix = []
    split_matrix = []
    xs = []
    ys = []
    connection_list = []
    neuron_id_count = [0 for i in range(max_filters_x * max_filters_y * filter_split)]
    for corner in corners_list:
        for filter_x in range(len(filter_matrix)):
            x = []
            y = []
            visual_row = []
            split_row = []
            for filter_y in range(len(filter_matrix[0])):
                x_offset = corner[0] - filter_width
                y_offset = corner[1] - filter_height
                x.append(x_offset + filter_x)
                y.append(y_offset + filter_y)
                visual_row.append(filter_matrix[filter_x][filter_y])
                split_row.append(filter_split_matrix[filter_x][filter_y])
                if filter_matrix[filter_x][filter_y] and filter_split_matrix[filter_x][filter_y] >= 0:
                    pixel_value = convert_pixel_to_id(x_offset + filter_x, y_offset + filter_y)
                    split_value = filter_split_matrix[filter_x][filter_y]
                    neuron_id = (corner[2] * filter_split) + (max_filters_x * corner[3] * filter_split) + split_value
                    # print neuron_id
                    neuron_id_count[neuron_id] += 1
                    connection_list.append([pixel_value, neuron_id, base_weight, 1])
            xs.append(x)
            ys.append(y)
            visual_matrix.append(visual_row)
            split_matrix.append(split_row)

    # scale weights
    for connection in range(len(connection_list)):
        connection_list[connection][2] /= neuron_id_count[connection_list[connection][1]] * percentage_fire_threshold

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(np.array(xs), np.array(ys), np.array(visual_matrix))
        plt.show()
    return connection_list, len(neuron_id_count)

def create_filter_neuron_connections(filter_split, no_neurons, base_weight):
    connections = []
    for i in range(no_neurons):
        connections.append([i, int(i/filter_split), base_weight/filter_split, 1])
    return connections

def proto_objects(on_populations, off_populations, filter_width, filter_height, base_weight):
    max_filters_x = int(x_res / (filter_width * (1-overlap)))
    max_filters_y = int(y_res / (filter_height * (1-overlap)))
    proto_object_neurons = []
    # create the connections and populations
    for i in range(max_filters_x):
        for j in range(max_filters_y):
            n1 = (i) + (j * max_filters_x)
            if j + 1 < max_filters_y:
                n2 = (i) + ((j+1) * max_filters_x)
                proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                         label='{}-{}-up'.format(i, j+1)))
                p.Projection(on_populations[6], proto_object_neurons[-1], p.FromListConnector([[n2, 1, base_weight/2., 1]]))
                p.Projection(off_populations[2], proto_object_neurons[-1], p.FromListConnector([[n1, 1, base_weight/2., 1]]))
                proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                         label='{}-{}-down'.format(i, j+1)))
                p.Projection(on_populations[2], proto_object_neurons[-1], p.FromListConnector([[n1, 1, base_weight/2., 1]]))
                p.Projection(off_populations[6], proto_object_neurons[-1], p.FromListConnector([[n2, 1, base_weight/2., 1]]))
                if i - 1 >= 0:
                    n2 = (i-1) + ((j+1) * max_filters_x)
                    proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                             label='{}-{}-upl'.format(i, j+1)))
                    p.Projection(on_populations[5], proto_object_neurons[-1], p.FromListConnector([[n2, 1, base_weight/2., 1]]))
                    p.Projection(off_populations[1], proto_object_neurons[-1], p.FromListConnector([[n1, 1, base_weight/2., 1]]))
                    proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                             label='{}-{}-downr'.format(i, j+1)))
                    p.Projection(on_populations[1], proto_object_neurons[-1], p.FromListConnector([[n1, 1, base_weight/2., 1]]))
                    p.Projection(off_populations[5], proto_object_neurons[-1], p.FromListConnector([[n2, 1, base_weight/2., 1]]))
            if i + 1 < max_filters_x:
                n2 = (i+1) + ((j) * max_filters_x)
                proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                         label='{}-{}-l'.format(i, j+1)))
                p.Projection(on_populations[0], proto_object_neurons[-1], p.FromListConnector([[n2, 1, base_weight/2., 1]]))
                p.Projection(off_populations[4], proto_object_neurons[-1], p.FromListConnector([[n1, 1, base_weight/2., 1]]))
                proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                         label='{}-{}-r'.format(i, j+1)))
                p.Projection(on_populations[4], proto_object_neurons[-1], p.FromListConnector([[n1, 1, base_weight/2., 1]]))
                p.Projection(off_populations[0], proto_object_neurons[-1], p.FromListConnector([[n2, 1, base_weight/2., 1]]))
                if j + 1 < max_filters_y:
                    n2 = (i+1) + ((j+1) * max_filters_x)
                    proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                             label='{}-{}-upr'.format(i, j+1)))
                    p.Projection(on_populations[7], proto_object_neurons[-1], p.FromListConnector([[n2, 1, base_weight/2., 1]]))
                    p.Projection(off_populations[3], proto_object_neurons[-1], p.FromListConnector([[n1, 1, base_weight/2., 1]]))
                    proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                             label='{}-{}-downl'.format(i, j+1)))
                    p.Projection(on_populations[3], proto_object_neurons[-1], p.FromListConnector([[n1, 1, base_weight/2., 1]]))
                    p.Projection(off_populations[7], proto_object_neurons[-1], p.FromListConnector([[n2, 1, base_weight/2., 1]]))
    return proto_object_neurons

# depricated
def create_proto_object_connections(on_populations, off_populations, filter_width, filter_height, base_weight):
    max_filters_x = int(x_res / (filter_width * (1 - overlap)))
    max_filters_y = int(y_res / (filter_height * (1 - overlap)))
    horizontal_connections = []
    veritcal_connections = []
    bot_left_connections = []
    top_left_connections = []
    # create the connections
    for i in range(max_filters_x):
        for j in range(max_filters_y):
            n1 = (i) + (j * max_filters_x)
            if j + 1 < max_filters_y:
                n2 = (i) + ((j + 1) * max_filters_x)
                veritcal_connections.append([n1, n2, base_weight / 2., 1])
                veritcal_connections.append([n2, n1, base_weight / 2., 1])
                if i - 1 >= 0:
                    n2 = (i - 1) + ((j + 1) * max_filters_x)
                    top_left_connections.append([n1, n2, base_weight / 2., 1])
                    top_left_connections.append([n2, n1, base_weight / 2., 1])
            if i + 1 < max_filters_x:
                n2 = (i + 1) + ((j) * max_filters_x)
                horizontal_connections.append([n1, n2, base_weight / 2., 1])
                horizontal_connections.append([n2, n1, base_weight / 2., 1])
                if j + 1 < max_filters_y:
                    n2 = (i + 1) + ((j + 1) * max_filters_x)
                    bot_left_connections.append([n1, n2, base_weight / 2., 1])
                    bot_left_connections.append([n2, n1, base_weight / 2., 1])
    # connect them up
    for rotation in range(8):
        p.Projection(on_population, off_population, p.FromListConnector(horizontal_connections))
        # and the rest
    print "done"

def parse_ATIS(file_location, file_name):
    f = open("{}/{}".format(file_location, file_name), "r")
    on_events = [[] for i in range(x_res*y_res)]
    off_events = [[] for i in range(x_res*y_res)]
    for line in f:
        line = np.array(line.split(','))
        if float(line[0]) < 0:
            time = (float(line[0]) + 87.) * 1000.
            # print time, "n"
        else:
            time = float(line[0]) * 1000.
            # print time
        if line[4]:
            on_events[convert_pixel_to_id(int(line[2]), int(line[3]))].append(time)
        else:
            off_events[convert_pixel_to_id(int(line[2]), int(line[3]))].append(time)
    return on_events, off_events

def test_orientation(filter_width, filter_height, overlap=0., filter_split=4, rotation=0,
                              base_weight=1., percentage_fire_threshold=0.5, plot=False):
    max_filters_x = int(x_res / (filter_width * (1-overlap)))
    max_filters_y = int(y_res / (filter_height * (1-overlap)))

    # create the filter which is to be copied in a grid like fashion around the visual field
    filter_matrix = []
    rotated_matrix = []
    xs = []
    ys = []
    for i in range(filter_width):
        filter_row = []
        rotate_row = []
        x = []
        y = []
        for j in range(filter_height):
            x_value = ((float(i) / float(filter_width)) * 2) - 1
            y_value = ((float(j) / float(filter_height)) * 2) - 1
            pixel_value, split = VM(x_value, y_value, theta=rotation, filter_split=filter_split)
            rotate_value, split = VM(x_value, y_value, theta=rotation+4, filter_split=filter_split)
            x.append(x_value)
            y.append(y_value)
            filter_row.append(pixel_value)
            rotate_row.append(rotate_value)
        filter_matrix.append(filter_row)
        rotated_matrix.append(rotate_row)
        xs.append(x)
        ys.append(y)

    # calculate all filter corners for later processing
    corners_list = [[filter_width, filter_height]] #create_filter_boundaries(filter_width, filter_height, overlap)

    # copy the filter dimensions around the board
    visual_matrix = []
    visual_rotate = []
    rotate_x = -filter_width
    rotate_y = filter_height
    xs = []
    ys = []
    xsr = []
    ysr = []
    for corner in corners_list:
        for filter_x in range(len(filter_matrix)):
            x = []
            y = []
            xr = []
            yr = []
            visual_row = []
            rotate_row = []
            for filter_y in range(len(filter_matrix[0])):
                x_offset = corner[0] - filter_width
                y_offset = corner[1] - filter_height
                x.append(x_offset + filter_x)
                y.append(y_offset + filter_y)
                xr.append(x_offset + filter_x + rotate_x)
                yr.append(y_offset + filter_y + rotate_y)
                visual_row.append(filter_matrix[filter_x][filter_y])
                rotate_row.append(rotated_matrix[filter_x][filter_y])
                # if filter_matrix[filter_x][filter_y] and filter_split_matrix[filter_x][filter_y] >= 0:
                #     pixel_value = convert_pixel_to_id(x_offset + filter_x, y_offset + filter_y)
                #     split_value = filter_split_matrix[filter_x][filter_y]
                #     neuron_id = (corner[2] * filter_split) + (max_filters_x * corner[3] * filter_split) + split_value
                #     # print neuron_id
                #     neuron_id_count[neuron_id] += 1
            xs.append(x)
            ys.append(y)
            xsr.append(xr)
            ysr.append(yr)
            visual_matrix.append(visual_row)
            visual_rotate.append(rotate_row)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(np.array(xs), np.array(ys), np.array(visual_matrix))
        ax.plot_wireframe(np.array(xsr), np.array(ysr), np.array(visual_rotate))
        plt.show()
    return None

# generate_visual_field(3, 2, rotation=1, plot=True)
# first_layer_connections_r0 = visual_field_with_overlap(100, 100, 0., plot=False)

x_res = 304
y_res = 240

# test_orientation(100, 100, rotation=0, plot=True)
# test_orientation(100, 100, rotation=1, plot=True)
# test_orientation(100, 100, rotation=2, plot=True)
# test_orientation(100, 100, rotation=3, plot=True)

neuron_params = {
    # balance the refractory period/tau_mem so membrane has lost contribution before next spike
}
# connection configurations:
list_of_filter_sizes = [
    [46, 46],
    [70, 70],
    [100, 100]]
filter_split = 4
overlap = 0.6
base_weight = 5
percentage_fire_threshold = 0.8

# extract input data
# dm = DataManager()
# dm.load_AE_from_yarp('ATIS')
on_events, off_events = parse_ATIS('ATIS/data_surprise', 'decoded_events.txt')
p.setup(timestep=1.0)
on_population = p.Population(x_res*y_res, p.SpikeSourceArray(on_events), label='on_events')
off_population = p.Population(x_res*y_res, p.SpikeSourceArray(off_events), label='off_events')

# create first layer populations and connections from input video stream
connections_icub_to_neurons = []
first_layer_populations = [[], []] # off and on
for filter in list_of_filter_sizes:
    print "SpiNN setup for filter", filter
    # for each rotation, 0 -> 7pi/4
    on_filter_segments = []
    off_filter_segments = []
    on_filter_populations = []
    off_filter_populations = []
    for rotation in range(8):
        print "Rotation", rotation+1, "/ 8"
        connection, no_neurons = visual_field_with_overlap(filter[0], filter[1],
                                                           overlap=overlap,
                                                           rotation=rotation,
                                                           filter_split=filter_split,
                                                           base_weight=base_weight/float(filter_split),
                                                           percentage_fire_threshold=percentage_fire_threshold)
        # create neurons for each segment of filter
        # on_filter_segments = p.Population(no_neurons, p.SpikeSourcePoisson(rate=100),
        #                                   label='on seg {} - {}'.format(filter, rotation))
        # off_filter_segments = p.Population(no_neurons, p.SpikeSourcePoisson(rate=100),
        #                                    label='off seg {} - {}'.format(filter, rotation))
        on_filter_segments.append(p.Population(no_neurons, p.IF_curr_exp(*neuron_params),
                                               label='on seg {} - {}'.format(filter, rotation)))
        off_filter_segments.append(p.Population(no_neurons, p.IF_curr_exp(*neuron_params),
                                                label='off seg {} - {}'.format(filter, rotation)))
        # project events to neurons/filter segments
        p.Projection(on_population, on_filter_segments[-1], p.FromListConnector(connection))
        p.Projection(off_population, off_filter_segments[-1], p.FromListConnector(connection))
        # connect segments into a single filter neuron
        filter_connections = create_filter_neuron_connections(filter_split=filter_split,
                                                              no_neurons=no_neurons,
                                                              base_weight=base_weight)
        on_filter_populations.append(p.Population(no_neurons/filter_split, p.IF_curr_exp(*neuron_params),
                                                  label='on {} - {}'.format(filter, rotation)))
        p.Projection(on_filter_segments[-1], on_filter_populations[-1], p.FromListConnector(filter_connections))
        off_filter_populations.append(p.Population(no_neurons/filter_split, p.IF_curr_exp(*neuron_params),
                                                   label='off {} - {}'.format(filter, rotation)))
        p.Projection(off_filter_segments[-1], off_filter_populations[-1], p.FromListConnector(filter_connections))
        off_filter_populations[-1].record('all')
        on_filter_populations[-1].record('all')
        on_filter_segments[-1].record('all')
        off_filter_segments[-1].record('all')

    # create proto object
    proto_object_pop = proto_objects(on_filter_populations, off_filter_populations, filter[0], filter[1], base_weight)
    ################################
    # mutually inhibit everything? #
    ################################
    # opposite inhibition for filter segments
    # for rotation in range(4):
    #     p.Projection(on_filter_populations[rotation], on_filter_populations[rotation+4],
    #                  p.OneToOneConnector(),
    #                  p.StaticSynapse(weight=base_weight, delay=1),
    #                  receptor_type='inhibitory')
    #     p.Projection(on_filter_populations[rotation+4], on_filter_populations[rotation],
    #                  p.OneToOneConnector(),
    #                  p.StaticSynapse(weight=base_weight, delay=1),
    #                  receptor_type='inhibitory')
    #     p.Projection(off_filter_populations[rotation], off_filter_populations[rotation+4],
    #                  p.OneToOneConnector(),
    #                  p.StaticSynapse(weight=base_weight, delay=1),
    #                  receptor_type='inhibitory')
    #     p.Projection(off_filter_populations[rotation+4], off_filter_populations[rotation],
    #                  p.OneToOneConnector(),
    #                  p.StaticSynapse(weight=base_weight, delay=1),
    #                  receptor_type='inhibitory')
    first_layer_populations[0].append(off_filter_populations)
    first_layer_populations[1].append(on_filter_populations)

for object in proto_object_pop:
    object.record('all')
p.run(15*1000)

print "saving"
on_filter_segments_data = []
for data in on_filter_segments:
    on_filter_segments_data.append(data.get_data())
    spikes = on_filter_segments_data[-1].segments[0].spiketrains
    for neuron in spikes:
        print "on segment spike count:", neuron.size
np.save('on filter segments data.npy', on_filter_segments_data)

off_filter_segments_data = []
for data in off_filter_segments:
    off_filter_segments_data.append(data.get_data())
    spikes = off_filter_segments_data[-1].segments[0].spiketrains
    for neuron in spikes:
        print "off segment spike count:", neuron.size
np.save('off filter segments data.npy', off_filter_segments_data)

on_filter_populations_data = []
for data in on_filter_populations:
    on_filter_populations_data.append(data.get_data())
    spikes = on_filter_populations_data[-1].segments[0].spiketrains
    for neuron in spikes:
        print "on pop spike count:", neuron.size
np.save('on filter population data.npy', on_filter_populations_data)

off_filter_populations_data = []
for data in off_filter_populations:
    off_filter_populations_data.append(data.get_data())
    spikes = off_filter_populations_data[-1].segments[0].spiketrains
    for neuron in spikes:
        print "off pop spike count:", neuron.size
np.save('off filter population data.npy', off_filter_populations_data)

object_data = []
for object in proto_object_pop:
    object_data.append(object.get_data())
np.save('proto object data.npy', object_data)
print "all saved"

# pop_rec_data = pop_rec.get_data('spikes')
# pop_out_data = pop_out.get_data()
#
# Plot
F = Figure(
#         # plot data for postsynaptic neuron
#         Panel(in_spikes.segments[0].spiketrains,
#               yticks=True, markersize=2, xlim=(plot_start, plot_end)),
    Panel(object_data[0].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[1].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[2].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[3].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[4].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[5].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[6].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[7].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    # Panel(pop_out_data.segments[0].filter(name='v')[0],
    #       ylabel="Membrane potential (mV)",
    #       data_labels=[pop_out.label], yticks=True, xlim=(plot_start, plot_end)
    #       ),
    # Panel(pop_out_data.segments[0].filter(name='gsyn_exc')[0],
    #       ylabel="gsyn excitatory (mV)",
    #       data_labels=[pop_out.label], yticks=True, xlim=(plot_start, plot_end)
    #       ),
    # Panel(pop_out_data.segments[0].filter(name='gsyn_inh')[0],
    #       ylabel="gsyn inhibitory (mV)",
    #       data_labels=[pop_out.label], yticks=True, xlim=(plot_start, plot_end)
    #       ),
    # Panel(pop_out_data.segments[0].spiketrains,
    #       yticks=True, markersize=2, xlim=(plot_start, plot_end)),
    # annotations="Batch: {}".format(i)
    )
plt.show()
F = Figure(
#         # plot data for postsynaptic neuron
#         Panel(in_spikes.segments[0].spiketrains,
#               yticks=True, markersize=2, xlim=(plot_start, plot_end)),
    Panel(object_data[8].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[9].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[10].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[11].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[12].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[13].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[14].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[15].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    # Panel(pop_out_data.segments[0].filter(name='v')[0],
    #       ylabel="Membrane potential (mV)",
    #       data_labels=[pop_out.label], yticks=True, xlim=(plot_start, plot_end)
    #       ),
    # Panel(pop_out_data.segments[0].filter(name='gsyn_exc')[0],
    #       ylabel="gsyn excitatory (mV)",
    #       data_labels=[pop_out.label], yticks=True, xlim=(plot_start, plot_end)
    #       ),
    # Panel(pop_out_data.segments[0].filter(name='gsyn_inh')[0],
    #       ylabel="gsyn inhibitory (mV)",
    #       data_labels=[pop_out.label], yticks=True, xlim=(plot_start, plot_end)
    #       ),
    # Panel(pop_out_data.segments[0].spiketrains,
    #       yticks=True, markersize=2, xlim=(plot_start, plot_end)),
    # annotations="Batch: {}".format(i)
    )
plt.show()
F = Figure(
#         # plot data for postsynaptic neuron
#         Panel(in_spikes.segments[0].spiketrains,
#               yticks=True, markersize=2, xlim=(plot_start, plot_end)),
    Panel(object_data[16].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[17].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[18].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[19].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[20].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    Panel(object_data[21].segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, 15000)
          ),
    # Panel(pop_out_data.segments[0].filter(name='v')[0],
    #       ylabel="Membrane potential (mV)",
    #       data_labels=[pop_out.label], yticks=True, xlim=(plot_start, plot_end)
    #       ),
    # Panel(pop_out_data.segments[0].filter(name='gsyn_exc')[0],
    #       ylabel="gsyn excitatory (mV)",
    #       data_labels=[pop_out.label], yticks=True, xlim=(plot_start, plot_end)
    #       ),
    # Panel(pop_out_data.segments[0].filter(name='gsyn_inh')[0],
    #       ylabel="gsyn inhibitory (mV)",
    #       data_labels=[pop_out.label], yticks=True, xlim=(plot_start, plot_end)
    #       ),
    # Panel(pop_out_data.segments[0].spiketrains,
    #       yticks=True, markersize=2, xlim=(plot_start, plot_end)),
    # annotations="Batch: {}".format(i)
    )
plt.show()

print "done"









