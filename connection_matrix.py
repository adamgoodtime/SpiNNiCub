#!usr/bin/python

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
from os.path import join, getsize
import yarp
import warnings
import spynnaker8 as p
from ATIS.decode_events import *
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from generate_events import FakeStimuliBarMoving, fake_circle_moving

from pacman.model.constraints.key_allocator_constraints import FixedKeyAndMaskConstraint
from pacman.model.graphs.application import ApplicationSpiNNakerLinkVertex
from pacman.model.routing_info import BaseKeyAndMask
from spinn_front_end_common.abstract_models.abstract_provides_n_keys_for_partition import AbstractProvidesNKeysForPartition
from spinn_front_end_common.abstract_models.abstract_provides_outgoing_partition_constraints import AbstractProvidesOutgoingPartitionConstraints
from spinn_utilities.overrides import overrides
from spinn_front_end_common.abstract_models.abstract_provides_incoming_partition_constraints import AbstractProvidesIncomingPartitionConstraints
from pacman.executor.injection_decorator import inject_items
from pacman.operations.routing_info_allocator_algorithms.malloc_based_routing_allocator.utils import get_possible_masks
from spinn_front_end_common.utility_models.command_sender_machine_vertex import CommandSenderMachineVertex

from spinn_front_end_common.abstract_models \
    import AbstractSendMeMulticastCommandsVertex
from spinn_front_end_common.utility_models.multi_cast_command \
    import MultiCastCommand

warnings.filterwarnings("error")

'''
Components:
- create the filter
- downsample filter for various resolutions
- pass filter into a single neuron for part of proto-object
- combine parts of the proto-objects into a object/neuron
- inhibit between competing objects
'''

NUM_NEUR_IN = 1024 #1024 # 2x240x304 mask -> 0xFFFE0000
MASK_IN = 0xFFFFFC00 #0xFFFFFC00
NUM_NEUR_OUT = 1024
MASK_OUT =0xFFFFFFFC

class ICUBInputVertex(
        ApplicationSpiNNakerLinkVertex,
        AbstractProvidesOutgoingPartitionConstraints,
        AbstractProvidesIncomingPartitionConstraints,
        AbstractSendMeMulticastCommandsVertex):

    def __init__(self, spinnaker_link_id, board_address=None,
                 constraints=None, label=None):

        ApplicationSpiNNakerLinkVertex.__init__(
            self, n_atoms=NUM_NEUR_IN, spinnaker_link_id=spinnaker_link_id,
            board_address=board_address, label=label)

        AbstractProvidesNKeysForPartition.__init__(self)
        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        AbstractSendMeMulticastCommandsVertex.__init__(self)

    @overrides(AbstractProvidesOutgoingPartitionConstraints.
               get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        return [FixedKeyAndMaskConstraint(
            keys_and_masks=[BaseKeyAndMask(
                base_key=0, #upper part of the key,
                mask=MASK_IN)])]
                #keys, i.e. neuron addresses of the input population that sits in the ICUB vertex,

    @inject_items({"graph_mapper": "MemoryGraphMapper"})
    @overrides(AbstractProvidesIncomingPartitionConstraints.
               get_incoming_partition_constraints,
               additional_arguments=["graph_mapper"])
    def get_incoming_partition_constraints(self, partition, graph_mapper):
        if isinstance(partition.pre_vertex, CommandSenderMachineVertex):
            return []
        index = graph_mapper.get_machine_vertex_index(partition.pre_vertex)
        vertex_slice = graph_mapper.get_slice(partition.pre_vertex)
        mask = get_possible_masks(vertex_slice.n_atoms)[0]
        key = (0x10 + index) << 24
        return [FixedKeyAndMaskConstraint(
            keys_and_masks=[BaseKeyAndMask(key, mask)])]

    @property
    @overrides(AbstractSendMeMulticastCommandsVertex.start_resume_commands)
    def start_resume_commands(self):
        return [MultiCastCommand(
            key=0x80000000, payload=0, repeat=5, delay_between_repeats=100)]

    @property
    @overrides(AbstractSendMeMulticastCommandsVertex.pause_stop_commands)
    def pause_stop_commands(self):
        return [MultiCastCommand(
            key=0x40000000, payload=0, repeat=5, delay_between_repeats=100)]

    @property
    @overrides(AbstractSendMeMulticastCommandsVertex.timed_commands)
    def timed_commands(self):
        return []

def zero_2pi_tan(x, y):
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += np.pi * 2
    if angle > 6 * np.pi / 4:
        angle -= 6 * np.pi / 4
    else:
        angle += np.pi / 2
    return angle

# Von Mises filter function, takes values between -1 and 1 and outputs the corresponding value
def VM(x, y, r=10, p=0.08, theta=0, threshold=0.75, filter_split=4):
    theta *= np.pi/4
    # rescaling of x and y for correct filter placement
    x = (x*15) + (0.8*r) * np.cos(theta)
    y = (y*15) - (0.8*r) * np.sin(theta)
    # slit the value for the filter
    angle = zero_2pi_tan(x, y)
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

def convert_pixel_to_id(x, y):
    if isinstance(simulate, str):
        return (y*x_res) + x
    else:
        return (int(y) << 12) + (int(x) << 1) + 1

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
    return list_of_corners, x, y

def create_peripheral_mapping(base_weight, percentage_fire_threshold=0.5, plot=False):
    '''
    0........w
    w+1      w+2
    .        .
    .        .
    .        .
    w+2h.....2w+2h
    '''
    max_blocks = 4 + horizontal_split + horizontal_split + veritcal_split + veritcal_split
    block_count = [0 for i in range(max_blocks)]
    pixel_mapping = []
    xs = []
    ys = []
    zs = []
    for i in range(x_res):
        x = []
        y = []
        z = []
        for j in range(y_res):
            if peripheral_x <= i < x_res - peripheral_x and peripheral_y <= j < y_res - peripheral_y:
                x.append(i)
                y.append(j)
                z.append(-5)
            else:
                x.append(i)
                y.append(j)
                # left side
                if i < peripheral_x:
                    if j < peripheral_y:
                        id = 0
                        block_count[id] += 1
                        pixel_mapping.append([convert_pixel_to_id(i, j), id, base_weight, 1])
                    elif j >= y_res - peripheral_y:
                        id = max_blocks - horizontal_split - 2
                        block_count[id] += 1
                        pixel_mapping.append([convert_pixel_to_id(i, j), id, base_weight, 1])
                    else:
                        block = float(j - peripheral_y) / (float(fovea_y) / float(veritcal_split))
                        id = 2 + horizontal_split + (2 * int(block))
                        block_count[id] += 1
                        pixel_mapping.append([convert_pixel_to_id(i, j), id, base_weight, 1])
                # top side
                if j < peripheral_y:
                    if i < peripheral_x:
                        None
                    elif i >= x_res - peripheral_x:
                        id = 1 + horizontal_split
                        block_count[id] += 1
                        pixel_mapping.append([convert_pixel_to_id(i, j), id, base_weight, 1])
                    else:
                        block = float(i - peripheral_x) / (float(fovea_x) / float(horizontal_split))
                        id = 1 + int(block)
                        block_count[id] += 1
                        pixel_mapping.append([convert_pixel_to_id(i, j), id, base_weight, 1])
                # right side
                if i >= x_res - peripheral_x:
                    if j < peripheral_y:
                        None
                    elif j >= y_res - peripheral_y:
                        id = max_blocks - 1
                        block_count[id] += 1
                        pixel_mapping.append([convert_pixel_to_id(i, j), id, base_weight, 1])
                    else:
                        block = float(j - peripheral_y) / (float(fovea_y) / float(veritcal_split))
                        id = 2 + horizontal_split + (2 * int(block)) + 1
                        block_count[id] += 1
                        pixel_mapping.append([convert_pixel_to_id(i, j), id, base_weight, 1])
                # bottom side
                if j >= y_res - peripheral_y:
                    if i < peripheral_x:
                        None
                    elif i >= x_res - peripheral_x:
                        None
                    else:
                        block = float(i - peripheral_x) / (float(fovea_x) / float(horizontal_split))
                        id = max_blocks - 1 - (horizontal_split - int(block))
                        block_count[id] += 1
                        pixel_mapping.append([convert_pixel_to_id(i, j), id, base_weight, 1])
                z.append(id)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # scale weights
    for connection in range(len(pixel_mapping)):
        pixel_mapping[connection][2] /= float(block_count[pixel_mapping[connection][1]]) * percentage_fire_threshold

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(np.array(xs), np.array(ys), np.array(zs))
        plt.show()

    return pixel_mapping


def visual_field_with_overlap(filter_width, filter_height, overlap=0., filter_split=4, rotation=0,
                              base_weight=1., percentage_fire_threshold=0.5, inhib_percentage_fire_threshold=1.,
                              inhib_connect_prob=1., plot=False):
    # create the filter which is to be copied in a grid like fashion around the visual field
    filter_split_matrix = []
    filter_matrix = []
    xs = []
    ys = []
    zero_count = 0
    for i in range(filter_width):
        split_row = []
        filter_row = []
        x = []
        y = []
        for j in range(filter_height):
            x_value = ((float(i) / float(filter_width)) * 2) - 1
            y_value = ((float(j) / float(filter_height)) * 2) - 1
            pixel_value, split = VM(x_value, y_value, theta=rotation, filter_split=filter_split)
            if not pixel_value:
                zero_count += 1
            x.append(x_value)
            y.append(y_value)
            filter_row.append(pixel_value)
            split_row.append(split)
        filter_split_matrix.append(split_row)
        filter_matrix.append(filter_row)
        xs.append(x)
        ys.append(y)

    # calculate all filter corners for later processing
    corners_list, max_filters_x, max_filters_y = create_filter_boundaries(filter_width, filter_height, overlap)

    # copy the filter dimensions around the board
    visual_matrix = []
    split_matrix = []
    xs = []
    ys = []
    exc_connection_list = []
    inh_connection_list = []
    exc_neuron_id_count = [0 for i in range(max_filters_x * max_filters_y * filter_split)]
    inh_synapse_count = [0 for i in range(max_filters_x * max_filters_y)]
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
                pixel_value = convert_pixel_to_id(x_offset + filter_x, y_offset + filter_y)
                if filter_matrix[filter_x][filter_y] and filter_split_matrix[filter_x][filter_y] >= 0:
                    split_value = filter_split_matrix[filter_x][filter_y]
                    neuron_id = (corner[2] * filter_split) + (max_filters_x * corner[3] * filter_split) + split_value
                    exc_neuron_id_count[neuron_id] += 1
                    exc_connection_list.append([pixel_value, neuron_id, base_weight, 1])
                elif inhib_percentage_fire_threshold and \
                        not filter_matrix[filter_x][filter_y] \
                        and np.random.random() < inhib_connect_prob:
                    filter_id = corner[2] + (corner[3] * max_filters_x)
                    inh_synapse_count[filter_id] += 1
                    inh_connection_list.append([pixel_value, filter_id, base_weight, 1])
            xs.append(x)
            ys.append(y)
            visual_matrix.append(visual_row)
            split_matrix.append(split_row)

    # scale weights
    for connection in range(len(exc_connection_list)):
        exc_connection_list[connection][2] /= exc_neuron_id_count[exc_connection_list[connection][1]] * percentage_fire_threshold
    for connection in range(len(inh_connection_list)):
        inh_connection_list[connection][2] /= inh_synapse_count[inh_connection_list[connection][1]] * inhib_percentage_fire_threshold

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(np.array(xs), np.array(ys), np.array(visual_matrix))
        plt.show()
    return exc_connection_list, inh_connection_list, len(exc_neuron_id_count)

def create_filter_neuron_connections(filter_split, no_neurons, base_weight, percentage_fire_threshold):
    connections = []
    for i in range(no_neurons):
        connections.append([i, int(i/filter_split), (base_weight/filter_split)/percentage_fire_threshold, 1])
    return connections

def proto_objects(population_1, population_2, filter_width, filter_height, base_weight, weight_scale=0.5):
    max_filters_x_wrong = int((x_res - peripheral_x*2) / (filter_width * (1-overlap)))
    max_filters_y_wrong = int((y_res - peripheral_y*2) / (filter_height * (1-overlap)))
    rubbish, max_filters_x, max_filters_y = create_filter_boundaries(filter_width, filter_height, overlap)
    proto_object_neurons = []
    # create the connections and populations
    for i in range(max_filters_x):
        for j in range(max_filters_y):
            n1 = (i) + (j * max_filters_x)
            if j + 1 < max_filters_y:
                n2 = (i) + ((j+1) * max_filters_x)
                proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                         label='{}-{}-{}-ver'.format(filter_width, i, j)))
                p.Projection(population_1[6], proto_object_neurons[-1], p.FromListConnector([[n2, 0, base_weight*weight_scale, 1]]))
                p.Projection(population_2[2], proto_object_neurons[-1], p.FromListConnector([[n1, 0, base_weight*weight_scale, 1]]))
                if i - 1 >= 0:
                    n2 = (i-1) + ((j+1) * max_filters_x)
                    proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                             label='{}-{}-{}-upl'.format(filter_width, i, j)))
                    p.Projection(population_1[5], proto_object_neurons[-1], p.FromListConnector([[n2, 0, base_weight*weight_scale, 1]]))
                    p.Projection(population_2[1], proto_object_neurons[-1], p.FromListConnector([[n1, 0, base_weight*weight_scale, 1]]))
            if i + 1 < max_filters_x:
                n2 = (i+1) + ((j) * max_filters_x)
                proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                         label='{}-{}-{}-hor'.format(filter_width, i, j)))
                p.Projection(population_1[0], proto_object_neurons[-1], p.FromListConnector([[n2, 0, base_weight*weight_scale, 1]]))
                p.Projection(population_2[4], proto_object_neurons[-1], p.FromListConnector([[n1, 0, base_weight*weight_scale, 1]]))
                if j + 1 < max_filters_y:
                    n2 = (i+1) + ((j+1) * max_filters_x)
                    proto_object_neurons.append(p.Population(1, p.IF_curr_exp(*neuron_params),
                                                             label='{}-{}-{}-upr'.format(filter_width, i, j)))
                    p.Projection(population_1[7], proto_object_neurons[-1], p.FromListConnector([[n2, 0, base_weight*weight_scale, 1]]))
                    p.Projection(population_2[3], proto_object_neurons[-1], p.FromListConnector([[n1, 0, base_weight*weight_scale, 1]]))
    return proto_object_neurons

def convert_filter_xy_to_proto_centre(split_data, overlap):
    [filter_size, filter_x, filter_y, direction] = split_data
    filter_x = float(filter_x)
    filter_y = float(filter_y)
    filter_width = float(filter_size)
    filter_height = float(filter_size)
    if direction == 'ver':
        offset_x = 0
        offset_y = (filter_height / 2.) * (1.-overlap)
    elif direction == 'hor':
        offset_x = (filter_width / 2.) * (1.-overlap)
        offset_y = 0
    elif direction == 'upl':
        offset_x = -(filter_width / 2.) * (1.-overlap)
        offset_y = (filter_height / 2.) * (1.-overlap)
    elif direction == 'upr':
        offset_x = (filter_width / 2.) * (1.-overlap)
        offset_y = (filter_height / 2.) * (1.-overlap)
    else:
        Exception
    # print filter_size, filter_x, filter_y, direction
    x = filter_x * (filter_width - (overlap * filter_width))
    x += offset_x + peripheral_x + (filter_width / 2.)
    y = filter_y * (filter_height - (overlap * filter_height))
    y += offset_y + peripheral_y + (filter_height / 2.)

    return x, y

def parse_ATIS(file_location, file_name):
    f = open("{}/{}".format(file_location, file_name), "r")
    events = [[] for i in range(x_res*y_res)]
    for line in f:
        line = np.array(line.split(','))
        if float(line[0]) < 0:
            time = (float(line[0]) + 87.) * 1000.
            # print time, "n"
        else:
            time = float(line[0]) * 1000.
            # print time
        events[convert_pixel_to_id(int(line[2]), int(line[3]))].append(time)
    return events

def combine_parsed_ATIS(event_list):
    time_offset = 0
    max_time = 0
    events = [[] for i in range(x_res*y_res)]
    for data_set in event_list:
        print 'combining', event_list.index(data_set) + 1, '/', len(event_list)
        for neuron_id in range(len(data_set)):
            for time in range(len(data_set[neuron_id])):
                new_time = data_set[neuron_id][time] + time_offset
                events[neuron_id].append(new_time)
                if new_time > max_time:
                    max_time = new_time
        time_offset = max_time
    print 'maximum time after combining was', np.ceil(max_time)
    return events

def gather_all_ATIS_log(top_directory):
    all_directories = []
    for root, dirs, files in os.walk(top_directory):
        # dm = DataManager()
        # if 'data.log' in files:
        #     print root
        #     print 'data.log size:', getsize(join(root, 'data.log'))
        #     if getsize(join(root, 'data.log')):
        #         dm.load_AE_from_yarp(root)
        # for file in files:
            # if file != 'data.log' and file != 'events.gif' and file != 'decoded_events.txt':
            #     os.remove(root+'/'+file)
            # if file == 'events.gif' and 'videos' in root:
            #     shutil.move(root+'/'+file, root.replace('videos', '')+'events.gif')

        # if 'videos' in dirs:
        #     shutil.rmtree(root+'/'+'videos')
        #     dirs.remove('videos')
        if 'decoded_events.txt' in files:
            # print root
            # print 'data.log size:', getsize(join(root, 'data.log')), 'and events.txt:', getsize(join(root, 'decoded_events.txt'))
            all_directories.append(root)
    return all_directories

def generate_fake_stimuli(directions, stimulus):
    all_on = []
    all_off = []
    time_stamp = 0
    for direction in directions:
        if stimulus == 'square':
            on, off, time_stamp = FakeStimuliBarMoving([x_res, y_res], 1., 4., [50, 50], direction, 'BlackOverWhite', time_stamp)
        else:
            on, off, time_stamp = fake_circle_moving(direction, r=30, mspp=7, start_time=time_stamp, circle_contrast='black')
        all_on.append(on)
        all_off.append(off)
    if stimulus == 'square':
        events = parse_event_class(all_on, all_off)
    else:
        events = [[] for i in range(x_res*y_res)]
        for direction in all_on:
            for [x, y, t] in direction:
                events[convert_pixel_to_id(x, y)].append(t)
        for direction in all_off:
            for [x, y, t] in direction:
                events[convert_pixel_to_id(x, y)].append(t)
    return events

def parse_event_class(eventsON, eventsOFF):
    events = [[] for i in range(x_res*y_res)]
    for direction in eventsON:
        for event in direction:
            x = event.x
            y = event.y
            timestamp = event.timestamp
            polarity = event.polarity
            events[convert_pixel_to_id(x, y)].append(timestamp)
    for direction in eventsOFF:
        for event in direction:
            x = event.x
            y = event.y
            timestamp = event.timestamp
            polarity = event.polarity
            events[convert_pixel_to_id(x, y)].append(timestamp)
    return events

def connect_vis_pop(vis, post, connections, receptor_type='excitatory'):
    if isinstance(simulate, str):
        p.Projection(vis, post, p.FromListConnector(connections), receptor_type=receptor_type)
    else:
        zero_polarity = []
        for conn in connections:
            zero_polarity.append([conn[0]-1, conn[1], conn[2], conn[3]])
        p.Projection(vis, post, p.FromListConnector(connections), receptor_type=receptor_type)
        p.Projection(vis, post, p.FromListConnector(zero_polarity), receptor_type=receptor_type)

def create_movement(proto, boarder, proto_weight_scale, boarder_weight_scale, base_weight):
    # up, down, left, right = 0, 1, 2, 3
    move_pop = p.Population(4, p.IF_curr_exp(*neuron_params), label='movement_udlr')
    boarder_movement_connections = []
    for neuron in range(boarder.size):
        # down
        if neuron < 2 + horizontal_split:
            boarder_movement_connections.append([neuron, 1, base_weight*boarder_weight_scale, 1])
        # up
        elif neuron >= 2 + horizontal_split + (2 * veritcal_split):
            boarder_movement_connections.append([neuron, 0, base_weight*boarder_weight_scale, 1])
        # left
        elif (neuron - (2 + horizontal_split)) % 2 == 0 or neuron == 0 or neuron == 2 + horizontal_split + (2 * veritcal_split):
            boarder_movement_connections.append([neuron, 2, base_weight*boarder_weight_scale, 1])
        # right
        elif (neuron - (2 + horizontal_split)) % 2 == 1 or neuron == 1 + veritcal_split or neuron == 3 + (2 * horizontal_split) + (2 * veritcal_split):
            boarder_movement_connections.append([neuron, 3, base_weight*boarder_weight_scale, 1])
    p.Projection(boarder, move_pop, p.FromListConnector(boarder_movement_connections))

    for proto_size in proto:
        for proto_object in proto_size:
            split_data = proto_object.label.split('-')
            x, y = convert_filter_xy_to_proto_centre(split_data, overlap)
            if x < x_res / 2:
                p.Projection(proto_object, move_pop, p.FromListConnector([[0, 1, base_weight*boarder_weight_scale, 1]]))
            if x >= x_res / 2:
                p.Projection(proto_object, move_pop, p.FromListConnector([[0, 0, base_weight*boarder_weight_scale, 1]]))
            if y < y_res / 2:
                p.Projection(proto_object, move_pop, p.FromListConnector([[0, 2, base_weight*boarder_weight_scale, 1]]))
            if y >= y_res / 2:
                p.Projection(proto_object, move_pop, p.FromListConnector([[0, 3, base_weight*boarder_weight_scale, 1]]))
    # mutual inhibition
    # inhibit_list = []
    # for i in range(4):
    #     for j in range(4):
    #         if i != j:
    #             inhibit_list.append([i, j, base_weight, 1])
    # p.Projection(move_pop, move_pop, p.FromListConnector(inhibit_list), receptor_type='inhibitory')
    return move_pop

x_res = 304
y_res = 240

simulate = 'real'

if __name__ == '__main__':
    fovea_x = 300
    fovea_y = 236
    # min for 1 board
    # fovea_x = 170
    # fovea_y = 135
    peripheral_x = (x_res - fovea_x) / 2
    peripheral_y = (y_res - fovea_y) / 2
    horizontal_split = 1
    veritcal_split = 1

    neuron_params = {
        # balance the refractory period/tau_mem so membrane has lost contribution before next spike
    }
    ##############################
    # connection configurations: #
    ##############################
    # filter_sizes = [30, 46, 70, 100]
    filter_sizes = [100, 70, 55, 40]
    # filter_sizes = [100, 90, 80, 70, 60, 50, 40, 30, 20]
    # filter_sizes = [46, 30]
    list_of_filter_sizes = []
    for filter_size in filter_sizes:
        list_of_filter_sizes.append([filter_size, filter_size])
    filter_split = 4
    overlap = 0.6
    base_weight = 5.
    boarder_percentage_fire_threshold = 0.2
    segment_percentage_fire_threshold = 0.02
    filter_percentage_fire_threshold = 0.8
    inhib_percentage_fire_threshold = 0.02
    inhib_connect_prob = 1.
    proto_scale = 0.75
    inhib = 'all' #[0]: +ve+ve, -ve-ve   [1]:+ve-ve, -ve+ve
    WTA = False
    to_wta = 10.
    from_wta = 10.
    self_excite = 0.

    simulate = 'proto'
    # simulate = None
    label = "{} fs-{} ol-{} w-{} bft-{} sft-{} fft-{} ift-{} icp-{} ps-{} in-{}".format(simulate, filter_split, overlap,
                                                                                       base_weight,
                                                                                       boarder_percentage_fire_threshold,
                                                                                       segment_percentage_fire_threshold,
                                                                                       filter_percentage_fire_threshold,
                                                                                       inhib_percentage_fire_threshold,
                                                                                       inhib_connect_prob, proto_scale,
                                                                                       inhib)
    if WTA:
        label += ' to-{} from-{}'.format(to_wta, from_wta)
    if self_excite:
        label += ' self-{}'.format(self_excite)
    label += ' {}'.format(filter_sizes)

    print "\nCreating events for", label
    # extract input data
    # dm = DataManager()
    # dm.load_AE_from_yarp('ATIS')
    if isinstance(simulate, str):
        if simulate == 'square':
            directions = ['LR', 'RL', 'BT', 'TB', 'LR', 'RL', 'BT', 'TB', 'LR', 'RL', 'BT', 'TB']
            events = generate_fake_stimuli(directions, 'square')
        elif simulate == 'basic':
            events = parse_ATIS('ATIS/data_surprise', 'decoded_events.txt')
        elif simulate == 'circle':
            directions = ['up', 'down', 'left', 'right', 'up', 'down', 'left', 'right', 'up', 'down', 'left', 'right']
            events = generate_fake_stimuli(directions, 'circle')
        elif simulate == 'sim_dir':
            contrasts = ['high', 'medium', 'low']
            locations = ['RL', 'LR', 'BT', 'TB']
            combined_events = []
            for contrast in contrasts:
                print contrast, ':'
                for location in locations:
                    print "\t", location
                    combined_events.append(parse_ATIS('ATIS/{}/{}'.format(contrast, location), 'decoded_events.txt'))
            events = combine_parsed_ATIS(combined_events)
        else:
            if simulate == 'all_ATIS':
                all_directories = gather_all_ATIS_log('ATIS/IROS_attention')
            elif simulate == 'subset':
                all_directories = gather_all_ATIS_log('ATIS/IROS_subset')
            elif simulate == 'proto':
                all_directories = gather_all_ATIS_log('ATIS/(no)proto_object')
            else:
                print "incorrect stimulus setting"
                Exception
            combined_events = []
            for directory in all_directories:
                print 'extracting directory', all_directories.index(directory) + 1, '/', len(all_directories)
                combined_events.append(parse_ATIS(directory, 'decoded_events.txt'))
            events = combine_parsed_ATIS(combined_events)
        print "Events created"
        runtime = 0
        for neuron_id in range(len(events)):
            for time in events[neuron_id]:
                if time > runtime:
                    runtime = time
        runtime = int(np.ceil(runtime)) + 1000
        print "running for", runtime, "ms"
        p.setup(timestep=1.0)
        vis_pop = p.Population(x_res*y_res, p.SpikeSourceArray(events), label='pop_in')
    else:
        print "running until enter is pressed"
        p.setup(timestep=1.0)
        vis_pop = p.Population(None, ICUBInputVertex(spinnaker_link_id=0), label='pop_in')

    # create boarder connections and populations
    boarder_connections = create_peripheral_mapping(base_weight=base_weight,
                                                    percentage_fire_threshold=boarder_percentage_fire_threshold,
                                                    plot=False)
    boarder_population = p.Population(4+(veritcal_split*2)+(horizontal_split*2), p.IF_curr_exp(*neuron_params),
                                      label="boarder populations")
    if not isinstance(simulate, str):
        boarder_population.record('spikes')
    connect_vis_pop(vis_pop, boarder_population, boarder_connections)
    # p.Projection(vis_pop, boarder_population, p.FromListConnector(boarder_connections))

    # create filters and connections from input video stream
    all_filter_segments = []
    all_filter_populations = []
    all_proto_object_pops = []
    for filter in list_of_filter_sizes:
        print "SpiNN setup for filter", filter
        # for each rotation, 0 -> 7pi/4
        filter_segments = []
        filter_populations = []
        for rotation in range(8):
            print "Rotation", rotation+1, "/ 8"
            segment_connection, inhib_connection, no_neurons = visual_field_with_overlap(filter[0], filter[1],
                                                                                         overlap=overlap,
                                                                                         rotation=rotation,
                                                                                         filter_split=filter_split,
                                                                                         base_weight=base_weight/float(filter_split),
                                                                                         percentage_fire_threshold=segment_percentage_fire_threshold,
                                                                                         inhib_percentage_fire_threshold=inhib_percentage_fire_threshold,
                                                                                         inhib_connect_prob=inhib_connect_prob,
                                                                                         plot=False)
            # create neurons for each segment of filter
            filter_segments.append(p.Population(no_neurons, p.IF_curr_exp(*neuron_params),
                                                   label='segments {} - {}'.format(filter, rotation)))
            # project events to neurons/filter segments
            connect_vis_pop(vis_pop, filter_segments[-1], segment_connection)
            # p.Projection(vis_pop, filter_segments[-1], p.FromListConnector(segment_connection))
            # connect segments into a single filter neuron
            filter_connections = create_filter_neuron_connections(filter_split=filter_split,
                                                                  no_neurons=no_neurons,
                                                                  base_weight=base_weight,
                                                                  percentage_fire_threshold=filter_percentage_fire_threshold)
            filter_populations.append(p.Population(no_neurons/filter_split, p.IF_curr_exp(*neuron_params),
                                                      label='filter {} - {}'.format(filter, rotation)))
            if len(inhib_connection) != 0:
                connect_vis_pop(vis_pop, filter_populations[-1], inhib_connection, receptor_type='inhibitory')
                # p.Projection(vis_pop, filter_populations[-1], p.FromListConnector(inhib_connection), receptor_type='inhibitory')
            p.Projection(filter_segments[-1], filter_populations[-1], p.FromListConnector(filter_connections))
            if not isinstance(simulate, str):
                filter_populations[-1].record('spikes')
                filter_segments[-1].record('spikes')
            print "number of neurons in segments = ", no_neurons
            print "number of neurons in filters = ", (no_neurons/filter_split)
            print "number of synapses in ATIS->segments: {}, segments->filters: {}, ATIS->filters: {}".format(len(segment_connection), len(filter_connections), len(inhib_connection))

        print "total number of neurons in segments = ", no_neurons * 8
        print "total number of neurons in filters = ", (no_neurons / filter_split) * 8
        print "total number of synapses in ATIS->segments: {}, segments->filters: {}, ATIS->filters: {}".format(len(segment_connection)*8, len(filter_connections)*8, len(inhib_connection)*8)
        # create proto object
        all_proto_object_pops.append(proto_objects(filter_populations, filter_populations, filter[0], filter[1], base_weight, weight_scale=proto_scale))
        all_filter_segments.append(filter_segments)
        all_filter_populations.append(filter_populations)
        # opposite inhibition for filter segments
        if inhib:
            if inhib == 'all':
                for rotation1 in range(8):
                    for rotation2 in range(8):
                        if rotation1 != rotation2:
                            p.Projection(filter_populations[rotation1], filter_populations[rotation2],
                                         p.OneToOneConnector(),
                                         p.StaticSynapse(weight=base_weight, delay=1),
                                         receptor_type='inhibitory')
            else:
                for rotation in range(4):
                    p.Projection(filter_populations[rotation], filter_populations[rotation+4],
                                 p.OneToOneConnector(),
                                 p.StaticSynapse(weight=base_weight, delay=1),
                                 receptor_type='inhibitory')
                    p.Projection(filter_populations[rotation+4], filter_populations[rotation],
                                 p.OneToOneConnector(),
                                 p.StaticSynapse(weight=base_weight, delay=1),
                                 receptor_type='inhibitory')

    if WTA:
        wta_neuron = p.Population(1, p.IF_curr_exp(*neuron_params), label='WTA')
        if not isinstance(simulate, str):
            wta_neuron.record('spikes')

    for idx, proto_object_pop in enumerate(all_proto_object_pops):
        to_wta_scale = float(len(proto_object_pop))
        from_wta_scale = float(len(proto_object_pop))
        print 'to wta weight:', base_weight * (to_wta / to_wta_scale), '- from wta weight:', base_weight * (from_wta / to_wta_scale)
        print "number of neurons and synapses in filter", filter_sizes[idx], "proto-objects = ", len(proto_object_pop)
        for object in proto_object_pop:
            if not isinstance(simulate, str):
                object.record('spikes')
            if WTA:
                p.Projection(object, wta_neuron, p.FromListConnector([[0, 0, base_weight * (to_wta / to_wta_scale), 1]]), receptor_type='excitatory')
                p.Projection(wta_neuron, object, p.FromListConnector([[0, 0, base_weight * (from_wta / to_wta_scale), 1]]), receptor_type='inhibitory')
            if self_excite:
                p.Projection(object, object, p.FromListConnector([[0, 0, base_weight * self_excite, 1]]))

    move_pop = create_movement(all_proto_object_pops, boarder_population, 0.1, 0.1, base_weight)
    if not isinstance(simulate, str):
        move_pop.record('spikes')

    # p.run(runtime)
    if isinstance(simulate, str):
        p.run(runtime)
    else:
        p.external_devices.activate_live_output_to(move_pop, vis_pop)
        # out_port = yarp.BufferedPortBottle()
        # out_port.open('/spinn:o')
        # # bottle = out_port.prepare()
        # # bottle.clear()
        # # bottle.addInt32(2)
        # # out_port.write()
        # # out_port
        # # b.addString("thing")
        # while True:
        #     bottle = out_port.prepare()
        #     bottle.clear()
        #     bottle.addInt32(2)
        #     out_port.write()
        p.external_devices.run_forever()
        # stop recording
        raw_input('Press enter to stop')

    print "saving"
    boarder_data = boarder_population.get_data()
    np.save('board pop data {}.npy'.format(label), boarder_data)

    all_filter_segments_data = []
    for filter_size, filter_data in enumerate(all_filter_segments):
        filter_segments_data = []
        for data in filter_data:
            filter_segments_data.append([data.get_data(), data.label])
        all_filter_segments_data.append(filter_segments_data)
        np.save('filter segments data {}-{}.npy'.format(filter_sizes[filter_size], label), filter_segments_data)

    all_filter_populations_data = []
    for filter_size, filter_data in enumerate(all_filter_populations):
        filter_populations_data = []
        for data in filter_data:
            filter_populations_data.append([data.get_data(), data.label])
        all_filter_populations_data.append(filter_populations_data)
        np.save('filter pop data {}-{}.npy'.format(filter_sizes[filter_size], label), filter_populations_data)

    all_proto_object_data = []
    for filter_size, proto_pop in enumerate(all_proto_object_pops):
        object_data = []
        for object in proto_pop:
            object_data.append([object.get_data(), object.label])
        all_proto_object_data.append(object_data)
        np.save('proto object data {}-{}.npy'.format(filter_sizes[filter_size], label), object_data)

    move_data = move_pop.get_data()
    np.save('movement data {}.npy'.format(label), move_data)
    print "all saved"

    filter_segment_spikes = [0 for i in range(len(filter_sizes))]
    for filter_idx, filter_segments_data in enumerate(all_filter_segments_data):
        for idx, pop in enumerate(filter_segments_data):
            spikes = pop[0].segments[0].spiketrains
            for id2, neuron in enumerate(spikes):
                filter_segment_spikes[filter_idx] += neuron.size
                print pop[1], ":", idx, "-", id2, "segment spike count:", neuron.size
    filter_pop_spikes = [0 for i in range(len(filter_sizes))]
    filter_spikes_times = [[] for i in range(len(filter_sizes))]
    for filter_idx, filter_populations_data in enumerate(all_filter_populations_data):
        rotation_spike_times = []
        for idx, pop in enumerate(filter_populations_data):
            spikes = pop[0].segments[0].spiketrains
            for id2, neuron in enumerate(spikes):
                filter_pop_spikes[filter_idx] += neuron.size
                if neuron.size:
                    spike_times = spikes[0].magnitude
                    for spike_time in spike_times:
                        # idx = rotation, id2 = neuron id for that rotation
                        rotation_spike_times.append([pop[1], idx, id2, spike_time])
        filter_spikes_times.append(rotation_spike_times)
    np.save('filter rotations spikes {}'.format(label), filter_spikes_times)
                print pop[1], ":", idx, "-", id2, "pop spike count:", neuron.size
    object_spikes = [0 for i in range(len(filter_sizes))]
    coords_and_times = []
    all_spike_count = {}
    for filter_idx, object_data in enumerate(all_proto_object_data):
        spike_count = {}
        for idx, pop in enumerate(object_data):
            spikes = pop[0].segments[0].spiketrains
            for id2, neuron in enumerate(spikes):
                object_spikes[filter_idx] += neuron.size
                print pop[1], ":", idx, "-", id2, "proto-object spike count:", neuron.size
                split_data = pop[1].split('-')
                spike_data = pop[0].segments[0].spiketrains
                spikes = 0
                for neuron in spike_data:
                    spikes += neuron.size
                if spikes:
                    spike_times = spike_data[0].magnitude
                    x, y = convert_filter_xy_to_proto_centre(split_data, overlap)
                    for spike_time in spike_times:
                        coords_and_times.append([x, y, spike_time, filter_sizes[filter_idx]])
                        if '({}, {})'.format(x, y) in spike_count:
                            spike_count['({}, {})'.format(x, y)] += 1
                        else:
                            spike_count['({}, {})'.format(x, y)] = 1
        all_spike_count['{}'.format(filter_sizes[filter_idx])] = spike_count
    for filter_size in all_spike_count:
        print "filter size:", filter_size
        for location in all_spike_count[filter_size]:
            print location, all_spike_count[filter_size][location]
    np.save('all extracted proto spikes {}.npy'.format(label), coords_and_times)
    print '\nup = 0, down = 1, left = 2, right = 3'
    direction_key = {'0': [1, 2], '1': [1, 0], '2': [0, 1], '3': [2, 1]}
    move_spike_data = []
    for idx, neuron in enumerate(move_data.segments[0].spiketrains):
        print 'direction: {} - spikes: {}'.format(idx, neuron.size)
        if neuron.size:
            spike_times = neuron.magnitude
            for spike_time in spike_times:
                # print idx, spike_time
                move_spike_data.append([direction_key['{}'.format(idx)][0], direction_key['{}'.format(idx)][1], spike_time])
    np.save('extracted move spikes {}.npy'.format(label), move_spike_data)

    boarder_spikes = 0
    spikes = boarder_data.segments[0].spiketrains
    for id2, neuron in enumerate(spikes):
        boarder_spikes += neuron.size
        print id2, "boarder spike count:", neuron.size
    print "total boarder spikes:", boarder_spikes
    for idx, filter_size in enumerate(filter_sizes):
        print "total spikes for {}:\n" \
              "filter size: {}\n" \
              "segment: {}\n" \
              "filter: {}\n" \
              "objects: {}".format(label, filter_size,
                                   filter_segment_spikes[idx],
                                   filter_pop_spikes[idx],
                                   object_spikes[idx])

    wta_spikes = wta_neuron.get_data()
    print 'wta_spikes:', wta_spikes.segments[0].spiketrains[0].size

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









