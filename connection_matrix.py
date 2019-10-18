#!usr/bin/python

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("error")

'''
Components:
- create the filter
- downsample filter for various resolutions
- pass filter into a single neuron for part of proto-object
- combine parts of the proto-objects into a object/neuron
- inhibit between competing objects
'''

x_res = 304
y_res = 240

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

# down_sample = generate_matrix(20, rotation=0, plot=True)
# down_sample = generate_matrix(20, rotation=1, plot=True)
# down_sample = generate_matrix(20, rotation=2, plot=True)
# down_sample = generate_matrix(20, rotation=3, plot=True)

def convert_pixel_to_id(x, y):
    return (y*304) + x

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
                              base_weight=1, percentage_fire_threshold=0.5, plot=False):
    # create inputs of -1 to 1 for entire visual field, assuming no overlap for now
    filter_x_res = filter_width
    filter_y_res = filter_height
    overlap_x = filter_width * overlap
    overlap_y = filter_height * overlap
    max_filters_x = int(x_res / (filter_width / (1-overlap)))
    max_filters_y = int(y_res / (filter_height / (1-overlap)))
    centre_offset_x = filter_width / 2
    centre_offset_y = filter_height / 2

    corners_list = create_filter_boundaries(filter_width, filter_height, overlap)

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
                    print neuron_id
                    neuron_id_count[neuron_id] += 1
                    connection_list.append([pixel_value, neuron_id, base_weight, 1])
            xs.append(x)
            ys.append(y)
            visual_matrix.append(visual_row)
            split_matrix.append(split_row)

    for connection in range(len(connection_list)):
        connection_list[connection][2] /= neuron_id_count[connection_list[connection][1]] * percentage_fire_threshold

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(np.array(xs), np.array(ys), np.array(visual_matrix))
        plt.show()
    return connection_list

# generate_visual_field(3, 2, rotation=1, plot=True)
first_layer_connections_r0 = visual_field_with_overlap(100, 100, 0., plot=False)

print "done"









