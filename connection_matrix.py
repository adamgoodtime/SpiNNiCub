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

def generate_visual_field(filter_width, filter_height, filter_split=4, rotation=0, plot=False):
    # create inputs of -1 to 1 for entire visual field, assuming no overlap for now
    filter_x_res = x_res / filter_width
    filter_y_res = y_res / filter_height
    connection_list = [[] for i in range(filter_width * filter_height * filter_split)]
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
                neuron_index = (int(y_res / filter_y_res) * filter_width) + int(x_res / filter_x_res) + filter_split
                connection_list.append([convert_pixel_to_id(i, j), neuron_index])
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

# generate_visual_field(3, 2, rotation=1, plot=True)



# print zero_2pi_tan(1, 1)
# print zero_2pi_tan(0, 1)
# print zero_2pi_tan(-1, 1)
# print zero_2pi_tan(-1, 0)
# print zero_2pi_tan(-1, -1)
# print zero_2pi_tan(-0.01, -1)
# print zero_2pi_tan(0, -1)
# print zero_2pi_tan(0.01, -1)
# print zero_2pi_tan(1, -1)
# print zero_2pi_tan(1, 0)

print "done"






