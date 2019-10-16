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

def banana(x, y, a=1, b=100):
    return 1. / (np.power(a - x, 2) + (b*np.power(y - np.power(x, 2), 2)))

def circle(x, y, radius=0.5, origin_x=0.25, origin_y=0.25, stdev=0.1):
    if y > origin_y*2:
        circle_value = 0 #y - origin_y
    else:
        circle_value = np.power(x - np.sqrt(origin_x), 2) + np.power(y - np.sqrt(origin_y), 2)
    distance_from_circle = abs(radius - circle_value)
    # some gauss stuffx`
    return 1./(distance_from_circle+0.1), norm.ppf(distance_from_circle+0.001)
    # return distance_from_circle, norm.ppf(distance_from_circle+0.001)

def VM(x, y, r=10, p=0.08, theta=0):
    x *= 50
    y *= 50
    numerator = np.exp(p * r * np.cos(np.arctan2(-y, x) - theta))
    # if np.square(x) + np.square(y) < r:
    #     denominator = np.sqrt(abs(np.square(x) + np.square(y) - r))
    # elif np.square(x) + np.square(y) == r:
    #     denominator = 0.01
    # else:
    denominator = np.i0(np.sqrt(np.square(x) + np.square(y)) - r)
    if denominator == 0:
        denominator = 0.01
    return numerator / denominator


resolution = 100
banana_matrix = []
value_matrix = []
xs = []
ys = []
for i in range(-resolution, resolution):
    x = []
    y = []
    banana_row = []
    value_row = []
    for j in range(-resolution, resolution):
        x.append(float(i) / float(resolution))
        y.append(float(j) / float(resolution))
        # value, banan = circle(float(i)/float(resolution), float(j)/float(resolution))
        banan = VM(float(i)/float(resolution), float(j)/float(resolution))
        banana_row.append(round(banan, 2))
        # value_row.append(round(value, 2))
    xs.append(x)
    ys.append(y)
    banana_matrix.append(banana_row)
    # value_matrix.append(value_row)


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Grab some test data.
X, Y, Z = axes3d.get_test_data(0.05)
#
# # Plot a basic wireframe.
# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
#
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(np.array(xs), np.array(ys), np.array(banana_matrix))
# ax.plot_wireframe(np.array(xs), np.array(ys), np.array(value_matrix))

plt.show()

print "done"






