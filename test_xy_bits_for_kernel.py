import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import warnings
import spynnaker8 as p
from ATIS.decode_events import *
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt



x_res = 304
y_res = 240

# retina bits
# y8e2x9p1 (int(y) << 12) + (int(x) << 1) + 1
y_bits = 8
x_bits = 9
p_bits = 1
e_bits = 2

# plt.figure()
polarity_0 = [[], []]
polarity_1 = [[], []]
not_in_use = [[], []]
for i in range(np.power(2, x_bits+p_bits+e_bits)):
    for j in range(np.power(2, y_bits)):
        ATIS_id = i + (j << 12)
        x = (i >> 1) & 511
        y = j
        p = i & 1
        e = i >> 10
        if y >= y_res or e or x >= x_res:
            # plt.scatter(i, j, c='red')
            not_in_use[0].append(i)
            not_in_use[1].append(j)
        elif p:
            print "x:", x, "y:", y, "e:", e, "p:", p
            polarity_1[0].append(i)
            polarity_1[1].append(j)
        else:
            print "x:", x, "y:", y, "e:", e, "p:", p
            polarity_0[0].append(i)
            polarity_0[1].append(j)
    print "finished", i+1, "/", np.power(2, x_bits+p_bits+e_bits)

print "plotting"
plt.scatter(polarity_0[0], polarity_0[1], c='blue')
plt.scatter(polarity_1[0], polarity_1[1], c='green')
plt.scatter(not_in_use[0], not_in_use[1], c='red')
print"showing"
plt.show()