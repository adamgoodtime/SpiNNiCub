import matplotlib
matplotlib.use('Agg')

import warnings
import numpy as np
import spynnaker8 as p
import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

warnings.filterwarnings("error")

x_res = 304.
y_res = 240.

shape_pre = np.asarray([x_res, y_res])
shape_kernel = np.asarray([100, 100])
pre_sample_steps = shape_kernel * (1. - 0.6)
start_location = np.asarray([50, 50])
shape_post = np.asarray([int((x_res - shape_kernel[0]) / pre_sample_steps[0]) + 1, int((y_res - shape_kernel[1]) / pre_sample_steps[1]) + 1])
weight_kernel = []
for i in range(int(shape_kernel[0])):
    weight_row = []
    for j in range(int(shape_kernel[1])):
        x_value = ((float(shape_kernel[0]) / 2.) - np.abs(i - (float(shape_kernel[0]) / 2.))) / (float(shape_kernel[0]) / 2.)
        y_value = ((float(shape_kernel[1]) / 2.) - np.abs(j - (float(shape_kernel[1]) / 2.))) / (float(shape_kernel[1]) / 2.)
        weight_row.append(x_value * y_value)
    weight_kernel.append(weight_row)
weight_kernel = np.asarray(weight_kernel)

print "shape_pre", shape_pre
print "shape_post", shape_post
print "shape_kernel", shape_kernel
print "pre_sample_steps", pre_sample_steps
print "start_location", start_location
print "weight_kernel", weight_kernel

p.setup(timestep=1)
p.set_number_of_neurons_per_core(p.IF_curr_exp(), 128)

spike_times = [[i] for i in range(int(x_res*y_res))]
pre_pop = p.Population(x_res*y_res, p.SpikeSourceArray(spike_times=spike_times))
pre_pop.record('all')

post_pop = p.Population(shape_post[0]*shape_post[1], p.IF_curr_exp())
post_pop.record('all')

proj = p.Projection(pre_pop, post_pop, p.KernelConnector(shape_pre=shape_pre,
                                                         shape_post=shape_post,
                                                         shape_kernel=shape_kernel,
                                                         weight_kernel=weight_kernel,
                                                         post_sample_steps_in_pre=pre_sample_steps,
                                                         post_start_coords_in_pre=start_location))

runtime = x_res*y_res
p.run(runtime)

pre_spikes = pre_pop.get_data('all')
post_spikes = post_pop.get_data('all')

Figure(
    Panel(post_spikes.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=post_pop.label, yticks=True, xlim=(0, runtime)),
    Panel(post_spikes.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=post_pop.label, yticks=True, xlim=(0, runtime)),
    Panel(post_spikes.segments[0].filter(name='gsyn_inh')[0],
          xlabel="Time (ms)", xticks=True,
          ylabel="gsyn inhibitory (mV)",
          data_labels=post_pop.label, yticks=True, xlim=(0, runtime)),
    Panel(post_spikes.segments[0].spiketrains,
          ylabel="Output Spikes",
          data_labels=post_pop.label, yticks=True, xlim=(0, runtime)),
    # Panel(pre_spikes.segments[0].spiketrains,
    #       ylabel="Input Spikes",
    #       data_labels=pre_pop.label, yticks=True, xlim=(0, runtime)),
    title="Spike data"
)
plt.show()
Figure(
    # Panel(pre_spikes.segments[0].filter(name='v')[0],
    #       ylabel="Membrane potential (mV)",
    #       data_labels=pre_pop.label, yticks=True, xlim=(0, runtime)),
    # Panel(pre_spikes.segments[0].filter(name='gsyn_exc')[0],
    #       ylabel="gsyn excitatory (mV)",
    #       data_labels=pre_pop.label, yticks=True, xlim=(0, runtime)),
    # Panel(pre_spikes.segments[0].filter(name='gsyn_inh')[0],
    #       xlabel="Time (ms)", xticks=True,
    #       ylabel="gsyn inhibitory (mV)",
    #       data_labels=pre_pop.label, yticks=True, xlim=(0, runtime)),
    Panel(pre_spikes.segments[0].spiketrains,
          ylabel="Output Spikes",
          data_labels=pre_pop.label, yticks=True, xlim=(0, runtime)),
    title="Single eprop neuron"
)
plt.show()





