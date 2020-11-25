
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import warnings
import spynnaker8 as p
from SpiNNiCub.ATIS.decode_events import *
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import imageio
from scipy.stats import multivariate_normal
import sys
from SpiNNiCub.connection_matrix import parse_ATIS, combine_parsed_ATIS, gather_all_ATIS_log

warnings.filterwarnings("error")

def rotate_matrix_90_anticlockwise(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0])-1, -1, -1)]

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

    print("plotting")

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

def parse_filter_data(file_location, file_name, filter_sizes):
    corner_list = {}
    video_dict = {}
    for filter_size in filter_sizes:
        relative_overlap = 10. / filter_size
        corner_list['{}'.format(filter_size)] = create_filter_boundaries(filter_size, filter_size, relative_overlap)
        for rotation in range(8):
            video_dict['f{}-r{}'.format(filter_size, rotation)] = []
    filter_data = np.load("{}/{}".format(file_location, file_name))
    # extract rotation, x, y, and spike time from label, rotation, id, time
    for spike in filter_data:
        rotation = int(spike[1])
        for filter_size in filter_sizes:
            if '{}'.format(filter_size) in spike[0]:
                spike_filter_size = filter_size
                break
        [x, y, maxx, maxy] = corner_list['{}'.format(spike_filter_size)][0][int(spike[2])]
        if 0 > x > 304 or 0 < y > 240:
            print("hol up")
        x -= spike_filter_size / 2
        y -= spike_filter_size / 2
        if 0 > x > 304 or 0 < y > 240:
            print("hol up")
        spike_time = int(float(spike[3]))
        video_dict['f{}-r{}'.format(spike_filter_size, rotation)].append([x, y, spike_time, spike_filter_size])
    return video_dict

def create_video(file_location, file_name, frame_rate, spikes=[], proto=True, rotate=True):
    if spikes:
        spike_data = spikes
        if spike_data == []:
            print("no spike data from list\nexiting without creating images and video")
            return None
    else:
        if proto:
            spike_data = np.load("{}/all extracted proto spikes {}.npy".format(file_location, file_name))
        else:
            spike_data = np.load("{}/filter rotations spikes {}.npy".format(file_location, file_name))
    frame_duration = 1000. / frame_rate
    if not len(spike_data) > 0:
        print("no spike data\nexiting without creating images and video")
        return None

    print("parsing data")
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

    print("binning frames")
    binned_frames = [[[0 for y in range(y_res)] for x in range(x_res)] for i in range(int(np.ceil(max_time / frame_duration)))]

    # setup toolbar
    toolbar_width = 40
    print('[{}]'.format('-'*toolbar_width))
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    current_point = 0
    progression_count = 0
    for spike in list_data:
        x = int(round(spike[0]))
        y = int(round(spike[1]))
        t = spike[2]
        filter_size = int(spike[3])
        time_index = int(t / frame_duration)
        if time_index == len(binned_frames):
            time_index -= 1
        elif time_index > len(binned_frames):
            Exception
        if filter_sizes[0] == 1:
            binned_frames[time_index][x][y] += 1
        else:
            gaussian = filter_gaussians['{}'.format(filter_size)]
            for i in range(len(gaussian)):
                for j in range(len(gaussian[0])):
                    new_x = x + i - int(filter_size / 2)
                    new_y = y + j - int(filter_size / 2)
                    if 0 <= new_x < x_res and 0 <= new_y < y_res:
                        binned_frames[time_index][new_x][new_y] += gaussian[i][j]
        progression_count += 1
        # print progression_count, '/', len(list_data)
        if current_point < int(round((float(progression_count) / float(len(list_data))) * float(toolbar_width))):
            current_point = int(round((float(progression_count) / float(len(list_data))) * float(toolbar_width)))
            sys.stdout.write("-")
            sys.stdout.flush()
    sys.stdout.write("]\n")

    print('creating images')
    xlim = 304
    ylim = 240
    filenames = []
    # setup toolbar
    toolbar_width = 40
    print('[{}]'.format('-'*toolbar_width))
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    current_point = 0
    progression_count = 0
    for frame in binned_frames:
        if rotate:
            new_frame = rotate_matrix_90_anticlockwise(frame)
        else:
            new_frame = frame
        # plt.figure(frameon=False)
        plt.imshow(new_frame, cmap= 'hot', interpolation='nearest', aspect='auto')
        title = '{} - {}'.format(file_name, binned_frames.index(frame))
        title += '.png'
        filenames.append(file_location+'/pictures_and_gifs/'+title)
        plt.axis('off')
        imageio.imsave(file_location+'/pictures_and_gifs/'+title, new_frame)
        # plt.savefig(file_location+'/videos/'+title, format='png', bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.clf()
        progression_count += 1
        if current_point < int(round((float(progression_count) / float(len(binned_frames))) * float(toolbar_width))):
            current_point = int(round((float(progression_count) / float(len(binned_frames))) * float(toolbar_width)))
            sys.stdout.write("-")
            sys.stdout.flush()
    sys.stdout.write("]\n")

    print("creating video")
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(file_location+'/pictures_and_gifs/videos/'+file_name+'.gif', images)

    # for file in filenames:
    #     os.remove(file)

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

def process_movement(file_location, file_name, frame_rate):
    spike_times = np.load("{}/extracted move spikes {}.npy".format(file_location, file_name))
    frame_duration = 1000. / frame_rate
    print("parsing data")
    list_data = []
    max_time = 0.
    for spike_data in spike_times:
        x, y, spike_time = spike_data
        list_data.append([x, y, spike_time])
        if spike_time > max_time:
            max_time = spike_time
    list_data.sort(key=lambda x: x[2])

    print("binning frames")
    binned_frames = [[[0 for y in range(3)] for x in range(3)] for i in range(int(np.ceil(max_time / frame_duration)))]

    # setup toolbar
    toolbar_width = 40
    print('[{}]'.format('-'*toolbar_width))
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    current_point = 0
    progression_count = 0
    for spike in list_data:
        x = int(spike[0])
        y = int(spike[1])
        t = spike[2]
        time_index = int(t / frame_duration)
        binned_frames[time_index][x][y] += 1
        progression_count += 1
        # print progression_count, '/', len(list_data)
        if current_point < int(round((float(progression_count) / float(len(list_data))) * float(toolbar_width))):
            current_point = int(round((float(progression_count) / float(len(list_data))) * float(toolbar_width)))
            sys.stdout.write("-")
            sys.stdout.flush()
    sys.stdout.write("]\n")

    print('creating images')
    xlim = 304
    ylim = 240
    filenames = []
    # setup toolbar
    toolbar_width = 40
    print('[{}]'.format('-'*toolbar_width))
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

    print("creating video")
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(file_location+'/videos/movement '+file_name+'.gif', images)

    # for file in filenames:
    #     os.remove(file)

if __name__ == '__main__':
    x_res = 304
    y_res = 240
    fovea_x = 170
    fovea_y = 135
    peripheral_x = (x_res - fovea_x) / 2
    peripheral_y = (y_res - fovea_y) / 2
    horizontal_split = 1
    veritcal_split = 1

    # connection configurations:
    filter_sizes = [100, 70, 55, 40]
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
    inhib = False  # [0]: +ve+ve, -ve-ve   [1]:+ve-ve, -ve+ve
    WTA = True
    to_wta = 1.
    from_wta = 1.
    self_excite = 0.01
    fake_full_ATIS = False

    simulate = 'solo'
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
    # create_video("run_data", label, 2)
    # process_movement("run_data", label, 2)

   # plot_spikes("run_data", label)
   #  contrasts = ['high', 'medium', 'low']
   #  locations = ['RL', 'LR', 'BT', 'TB']
   #  combined_events = []
   #  for contrast in contrasts:
   #      print contrast, ':'
   #      for location in locations:
   #          print "\t", location
   #          combined_events.append(parse_ATIS('ATIS/{}/{}'.format(contrast, location), 'decoded_events.txt'))
   #  events = combine_parsed_ATIS(combined_events)
   #  spikes = parse_events_to_spike_times(events)
   #  create_video('run_data', 'input spikes', 2, spikes=spikes)

    video_of = 'solo'
    if video_of == 'raw':
        all_directories = gather_all_ATIS_log('ATIS/IROS_from Giulia')
        combined_events = []
        for directory in all_directories:
            print('extracting directory', all_directories.index(directory) + 1, '/', len(all_directories))
            combined_events.append(parse_ATIS(directory, 'decoded_events.txt'))
        events = combine_parsed_ATIS(combined_events)
        spikes = parse_events_to_spike_times(events)
        filter_sizes = [1]
        create_video('ATIS/IROS_from Giulia', 'All rotated ATIS input spikes', 2, spikes=spikes)
    elif video_of == 'solo_raw':
        all_directories = []
        for root, dirs, files in os.walk('ATIS'):
            if 'decoded_events.txt' in files:
                print(root)
                events = parse_ATIS(root, 'decoded_events.txt')
                spikes = parse_events_to_spike_times(events)
                if 'videos' not in dirs:
                    os.mkdir(root+'/videos')
                create_video(root, 'events', 2, spikes=spikes)
    elif video_of == 'filter_raw':
        all_directories = []
        for root, dirs, files in os.walk('run_data/filter_data'):
            for file in files:
                if 'rotation' in file:
                    spike_dict = parse_filter_data(root, file, filter_sizes)
                if 'videos' not in dirs:
                    os.mkdir(root + '/videos')
                for param_setting in spike_dict:
                    if spike_dict[param_setting]:
                        create_video(root, file+param_setting, 2, spikes=spike_dict[param_setting])
    elif video_of == 'solo':
        filter_sizes = [100, 70, 55, 40]
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
        inhib = False  # [0]: +ve+ve, -ve-ve   [1]:+ve-ve, -ve+ve
        WTA = True
        to_wta = 1.
        from_wta = 1.
        self_excite = 0.01

        simulate = 'subset'
        # simulate = None
        label = "{} fs-{} ol-{} w-{} bft-{} sft-{} fft-{} ift-{} icp-{} ps-{} in-{}".format(simulate, filter_split,
                                                                                            overlap,
                                                                                            base_weight,
                                                                                            boarder_percentage_fire_threshold,
                                                                                            segment_percentage_fire_threshold,
                                                                                            filter_percentage_fire_threshold,
                                                                                            inhib_percentage_fire_threshold,
                                                                                            inhib_connect_prob,
                                                                                            proto_scale,
                                                                                            inhib)
        if WTA:
            label += ' to-{} from-{}'.format(to_wta, from_wta)
        if self_excite:
            label += ' self-{}'.format(self_excite)
        label += ' {}'.format(filter_sizes)
        # label = 'IROS fs-4 ol-0.6 w-5.0 bft-0.2 sft-0.02 fft-0.8 ift-0.02 icp-1.0 ps-0.75 in-all [100, 70, 55, 40]'
        # label = 'subset fs-4 ol-0.6 w-5.0 bft-0.2 sft-0.03 fft-0.8 ift-0.04 icp-1.0 ps-0.75 in-all [104, 73, 51]'
        # label = 'proto fs-4 ol-0.9 w-5.0 bft-0.2 sft-0.02 fft-0.8 ift-0.02 icp-1.0 ps-0.75 in-all [104, 73, 51, 36]'
        # label = 'proto fs-4 ol-0.6 w-5.0 bft-0.2 sft-0.03 fft-0.8 ift-0.03 icp-1.0 ps-0.75 in-all [30, 46, 70, 100]'
        label = 'g_subset bos0.5 fs-4 ol-0.7 w-5.0 bft-0.2 sft-0.02 fft-0.8 ift-0.005 icp-1.0 ps-0.75 in-all [104, 73, 51, 36, 25]'
        # label = 'g_subset bos0.5 fs-4 ol-0.6 w-5.0 bft-0.2 sft-0.02 fft-0.8 ift-0.005 icp-1.0 ps-0.75 in-all [100, 70, 50, 30]'
        print(label)
        filter_sizes = [104, 73, 51, 36, 25]
        # filter_sizes = [100, 70, 50, 30]
        # filter_sizes = [1]
        create_video("run_data", label, 2, proto=True, rotate=True)
        # process_movement("run_data", label, 2)
    elif video_of == 'IROS':
        seperated_list = ['calib_circles',
                          'no_obj',
                          'obj',
                          '019',
                          '029',
                          '085',
                          '157',
                          'multi_objects_saccade1',
                          'object_clutter',
                          'object_clutter2',
                          'objects_approaching',
                          'objects_approaching_no_saccade',
                          'paddle_moving_clutter'
                          ]
        filter_sizes = [104, 73, 51, 36, 25]
        # label = ' IROS fs-4 ol-0.6 w-5.0 bft-0.2 sft-0.02 fft-0.8 ift-0.02 icp-1.0 ps-0.75 in-all [100, 70, 55, 40]'
        label = ' IROS strd fs-4 ol-0.8 w-5.0 bft-0.2 sft-0.02 fft-0.8 ift-0.005 icp-1.0 ps-0.75 in-all [104, 73, 51, 36, 25]'
        # label = ' pytorch IROS fs-4 ol-0.6 w-5.0 bft-0.2 sft-0.02 fft-0.8 ift-0.02 icp-1.0 ps-0.75 in-all [104, 73, 51, 36]'
        count = 1
        for test in seperated_list:
            print("starting: " + test + " - {}/{}".format(count, len(seperated_list)))
            create_video("data_for_processing", test+label, 20, proto=True, rotate=True)
            count += 1
    else:
        sfts = [0.04, 0.02]
        bfts = [0.005, 0.05]
        ifts = [0.02, 0.03, 0.04]
        to_wtas = [1., 0.1, 0.01]
        from_wtas = [1., 0.1, 0.01]
        fsizess = [[100, 90, 80, 70, 60, 50, 40, 30, 20], [100, 90, 80, 70, 60, 50, 40, 30, 20]]
        to_and_from = [0, 10, 100]

        # for sft in sfts:
        #     for bft in bfts:
        for to_w in to_wtas:
            for from_w in from_wtas:
            # for to_from in to_and_from:
            #     to_wta = to_from
            #     from_wta = to_from
                # segment_percentage_fire_threshold = sft
                # boarder_percentage_fire_threshold = bft
                # inhib_percentage_fire_threshold = ift
                to_wta = to_w
                from_wta = from_w
                label = "{} fs-{} ol-{} w-{} bft-{} sft-{} fft-{} ift-{} icp-{} ps-{} in-{}".format(simulate,
                                                                                                    filter_split,
                                                                                                    overlap,
                                                                                                    base_weight,
                                                                                                    boarder_percentage_fire_threshold,
                                                                                                    segment_percentage_fire_threshold,
                                                                                                    filter_percentage_fire_threshold,
                                                                                                    inhib_percentage_fire_threshold,
                                                                                                    inhib_connect_prob,
                                                                                                    proto_scale,
                                                                                                    inhib)
                if WTA:
                    label += ' to-{} from-{}'.format(to_wta, from_wta)
                if self_excite:
                    label += ' self-{}'.format(self_excite)
                label += ' {}'.format(filter_sizes)
                print(label)
                create_video("run_data", label, 2)
                # process_movement("run_data", label, 2)

# inhib_percentage_fire_threshold = 0.1
# values = [0.08, 0.06, 0.04, 0.02]
# for value in values:
#     print "current value:", value
#     segment_percentage_fire_threshold = value
#     label = "{} fs-{} ol-{} w-{} bft-{} sft-{} fft-{} ift-{} icp-{} ps-{} in-{} {}".format(simulate, filter_split, overlap,
#                                                                                            base_weight,
#                                                                                            boarder_percentage_fire_threshold,
#                                                                                            segment_percentage_fire_threshold,
#                                                                                            filter_percentage_fire_threshold,
#                                                                                            inhib_percentage_fire_threshold,
#                                                                                            inhib_connect_prob, proto_scale,
#                                                                                            inhib, filter_sizes)
#     create_video("run_data", label, 2)

print("done")

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

