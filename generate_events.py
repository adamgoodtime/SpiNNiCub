import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


BAR_SPEED = [0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # total_space/total_period #mm/ms
space_metric = 1  # pixel
camera_resolution = [128, 128]
bar_dimensions = [50, 50]

direction = 'LR'
# direction = 'RL'
# direction = 'BT'
# direction = 'TB'


# contrast = 'BlackOverWhite'
contrast = 'WhiteOverBlack'

class Event(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.timestamp = 0
        self.polarty = 0


def FakeStimuliBarMoving(camera_resolution, bar_speed, space_metric, bar_dimensions, direction, contrast, timestamp):

    time = space_metric / bar_speed
    eventsON = []
    eventsOFF = []

    if contrast == 'BlackOverWhite':
        polarity_front = 0
        polarity_back = 1
    elif contrast == 'WhiteOverBlack':
        polarity_front = 1
        polarity_back = 0

    if direction == 'LR':

        # place the bar in the center
        ybar_start = ((camera_resolution[1]/2)-(bar_dimensions[1]/2))
        ybar_end = ybar_start+bar_dimensions[1]

        for x in range (0,camera_resolution[0]):
            for y in range(ybar_start,ybar_end):
                if(x-bar_dimensions[0]>0):

                    timestamp = timestamp+((time/camera_resolution[1])/2)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x-bar_dimensions[0]
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                else:
                    timestamp = timestamp+(time/camera_resolution[1])

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

            timestamp = timestamp+time

    elif direction == 'RL':

        # place the bar in the center
        ybar_start = ((camera_resolution[1] / 2) - (bar_dimensions[1] / 2))
        ybar_end = ybar_start + bar_dimensions[1]

        for x in range(camera_resolution[0] - 1, -1, -1):
            for y in range(ybar_start, ybar_end):
                if (x - bar_dimensions[0] > 0):

                    timestamp = timestamp + ((time / camera_resolution[1]) / 2)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x - bar_dimensions[0]
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                else:
                    timestamp = timestamp + (time / camera_resolution[1])

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

            timestamp = timestamp + time

    elif direction == 'TB':

        # place the bar in the center
        xbar_start = ((camera_resolution[0]/2)-(bar_dimensions[0]/2))
        xbar_end = xbar_start+bar_dimensions[0]

        for y in range (0, camera_resolution[1]):
            for x in range(xbar_start,xbar_end):
                if(y-bar_dimensions[1]>0):

                    timestamp = timestamp+((time/camera_resolution[1])/2)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x-bar_dimensions[0]
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                else:
                    timestamp = timestamp+(time/camera_resolution[1])

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

            timestamp = timestamp+time

    elif direction == 'BT':

        # place the bar in the center
        xbar_start = ((camera_resolution[0]/2)-(bar_dimensions[0]/2))
        xbar_end = xbar_start+bar_dimensions[0]

        for y in range (camera_resolution[1] - 1, -1, -1):
            for x in range(xbar_start,xbar_end):
                if(y-bar_dimensions[1]>0):

                    timestamp = timestamp+((time/camera_resolution[1])/2)

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

                    event = Event()
                    event.x = x-bar_dimensions[0]
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_back
                    eventsOFF.append(event)
                else:
                    timestamp = timestamp+(time/camera_resolution[1])

                    event = Event()
                    event.x = x
                    event.y = y
                    event.timestamp = timestamp
                    event.polarity = polarity_front
                    eventsON.append(event)

            timestamp = timestamp+time

    return eventsON, eventsOFF, timestamp

def parse_event_class(eventsON, eventsOFF):
    events = [[] for i in range(x_res*y_res)]
    for event in eventsON:
        x = event.x
        y = event.y
        timestamp = event.timestamp / 1000.
        polarity = event.polarity
        print x, y, timestamp
        events[convert_pixel_to_id(x, y)].append(timestamp)
    for event in eventsOFF:
        x = event.x
        y = event.y
        timestamp = event.timestamp / 1000.
        polarity = event.polarity
        print x, y, timestamp
        events[convert_pixel_to_id(x, y)].append(timestamp)
    return events

def convert_pixel_to_id(x, y):
    return (y*x_res) + x

def list_circle_xy(center_x, center_y, r, x_res=304, y_res=240):
    point_list = []
    for x in range(-r, r):
        y = int(np.sqrt(np.square(r) - np.square(x)))
        point_list.append([x + center_x, y + center_y])
        point_list.append([x + center_x, -y + center_y])
    for y in range(-r, r):
        x = int(np.sqrt(np.square(r) - np.square(y)))
        if [x + center_x, y + center_y] not in point_list:
            point_list.append([x + center_x, y + center_y])
        if [-x + center_x, y + center_y] not in point_list:
            point_list.append([-x + center_x, y + center_y])

    # xs = []
    # ys = []
    # for point in point_list:
    #     xs.append(point[0])
    #     ys.append(point[1])
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(xs, ys)
    # plt.xlim(0, x_res)
    # plt.ylim(0, y_res)
    # plt.show()

    return point_list

def fake_circle_moving(direction, r=30, mspp=1, speed=1, start_time=0, circle_contrast='black'):
    x_res = 304
    y_res = 240
    if direction == 'up':
        start_x = x_res / 2
        start_y = r
        speed_x = 0
        speed_y = speed
        steps = y_res - (r * 2)
    elif direction == 'down':
        start_x = x_res / 2
        start_y = y_res - r
        speed_x = 0
        speed_y = -speed
        steps = y_res - (r * 2)
    elif direction == 'left':
        start_x = x_res - r
        start_y = y_res / 2
        speed_x = -speed
        speed_y = 0
        steps = x_res - (r * 2)
    elif direction == 'right':
        start_x = r + 1
        start_y = y_res / 2
        speed_x = speed
        speed_y = 0
        steps = x_res - (r * 2)
    else:
        print "incorrect direction"
        Exception
    if circle_contrast == 'black':
        pixel_values = [[1 for i in range(y_res)] for j in range(x_res)]
        new_pixel_value = 0
        old_pixel_value = 1
    else:
        pixel_values = [[0 for i in range(y_res)] for j in range(x_res)]
        new_pixel_value = 1
        old_pixel_value = 0
    pixel_list = list_circle_xy(start_x, start_y, r)
    new_pixel_list = []
    on_events = []
    off_events = []
    for step in range(steps):
        time = start_time + step*mspp
        if step:
            # loop through old circle values and remove changes
            for [x, y] in pixel_list:
                if 0 <= x < x_res and 0 <= y < y_res:
                    if [x, y] not in new_pixel_list:
                        pixel_values[x][y] = old_pixel_value
                        if old_pixel_value:
                            on_events.append([x, y, time])
                        else:
                            off_events.append([x, y, time])
            pixel_list = deepcopy(new_pixel_list)
            new_pixel_list = []
        # loop through circle values and add changes
        for [x, y] in pixel_list:
            new_pixel_list.append([x + speed_x, y + speed_y])
            if 0 <= x < x_res and 0 <= y < y_res:
                if pixel_values[x][y] == old_pixel_value:
                    pixel_values[x][y] = new_pixel_value
                    if new_pixel_value:
                        on_events.append([x, y, time])
                    else:
                        off_events.append([x, y, time])
    return on_events, off_events, time


if __name__ == '__main__':
    x_res = 304
    y_res = 240

    fake_circle_moving('up')
    fake_circle_moving('down')
    fake_circle_moving('left')
    fake_circle_moving('right')

    on, off = FakeStimuliBarMoving([x_res, y_res], 1., 40., [50, 50], 'LR', 'BlackOverWhite')

    events = parse_event_class(on, off)

    print "done"

