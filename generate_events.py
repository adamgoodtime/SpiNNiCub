
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


def FakeStimuliBarMoving(camera_resolution, bar_speed, space_metric, bar_dimensions, direction, contrast):

    time = space_metric / bar_speed
    timestamp = 0
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

        for x in range(camera_resolution[0], 0, -1):
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

        for y in range (camera_resolution[1], 0, -1):
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

    return eventsON, eventsOFF

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

if __name__ == '__main__':
    x_res = 304
    y_res = 240

    on, off = FakeStimuliBarMoving([x_res, y_res], 1., 40., [50, 50], 'LR', 'BlackOverWhite')

    events = parse_event_class(on, off)

    print "done"

