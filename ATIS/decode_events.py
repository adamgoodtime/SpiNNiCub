

import re
import os
import struct
import numpy as np
from tqdm import tqdm


class DataManager:
    def __init__(self):
        self.pattern = re.compile('(\d+) (\d+\.\d+) AE \((.*)\)')


    def decode_skinEvents(self, data):
        # Events are encoded as 32 bits with polarity(p), taxel(t), cross_base(c),
        # body_part(b), side(s), type(T) and skin(S) as shown below. Reserved bits are represented with (R)

        # 0000 0000 STsR RRbb bRRc RRttt tttt tttp

        if data[-1,0] == 0:
            data = np.delete(data, -1)

        timestamps = data[:, 0] & ~(0x1 << 31)

        polarity = data[:, 1] & 0x01
        data[:, 1] >>= 1
        taxel = data[:, 1] & 0x3FF
        data[:, 1] >>= 12
        cross_base = data[:, 1] & 0x01
        data[:, 1] >>= 3
        body_part = data[:, 1] & 0x07
        data[:, 1] >>= 6
        side = data[:, 1] & 0x01
        data[:, 1] >>= 1
        type = data[:, 1] & 0x01
        data[:, 1] >>= 1
        skin = data[:, 1] & 0x01

        return np.vstack([timestamps, polarity, taxel, cross_base, body_part, side, type, skin]).T.astype(np.float)




    def decode_events_24bit(self, data):
        # Events are encoded as 32 bits with x,y,channel(c) and polarity(p) as shown below
        # 0000 0000 tcrr yyyy yyyy rrxx xxxx xxxp

        #if data[-1] == 0 or len(data) % 2 != 0:
        #    data = np.delete(data, -1)
        #data = data.reshape((-1, len(data)))

        timestamps = data[:, 0] & ~(0x1 << 31)

        polarity = data[:, 1] & 0x01

        data[:, 1] >>= 1
        x = data[:, 1] & 0x1FF

        data[:, 1] >>= 11
        y = data[:, 1] & 0xFF

        data[:, 1] >>= 10
        channel = data[:, 1] & 0x01
        return np.vstack([timestamps, channel, x, y, polarity]).T.astype(np.float)


    def decode_events_bit(self, str_rep):
        # Events are encoded as 32 bits with x,y,channel(c) and polarity(p) as shown below
        # 0000 0000 tcrr yyyy yyyy rrxx xxxx xxxp

        k = list(map(int, str_rep.split()))
        data = np.array(k, dtype=np.uint32)
        # if data[-1] == 0 or len(data) % 2 != 0:
        #     data = np.delete(data, -1)
        data = data.reshape((-1, len(data)))

        timestamps = data[:, 0] & ~(0x1 << 31)

        polarity = data[:, 1] & 0x01

        data[:, 1] >>= 1
        x = data[:, 1] & 0x1FF

        data[:, 1] >>= 11
        y = data[:, 1] & 0xFF

        data[:, 1] >>= 10
        channel = data[:, 1] & 0x01
        return np.vstack([timestamps, channel, x, y, polarity]).T.astype(np.float)

    def load_AE_from_yarp(self, AE_file_path):
        pattern = re.compile('(\d*) (\d*.\d*) AE \((.*)\)')
        AE_to_save = []
        with open(os.path.join(AE_file_path, 'data.log')) as boxFile:
            content = boxFile.read()
            found = pattern.findall(content)
            for b in tqdm(found):
                b_num = b[0]
                b_ts = b[1]
                ev = np.array(b[2].split(' '), dtype=np.uint32)
                ev = ev.reshape(int(len(ev)/2), 2)
                # timestamps, channel, x, y, polarity = self.decode_events_24bit(ev)
                AE_to_save.append(np.array(self.decode_events_24bit(ev)))       # SELECT WHAT TO SAVE
                # AE_to_save.append(np.array([timestamps, channel, x, y, polarity]))       # SELECT WHAT TO SAVE
        AE_to_save = np.vstack(AE_to_save)
        AE_to_save[:, 0] = (AE_to_save[:, 0] - AE_to_save[0, 0]) * 80e-9  # 80ns to normalize w.r.t. the clock
        np.savetxt(os.path.join(AE_file_path, 'decoded_events.txt'), AE_to_save, delimiter=',', fmt=['%f', '%d', '%d', '%d', '%d'])     # SPECIFY THE FORMAT OF THE DATA



    def load_GAE_from_yarp(self, GAE_file_path):
        pattern = re.compile('(\d*) (\d*.\d*) GAE \((\d*) (\d*) \d{1} (\d*) (\d*) (\d*)\)')
        GAE_to_save = []
        with open(os.path.join(GAE_file_path, 'data.log')) as boxFile:
            content = boxFile.read()
            found = pattern.findall(content)
            for b in found:
                b_num = b[0]
                b_ts = b[1]
                v_ts = b[2]
                v = b[3]
                radius = np.float32(struct.unpack("<f", struct.pack("<I", np.int(
                    b[4]))))  # converts the integer to binary and then to float32
                # tw = struct.unpack("<f", struct.pack("<i", np.int(b[5])))
                # circle = struct.unpack("<f", struct.pack("<i", np.int(b[6])))
                ts, ch, x, y, p = self.decode_events(' '.join([v_ts, v]))[0]  # TODO ONLY WORKS FOR ONE EVENT PER BOTTLE
                GAE_to_save.append(np.array([ts, x, y, radius, p]))
        GAE_to_save = np.array(GAE_to_save)
        GAE_to_save[:, 0] = (GAE_to_save[:, 0] - GAE_to_save[0, 0]) * 80e-9  # 80ns to normalize w.r.t. the clock
        np.savetxt(os.path.join(GAE_file_path, "decoded_events.txt"), GAE_to_save, delimiter=',',
                   fmt=['%f', '%d', '%d', '%d', '%d'])


    def load_SkinEvents_from_yarp(self, AE_file_path):
        pattern = re.compile('(\d*) (\d*.\d*) AE \((.*)\)')
        AE_to_save = []
        with open(os.path.join(AE_file_path, 'data.log')) as boxFile:
            content = boxFile.read()
            found = pattern.findall(content)
            for b in tqdm(found):
                b_num = b[0]
                b_ts = b[1]
                ev = np.array(b[2].split(' '), dtype=np.uint32)
                ev = ev.reshape(int(len(ev)/2), 2)
                ts, pol, tax, cross_b, body_part, side, type, skin = self.decode_skinEvents(ev)[0]
                AE_to_save.append(np.array([ts, pol, tax, cross_b, body_part, side, type, skin]))       # SELECT WHAT TO SAVE
        AE_to_save = np.array(AE_to_save)
        AE_to_save[:, 0] = (AE_to_save[:, 0] - AE_to_save[0, 0]) * 80e-9  # 80ns to normalize w.r.t. the clock
        np.savetxt(os.path.join(AE_file_path, 'decoded_events.txt'), AE_to_save, delimiter=',', fmt=['%f', '%d', '%d', '%d', '%d', '%d', '%d', '%d'])     # SPECIFY THE FORMAT OF THE DATA



if __name__ == '__main__':

    
    dm = DataManager()
    dm.load_AE_from_yarp('data')

