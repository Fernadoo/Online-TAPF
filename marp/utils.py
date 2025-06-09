import os
from dataclasses import dataclass

import numpy as np


def Manhattan(loc1, loc2):
    return np.sum(np.abs(np.array(loc1) - np.array(loc2)))


def r_Manhattan(loc1, dir1, loc2):
    dx, dy = np.array(loc2) - np.array(loc1)
    if dx < 0 and dy == 0:
        ddir = 1 * (dir1 == 0) + 0 * (dir1 == 90) + 1 * (dir1 == 180) + 2 * (dir1 == 270)
    elif dx < 0 and dy > 0:
        ddir = 1 * (dir1 == 0) + 1 * (dir1 == 90) + 2 * (dir1 == 180) + 2 * (dir1 == 270)
    elif dx < 0 and dy < 0:
        ddir = 2 * (dir1 == 0) + 1 * (dir1 == 90) + 1 * (dir1 == 180) + 2 * (dir1 == 270)
    elif dx == 0 and dy > 0:
        ddir = 0 * (dir1 == 0) + 1 * (dir1 == 90) + 2 * (dir1 == 180) + 1 * (dir1 == 270)
    elif dx == 0 and dy < 0:
        ddir = 2 * (dir1 == 0) + 1 * (dir1 == 90) + 0 * (dir1 == 180) + 1 * (dir1 == 270)
    elif dx > 0 and dy == 0:
        ddir = 1 * (dir1 == 0) + 2 * (dir1 == 90) + 1 * (dir1 == 180) + 0 * (dir1 == 270)
    elif dx > 0 and dy > 0:
        ddir = 1 * (dir1 == 0) + 2 * (dir1 == 90) + 2 * (dir1 == 180) + 1 * (dir1 == 270)
    elif dx > 0 and dy < 0:
        ddir = 2 * (dir1 == 0) + 2 * (dir1 == 90) + 1 * (dir1 == 180) + 1 * (dir1 == 270)
    elif dx == 0 and dy == 0:
        ddir = 0
    else:
        print(loc1, dir1, loc2)
        raise ValueError('Impossible location and direction!')

    return np.abs(dx) + np.abs(dy) + ddir



@dataclass
class Marker:
    CELL = 0
    BLOCK = 1
    IMPORT = 2
    EXPORT = 3
    TURNING = 4
    BATTERY = 8
    ACCESSIBLE = [CELL, TURNING, BATTERY, ]
    INACCESSIBLE = [BLOCK, IMPORT, EXPORT]


def parse_map_from_file(map_config):
    PREFIX = 'marp/layouts/'
    POSTFIX = '.map'
    if not os.path.exists(PREFIX + map_config + POSTFIX):
        raise ValueError('Map config does not exist!')
    layout = []
    with open(PREFIX + map_config + POSTFIX, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('#'):
                pass
            else:
                row = []
                for char in line:
                    if char == '.':
                        row.append(Marker.CELL)
                    elif char == '@':
                        row.append(Marker.BLOCK)
                    elif char == 'I':
                        row.append(Marker.IMPORT)
                    elif char == 'E':
                        row.append(Marker.EXPORT)
                    elif char == 'B':
                        row.append(Marker.BATTERY)
                    elif char == 'T':
                        row.append(Marker.TURNING)
                    else:
                        continue
                layout.append(row)
            line = f.readline()
    return np.array(layout)


def parse_locs(locs):
    ret = []
    for i, l in enumerate(locs):
        ret.append(eval(l.replace('_', ',')))
    return ret


def show_args(args):
    args = vars(args)
    for key in args:
        print(f'{key.upper()}:')
        print(args[key])
        print('-------------\n')
