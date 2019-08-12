import sys
import codecs
import numpy as np
import shapely.geometry as shgeo
import os
import re
import math
# import polyiou
"""
    some basic functions which are useful for process DOTA data
"""

wordname_18 = ['__background__',
            'airport', 'baseball-diamond', 'basketball-court', 'bridge', 'container-crane',
            'ground-track-field', 'harbor', 'helicopter', 'helipad', 'large-vehicle',
            'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
            'storage-tank', 'swimming-pool', 'tennis-court']


def dots4ToRec4(poly):
    xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
                            max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
                             min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
                             max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    return xmin, ymin, xmax, ymax


def dots4ToRec8(poly):
    xmin, ymin, xmax, ymax = dots4ToRec4(poly)
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]


def dotalabel2boxeslabel(dotalabel_dir, boxeslabel_dir):
    for file_name in os.listdir(dotalabel_dir):
        dotalabel_path = os.path.join(dotalabel_dir, file_name)
        with open(dotalabel_path, "r") as read_f:
            anno_list = read_f.readlines()
        # boxeslabel_path = os.path.join(boxeslabel_dir, file_name)
        # with open(boxeslabel_path, "w") as save_f:
            # anno_list = save_f.readlines()
        for anno_ins in anno_list:
            box = np.reshape(np.array(anno_ins.strip().split(' ')[:8]).astype(float), (-1, 2)).tolist()
            category = anno_ins.strip().split(' ')[8]
            difficult = int(anno_ins.strip().split(' ')[9])
            poly = dots4ToRec8(box)
            poly = np.reshape(np.array(poly).astype(float), (-1, 2)).tolist()
            print(box)
            print(poly)
            line = '{} {} {}'.format(poly, category, difficult)
            line = line.replace('[', '').replace(',', '').replace(']', '')
            # print(line)
            # save_f.writelines(line + '\n')
            break
        break
            # save_f.writelines(line + '\n')
        # save_f.close()

if __name__ =='__main__':
    dotalabel2boxeslabel("data/DOTA/val_2019/labelTxt", "data/DOTA/val_2019/boxesLabelTxt")