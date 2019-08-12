import dota_utils as util
import os
import cv2
import json
import shutil
import numpy as np
import math
from DOTA_devkit import polyiou
# import collections
import csv

wordname_18 = ['__background__',
            'airport', 'baseball-diamond', 'basketball-court', 'bridge', 'container-crane',
            'ground-track-field', 'harbor', 'helicopter', 'helipad', 'large-vehicle',
            'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
            'storage-tank', 'swimming-pool', 'tennis-court']


def DOTA2COCO(srcpath, txt_file_path):
    imageparent = os.path.join(srcpath, 'images')
    image_id = 0
    filenames = [os.path.join(imageparent, filename) for filename in os.listdir(imageparent) if filename.endswith('.png')]
    imgsize_dict = {}
    imgratio_dict = {}
    for imagepath in filenames:
        # print(file)
        image_id = image_id + 1
        # images
        img = cv2.imread(imagepath)
        height, width, c = img.shape

        imgshortsize = min(width, height)
        ratio = float('%.2f' %(int(100*width / height + 0.5)/100))
        if imgshortsize not in imgsize_dict.keys():
            imgsize_dict[imgshortsize] = 1
        else:
            imgsize_dict[imgshortsize] += 1
        if ratio not in imgratio_dict.keys():
            imgratio_dict[ratio] = 1
        else:
            imgratio_dict[ratio] += 1

    print('img:{}, keys_num:{}'.format(image_id, len(imgsize_dict.keys())))

    size_sort = list(imgsize_dict.keys())#
    size_sort.sort()

    with open(txt_file_path, "w") as save_f:
        line = 'pic_number:{}  size_num:{}'.format(image_id, len(imgsize_dict.keys()))
        save_f.writelines(line + '\n')
        line = 'image_size  pic_number'
        save_f.writelines(line + '\n')
        inter_num = 0
        for key in size_sort:
            inter_num += imgsize_dict[key]
            line = str(float('%.02f' % key)) + ',' + str(inter_num/image_id)
            save_f.writelines(line + '\n')
    save_f.close()

if __name__ == '__main__':
    # # DOTA2COCO(r'./data/DOTA/data_ori/val', './val_cpt.txt')
    # DOTA2COCO(r'./data/DOTA/data_ori/test', './test_cpt.txt')
    # # DOTA2COCO(r'./data/DOTA/data_ori/train', './train_cpt.txt')
    DOTA2COCO(r'./data/DOTA/data_ori/val', './val_shortsize.txt')
    DOTA2COCO(r'./data/DOTA/data_ori/test', './test_shortsize.txt')
    DOTA2COCO(r'./data/DOTA/data_ori/train', './train_shortsize.txt')