import dota_utils as util
import os
import cv2
import json
import shutil
import numpy as np
import math
from DOTA_devkit import polyiou

wordname_18 = ['__background__',
            'airport', 'baseball-diamond', 'basketball-court', 'bridge', 'container-crane',
            'ground-track-field', 'harbor', 'helicopter', 'helipad', 'large-vehicle',
            'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
            'storage-tank', 'swimming-pool', 'tennis-court']


def DOTA2COCO(srcpath, destfile):
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    info = {'contributor': 'captain group',
           'data_created': '2019',
           'description': 'Object detection for aerial pictures.',
           'url': 'http://rscup.bjxintong.com.cn/#/theme/2',
           'version': 'preliminary contest',
           'year': 2019}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(wordname_18):
        if name =='__background__':
            continue
        single_cat = {'id': idex , 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    # file_list = []
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            # annotations
            objects = util.parse_dota_poly2(file)
            if not len(objects):
                continue
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = wordname_18.index(obj['name'])
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            # images
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
            # file_list.append(basename + '.png')
            # if image_id>=200:
            #     break
        print('img:{}, ins:{}'.format(image_id-1, inst_count-1))
        json.dump(data_dict, f_out)


if __name__ == '__main__':
    DOTA2COCO(r'./data/DOTA/data_last/train_800', r'./data/DOTA/annotation/sub800_train2019.json')
    DOTA2COCO(r'./data/DOTA/data_last/val_800', r'./data/DOTA/annotation/sub800_val2019.json')
