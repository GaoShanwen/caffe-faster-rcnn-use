import dota_utils as util
import os
import cv2
import json
import shutil
import numpy as np
import math
from DOTA_devkit import polyiou
import collections

wordname_18 = ['__background__',
            'airport', 'baseball-diamond', 'basketball-court', 'bridge', 'container-crane',
            'ground-track-field', 'harbor', 'helicopter', 'helipad', 'large-vehicle',
            'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
            'storage-tank', 'swimming-pool', 'tennis-court']


def dots4ToRec8(bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3],
    return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax


def multi_mkdir(path):
    if not os.path.isdir(path):
        multi_mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)


def DOTA2COCO(srcpath, dest_dir):
    multi_mkdir(dest_dir)
    labelparent = os.path.join(srcpath, 'labelTxt')

    inst_count = 0
    image_id = 0
    categories_dict = {}
    data_dict = {}
    for category in wordname_18:
        if category == '__background__':
            continue
        categories_dict[category] = collections.OrderedDict({'T': 0, 'S': 0, 'M': 0, 'L': 0, 'total': 0})
        data_dict[category] = []
    categories_dict['all'] = collections.OrderedDict({'T': 0, 'S': 0, 'M': 0, 'L': 0, 'total': 0})
    # with open(destfile, 'w') as f_out:
    filenames = util.GetFileFromThisRootDir(labelparent)
    for file in filenames:
        image_id = image_id + 1
        objects = util.parse_dota_poly2(file)
        if not len(objects):
            continue
        basename = util.custombasename(file)
        # data_dict[basename] = []
        for obj in objects:
            inst_count = inst_count + 1
            single_obj = {}
            single_obj['filename'] = basename
            if obj['area']  <16*16:
                categories_dict[obj['name']]['T'] += 1
            elif obj['area']<32*32:
                categories_dict[obj['name']]['S'] += 1
            elif obj['area']<96*96:
                categories_dict[obj['name']]['M'] += 1
            else:
                categories_dict[obj['name']]['L'] += 1
            xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                     max(obj['poly'][0::2]), max(obj['poly'][1::2])
            single_obj['bbox'] = dots4ToRec8([xmin, ymin, xmax, ymax])
            data_dict[obj['name']].append(single_obj)
            # data_dict['annotations']
            # single_obj['id'] = inst_count

        # if image_id>=2:
        #     break
    # print(data_dict)
    if 'val' in dest_dir:
        print('val/name:'.ljust(24), 'T:'.ljust(24), 'S:'.ljust(24), 'M:'.ljust(24), 'L:'.ljust(24))
    else:
        print('train/name:'.ljust(24), 'T:'.ljust(24), 'S:'.ljust(24), 'M:'.ljust(24), 'L:'.ljust(24))
    for category in wordname_18:
        if category == '__background__':
            continue
        categories_dict[category]['total'] = categories_dict[category]['T'] + categories_dict[category]['S'] \
                                             + categories_dict[category]['M'] + categories_dict[category]['L']
        txt_file_path = os.path.join(dest_dir, category + '.txt')
        with open(txt_file_path, "w") as save_f:
            # print(category, len(data_dict[category]))
            for category_ins in data_dict[category]:
                # line = '{}'.format(category_ins['bbox']).replace(',', '')
                line = '{} {} {}'.format(category_ins['filename'], 1.0, category_ins['bbox'])
                line = line.replace('(', '').replace(',', '').replace(')', '')
                save_f.writelines(line + '\n')
        save_f.close()
        print('{}'.format(category).ljust(24), '{}'.format(100* categories_dict[category]['T']/categories_dict[category]['total']).ljust(24),
              '{}'.format(100* categories_dict[category]['S'] / categories_dict[category]['total']).ljust(24),
              '{}'.format(100* categories_dict[category]['M'] / categories_dict[category]['total']).ljust(24),
              '{}'.format(100* categories_dict[category]['L'] / categories_dict[category]['total']).ljust(24),
              '{}'.format(categories_dict[category]['total']))
            # print(line)
            # break
        categories_dict['all']['T'] += categories_dict[category]['T']
        categories_dict['all']['S'] += categories_dict[category]['S']
        categories_dict['all']['M'] += categories_dict[category]['M']
        categories_dict['all']['L'] += categories_dict[category]['L']
        categories_dict['all']['total'] += categories_dict[category]['total']
        # break
    print('{}'.format('all').ljust(24),
          '{}'.format(100 * categories_dict['all']['T'] / categories_dict['all']['total']).ljust(24),
          '{}'.format(100 * categories_dict['all']['S'] / categories_dict['all']['total']).ljust(24),
          '{}'.format(100 * categories_dict['all']['M'] / categories_dict['all']['total']).ljust(24),
          '{}'.format(100 * categories_dict['all']['L'] / categories_dict['all']['total']).ljust(24),
          '{}'.format(categories_dict['all']['total']))
    print('img:{}, ins:{}'.format(image_id, inst_count))
    # print(categories_dict)
    # json.dump(data_dict, f_out)


if __name__ == '__main__':
    DOTA2COCO(r'./data/DOTA/data_ori/train', r'./data/cpt_train_label')
    DOTA2COCO(r'./data/DOTA/data_ori/val', r'./data/cpt_val_label')
    # DOTA2COCO(r'./data/DOTA/data_last/train_800', r'./data/DOTA/annotation/sub800_train2019.json')
    # DOTA2COCO(r'./data/DOTA/data_last/val_800', r'./data/DOTA/annotation/sub800_val2019.json')


