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

mininame_3 = ['__background__', 'container-crane', 'helicopter', 'helipad']

def minicategory2COCO(srcpath, destfile):
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
    for idex, name in enumerate(mininame_3):
        if name =='__background__':
            continue
        single_cat = {'id': idex , 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            # annotations
            objects = util.parse_dota_poly2(file)
            if not len(objects):
                continue
            box_num = 0
            for obj in objects:
                if obj['name'] not in mininame_3:
                    continue
                box_num += 1
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = mininame_3.index(obj['name'])
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
            if not box_num:
                continue
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
            # print(data_dict)
            # break
        json.dump(data_dict, f_out)


def mini_image_copy(anno_path, original_dir, target_dir):
    with open(anno_path, 'r') as load_f:
        load_dict = json.load(load_f)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, 'images'))
        os.makedirs(os.path.join(target_dir, 'labelTxt'))
    img_index_dict = load_dict["images"]
    for img_ins in img_index_dict:
        file_name = img_ins["file_name"][:-4]
        original_img_path = os.path.join(original_dir, 'images', file_name+'.png')
        target_img_path = os.path.join(target_dir, 'images', file_name+'.png')
        shutil.copyfile(original_img_path, target_img_path)
        original_anno_path = os.path.join(original_dir, 'labelTxt', file_name+'.txt')
        target_anno_path = os.path.join(target_dir, 'labelTxt', file_name+'.txt')
        shutil.copyfile(original_anno_path, target_anno_path)


if __name__ == '__main__':
    pass
    # minicategory2COCO(r'./data/DOTA/data_800/train_merge', r'./data/DOTA/annotations/sub800_3c_train2019.json')
    # minicategory2COCO(r'./data/DOTA/data_800/val_800', r'./data/DOTA/annotations/sub800_3c_val2019.json')
    #
    # img_dir = 'data/DOTA/data_800/train_merge'
    # target_dir = 'data/DOTA/data_800/minitrain_800'
    # anno_path = 'data/DOTA/annotations/sub800_3c_train2019.json'
    # mini_image_copy(anno_path, img_dir, target_dir)
    #
    # img_dir = 'data/DOTA/data_800/val_800'
    # target_dir = 'data/DOTA/data_800/minival_800'
    # anno_path = 'data/DOTA/annotations/sub800_3c_val2019.json'
    # mini_image_copy(anno_path, img_dir, target_dir)