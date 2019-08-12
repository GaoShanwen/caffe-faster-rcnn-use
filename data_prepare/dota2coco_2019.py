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
                # print('1', obj['poly'])
                # poly = [[float(obj['poly'][0]), float(obj['poly'][1])]]
                # poly = [float(obj['poly'][0]), float(obj['poly'][1]), float(obj['poly'][2]),
                #                 float(obj['poly'][3]), float(obj['poly'][4]), float(obj['poly'][5]),
                #                 float(obj['poly'][6]), float(obj['poly'][7]) ]
                # print('2', poly)
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
        # target_dir = '/home/user/workspace/PytorchEveryThing/data/img'
        # for file_name in file_list:
        #     original_img_path = os.path.join(imageparent, file_name)
        #     target_img_path = os.path.join(target_dir, file_name)
        #     shutil.copyfile(original_img_path, target_img_path)



def Nrotate(box, angle):
    # print(box)
    left_topx, left_topy = box[:2]
    value1, value2 = box[2:]
    left_topx = np.array(left_topx)
    left_topy = np.array(left_topy)
    value1 = np.array(value1) #W(x2-x1)
    value2 = np.array(value2) #H(y2-y1)
    Rotate1 = []
    Rotate1.append(int(value1*math.cos(angle) + left_topx))#.tolist())
    Rotate1.append(int(left_topy - value1*math.sin(angle)))#.tolist())

    Rotate2 = []
    Rotate2.append(int(value1*math.cos(angle) + value2*math.sin(angle) + left_topx))#.tolist())
    Rotate2.append(int(value2*math.cos(angle) - value1*math.sin(angle) + left_topy))#.tolist())

    Rotate3 = []
    Rotate3.append(int(value2*math.sin(angle) + left_topx))#.tolist())
    Rotate3.append(int(value2*math.cos(angle) + left_topy))#.tolist())
    return box[0], box[1], Rotate1[0], Rotate1[1], Rotate2[0], Rotate2[1], Rotate3[0], Rotate3[1]


def dots4ToRec8(bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3],
    return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax


def BOXES_cpt(single_poly):
    xmin, ymin, xmax, ymax = min(single_poly[0::2]), min(single_poly[1::2]), \
                             max(single_poly[0::2]), max(single_poly[1::2])
    width, height = xmax - xmin, ymax - ymin
    BOX = xmin, ymin, xmax, ymax

    BOXP = np.array(single_poly).reshape(1, -1, 2)
    BOXP = np.append(BOXP, BOXP[0, :2].reshape(2, 2)).reshape(-1, 2).tolist()
    # print(BOXP)
    box_lefttop = [xmin, ymin]
    poly_1 = BOXP.pop(0)
    poly_2 = BOXP.pop(0)

    rotate_num = 0
    while len(BOXP[0]) != 0:
        rotate_num += 1
        if poly_1[0] != box_lefttop[0]:
            poly_1 = poly_2
            poly_2 = BOXP.pop(0)
        else:
            break
    # print(box_lefttop)
    # print(poly_1, poly_2)
    poly_3 = BOXP.pop(0)
    poly_1 = np.array(poly_1)
    poly_2 = np.array(poly_2)
    poly_3 = np.array(poly_3)
    vecter_1 = poly_2 - box_lefttop
    vecter_2 = poly_2 - poly_1
    Lx = np.sqrt(vecter_1.dot(vecter_1))
    Ly = np.sqrt(vecter_2.dot(vecter_2))
    if (Lx * Ly) == 0.0:
        # print(Lx, Ly)
        theta = np.pi / 2
    else:
        cos_theta = vecter_1.dot(vecter_2) / (Lx * Ly)
        # theta = np.arccos(cos_theta)
        if cos_theta > 1:
            cos_theta = 1.0
            # print(theta)
        elif cos_theta < -1:
            cos_theta = -1.0
        theta = np.arccos(cos_theta)
    poly_W = math.hypot(vecter_2[0], vecter_2[1])
    vecter_3 = poly_3 - poly_2
    poly_H = math.hypot(vecter_3[0], vecter_3[1])
    poly_1 = poly_1.tolist()  # must be list
    # print(poly_1, obj['poly'])
    # print(poly_W, poly_H, theta)
    URBOX = poly_1[0], poly_1[1], poly_W, poly_H, theta
    # print(single_poly, Nrotate(URBOX[:4], URBOX[4]))
    URBOX_iou = polyiou.iou_poly(polyiou.VectorDouble(single_poly), polyiou.VectorDouble(Nrotate(URBOX[:4], URBOX[4])))
    BBOX_iou = polyiou.iou_poly(polyiou.VectorDouble(single_poly), polyiou.VectorDouble(dots4ToRec8(BOX)))
    if BBOX_iou> URBOX_iou:
        # print('exit!')
        URBOX = xmin, ymin, width, height, 0.0
        ORBOX = URBOX
    else:
        # print(URBOX_iou)
        if not rotate_num % 2:
            ORBOX = single_poly[0], single_poly[1], poly_W, poly_H, theta + np.pi * (rotate_num / 2.0)
        else:
            ORBOX = single_poly[0], single_poly[1], poly_H, poly_W, theta + np.pi * (rotate_num / 2.0)
    return BOX, URBOX, ORBOX


def DOTA2RBOXES(srcpath, destfile):
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
        single_cat = {'id': idex, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)
    data_dict['annotations'] = []

    inst_count = 1
    image_id = 1
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
                single_obj['category_id'] = wordname_18.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                single_obj['image_id'] = image_id

                # print(obj['poly'])
                single_obj['bbox'], single_obj['urbox'], single_obj['orbox'] = BOXES_cpt(obj['poly'])

                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
                # break
            # images
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            # print('name:', imagepath)
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id += 1
            # break
        json.dump(data_dict, f_out)


class add2train_and_copyfile(object):
    def __init__(self, train_anno_file, val_anno_file, resultFile):
        # self.data_list = open(data_list, 'r').read().splitlines()
        self.load_json_path = train_anno_file
        self.add_json_path = val_anno_file
        self.save_json_path = resultFile

        self.images = []
        self.categories = []
        self.annotations = []
        self.file_list = []

        self.load_json()
        self.file_name_get()
        self.transfer_process('./data/DOTA/data_800/val_800', './data/DOTA/data_800/train_merge')
        self.save_json()

    def load_json(self):
        with open(self.load_json_path,'r') as load_f:
            load_dict = json.load(load_f)

        self.images = load_dict["images"]
        self.categories = load_dict["categories"]
        self.annotations = load_dict["annotations"]
        print(len(self.annotations))
        print(len(self.images))

    def file_name_get(self):
        need_list = ['container-crane', 'helicopter', 'helipad', 'large-vehicle', 'plane', 'soccer-ball-field']
        with open(self.add_json_path, 'r') as add_f:
            anno_list = json.load(add_f)
        images = anno_list["images"]
        img_index_list = {}
        for img in images:
            img_index_list[img["id"]] = img["file_name"][:-4]
        # print(img_index_list[1])
        # categories = anno_list["categories"]
        annotations = anno_list["annotations"]
        # print(images[0])
        for num, anno_ins in enumerate(annotations):
            if wordname_18[anno_ins["category_id"]] in need_list:
                file_name = img_index_list[anno_ins["image_id"]]
                # self.annotations.append(anno_ins)
                if file_name not in self.file_list:
                    self.file_list.append(file_name)
                # break
        # print(len(self.annotations))
        # print(original_img_path, target_img_path)
        # print(original_anno_path, target_anno_path)

    def transfer_process(self, original_dir, target_dir):
        inst_count = len(self.annotations)+1
        image_id = len(self.images)+1
        for file_name in self.file_list:
            original_img_path = os.path.join(original_dir, 'images', file_name+'.png')
            target_img_path = os.path.join(target_dir, 'images', file_name+'.png')
            shutil.copyfile(original_img_path, target_img_path)
            original_anno_path = os.path.join(original_dir, 'labelTxt', file_name+'.txt')
            target_anno_path = os.path.join(target_dir, 'labelTxt', file_name+'.txt')
            shutil.copyfile(original_anno_path, target_anno_path)

            objects = util.parse_dota_poly2(target_anno_path)
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
                single_obj['id'] = inst_count
                self.annotations.append(single_obj)
                inst_count = inst_count + 1
            # images
            # basename = util.custombasename(file)
            # # image_id = int(basename[1:])
            #
            # imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(target_img_path)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = file_name + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            self.images.append(single_image)
            image_id = image_id + 1

    def save_json(self):
        data_coco = {'images': self.images, 'categories': self.categories, 'annotations': self.annotations}
        json.dump(data_coco, open(self.save_json_path, 'w'))


class renameCocoJsonCategories(object):
    def __init__(self, annoFile, resultFile):
        # self.data_list = open(data_list, 'r').read().splitlines()
        self.load_json_path = annoFile
        self.save_json_path = resultFile

        self.images = []
        self.categories = []
        self.annotations = []

        # self.label_map = {}
        # for i in range(len(_classes)):
        #     self.label_map[_classes[i]] = i
        #
        # self.annID = 1

        self.load_json()
        self.transfer_process()
        self.save_json()

    def load_json(self):
        with open(self.load_json_path,'r') as load_f:
            load_dict = json.load(load_f)

        self.images = load_dict["images"]
        self.categories = load_dict["categories"]
        self.annotations = load_dict["annotations"]

    def transfer_process(self):

        # self.categories.pop('__background__')
        if self.categories[0]["name"] == '__background__':
            self.categories.remove(self.categories[0])
        for num, cate_ins in enumerate(self.categories):
            cate_ins["id"] -= 1
        for num, anno_ins in enumerate(self.annotations):
            # print(anno_ins)
            anno_ins['category_id'] -= 1

            # im["file_name"] = im["file_name"][15:]
            # im["coco_url"] = im["coco_url"][:40] + im["file_name"]
            # print(im)
            # break

    def save_json(self):
        data_coco = {'images': self.images, 'categories': self.categories, 'annotations': self.annotations}
        json.dump(data_coco, open(self.save_json_path, 'w'))


if __name__ == '__main__':
    # DOTA2COCO(r'./data/train', r'./data/DOTA_train.json')
    # DOTA2COCO(r'./data/DOTA/data_1200/train_1200', r'./data/DOTA/annotations/sub1200_train2019.json')
    # DOTA2COCO(r'./data/DOTA/data_1200/val_1200', r'./data/DOTA/annotations/sub1200_val2019.json')
    DOTA2COCO(r'./data/DOTA/data_last/train_800', r'./data/DOTA/annotation/sub800_train2019.json')
    DOTA2COCO(r'./data/DOTA/data_last/val_800', r'./data/DOTA/annotation/sub800_val2019.json')

    # DOTA2RBOXES(r'./data/DOTA/val_800', r'./data/DOTA/annotations/sub800r_val2019.json')
    # DOTA2RBOXES(r'./data/DOTA/train_merge', r'./data/DOTA/annotations/sub800mr_train2019.json')
    # renameCocoJsonCategories(r'./data/annotations/sub800_train2019.json', r'./data/annotations/sub800_train.json')
    # add2train_and_copyfile(r'./data/DOTA/annotations/sub800_train2019.json', \
    #                         r'./data/DOTA/annotations/sub800_val2019.json', \
    #                        r'./data/DOTA/annotations/sub800m_train2019.json')
    # pass

