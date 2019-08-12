# import math
import numpy as np
# import torchvision
import cv2
import os
import json
import shutil
from data_prepare.dota_utils import wordname_18
import math
import xml.dom.minidom

# import pycocotools.mask as mask_util
# from collections import defaultdict
#
# from pet.utils.timer import Timer
# import pet.utils.colormap as colormap_utils
#
# from pet.pose.core.config import cfg
# from pet.pose.utils.keypoints import get_keypoints, get_max_preds

_GRAY = [218, 227, 218]
_GREEN = [18, 127, 15]
_WHITE = [255, 255, 255]
_RED = [0, 0, 255]

# img_dir = 'data/train_2019/images'
# anno_dir = 'data/annotations'
# anno_name = 'sub1024_train2019.json'

mininame_3 = ['__background__', 'container-crane', 'helicopter', 'helipad']


def get_class_string(class_index, score, dataset):
    # class_text = dataset.classes[class_index] if dataset is not None else \
    #     'id{:d}'.format(class_index)
    class_text = ''#''id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_bbox(img, bbox, bbox_color):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), bbox_color, thickness=2)

    return img


def vis_class(img, pos, class_str, bg_color):
    """Visualizes the class."""
    font_color = _WHITE
    font_scale = 0.7

    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, bg_color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, font_color, lineType=cv2.LINE_AA)

    return img


def vis_mask(img, mask, bbox_color, show_parss=False):
    """Visualizes a single binary mask."""
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    border_color = cfg.VIS.SHOW_SEGMS.BORDER_COLOR
    border_thick = cfg.VIS.SHOW_SEGMS.BORDER_THICK

    mask_color = bbox_color if cfg.VIS.SHOW_SEGMS.MASK_COLOR_FOLLOW_BOX else _WHITE
    mask_color = np.asarray(mask_color)
    mask_alpha = cfg.VIS.SHOW_SEGMS.MASK_ALPHA

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if cfg.VIS.SHOW_SEGMS.SHOW_BORDER:
        cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)

    if cfg.VIS.SHOW_SEGMS.SHOW_MASK and not show_parss:
        img[idx[0], idx[1], :] *= 1.0 - mask_alpha
        img[idx[0], idx[1], :] += mask_alpha * mask_color

    return img.astype(np.uint8)


def json_vis(anno_path, img_dir):
    with open(anno_path, 'r') as load_f:
        load_dict = json.load(load_f)
    # for key in load_dict.keys():
    #     print(key)
    # # annotations, info, categories, images
    # print('anno:', load_dict["annotations"][0])
    # # {'segmentation': [[320, 218, 701, 225, 668, 608, 311, 599]],
    # # 'bbox': [311, 218, 390, 390], 'area': 141126.0,
    # # 'image_id': 4, 'category_id': 2, 'id': 1, 'iscrowd': 0}
    #
    # print('img:', load_dict["images"][0])
    # # {'width': 1024, 'file_name': 'P3536__1__13860___11088.png', 'height': 1024, 'id': 1}

    anno_dict = load_dict["annotations"]
    img_index_dict = load_dict["images"]
    ins_color = _GREEN
    for img_ins in img_index_dict:
        index = img_ins["id"]
        if index ==1:
            continue
        img_name = img_ins["file_name"]
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        annos = [anno for anno in anno_dict if anno["image_id"]==index]
        # print('num:', len(annos))
        if len(annos)==0:
            continue
        for instance in annos:
            bbox = instance["bbox"]
            img = vis_bbox(img, bbox, ins_color)

            class_index = instance["category_id"]
            print('class_index:', class_index)
            txt = '{}'.format(mininame_3[class_index])  # .lstrip('0')wordname_18
            img = vis_class(img, (bbox[0], bbox[1] - 2), txt, ins_color)
        cv2.imwrite('./{}.png'.format(img_name[:-4]), img)
            # print('img:', img_path)
        # break


def dots4ToRec8(poly):
    xmin, ymin, xmax, ymax = poly[0], poly[1], poly[2], poly[3],
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]


def pred_vis(img_index_path, json_file_path, image_dir):
    img_index_dict = {}
    with open(img_index_path, 'r') as load_f:
        index_dict = json.load(load_f)["images"]
    for img_ins in index_dict:
        # print(img_ins)
        img_index_dict[img_ins["id"]] = img_ins["file_name"][:-4]

    file_list = os.listdir(image_dir)
    file_list.sort()
    file_name = file_list[8]
    # print(file_name)

    img_path = os.path.join(image_dir, file_name)
    img = cv2.imread(img_path)

    with open(json_file_path, 'r') as load_f:
        anno_list = json.load(load_f)
    for anno_ins in anno_list:
        img_name = img_index_dict[anno_ins["image_id"]]
        if img_name != file_name[:-4]:
            continue
        # print(anno_ins["category_id"])
        category = wordname_18[anno_ins["category_id"]]
        score = anno_ins["score"]
        bbox = anno_ins["bbox"]

        # rbox = dots4ToRec8([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        print (bbox)
        # img = vis_bbox(img, [rbox[0], rbox[1], rbox[4]-rbox[0], rbox[5]-rbox[1]], _GREEN)
        img = vis_bbox(img, bbox, _GREEN)
        # break
    print(file_name[:-4])
    cv2.imwrite('./{}.png'.format(file_name[:-4]), img)


def dots4result_vis(pred_dir, img_dir):
    objects = []

    for categroy in wordname_18:
        if categroy =='__background__':
            continue
        file_path = os.path.join(pred_dir, categroy+'.txt')
        with open(file_path, 'r') as f:
            lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            for splitline in splitlines:
                if len(splitline) < 9:
                    # print(splitline)
                    continue
                object_struct = {}
                object_struct['name'] = splitline[0]
                object_struct['bbox'] = [int(float(splitline[2])),
                                             int(float(splitline[3])),
                                             int(float(splitline[6])),
                                             int(float(splitline[7]))]
                objects.append(object_struct)


    file_list = os.listdir(img_dir)
    file_list.sort()
    file_name = 'P2916.png'#file_list[8]
    print(file_name)


    img_path =os.path.join(img_dir, file_name)
    img = cv2.imread(img_path)
    for obj in objects:
        # print(obj["name"])
        if obj["name"]!= file_name[:-4]:
            continue
        bbox = obj["bbox"]
        print([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
        img = vis_bbox(img, [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], _GREEN)
        # img = vis_bbox(img, [bbox[0], bbox[1], bbox[2]-bbox[1], bbox[3]-bbox[0]], _GREEN)
        # break
    cv2.imwrite('./{}.png'.format(file_name[:-4]), img)


def result_vis(pred_dir, img_dir, cat_list):
    objects = []

    for categroy in cat_list:
        if categroy =='__background__':
            continue
        if categroy !='helipad':#container-crane
            continue
        file_path = os.path.join(pred_dir, categroy+'.txt')
        with open(file_path, 'r') as f:
            lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            for splitline in splitlines:
                if len(splitline) < 9:
                    # print(splitline)
                    continue
                object_struct = {}
                object_struct['name'] = splitline[0]#categroy#
                object_struct['bbox'] = np.array(splitline[2:]).astype(float)#.tolist()
                objects.append(object_struct)
        print(len(objects))

    file_list = os.listdir(img_dir)
    file_list.sort()
    file_name = 'P7015.png'#file_list[1]#'P2181.png'#
    print(file_name)
    # for index, file_name in enumerate(file_list):
    #     print(file_name)

    img_path =os.path.join(img_dir, file_name)
    img = cv2.imread(img_path)
    for obj in objects:
        if obj["name"]!= file_name[:-4]:
            continue
        # print(obj["name"])
        bbox = obj["bbox"]
        box = np.reshape(bbox, (1, -1, 2)).astype(np.int32)
        # print(box)
        cv2.polylines(img, box.astype(np.int32), 1, _GREEN, 2)
        # cv2.circle(img, tuple(bbox[:2].astype(np.int32)), 1, _RED, 12)
        cv2.circle(img, (int(bbox[0])-20, int(bbox[1])-20), 1, _RED, 12)
        cv2.circle(img, tuple(bbox[2:4].astype(np.int32)), 1, _WHITE, 2)
    # cv2.imwrite('./{}_0.png'.format(file_name[:-4]), img[:4000, :4000, :])
    # cv2.imwrite('./{}_1.png'.format(file_name[:-4]), img[4000:8000, :4000, :])
    cv2.imwrite('./{}.png'.format(file_name[:-4]), img)
    # cv2.imwrite('./{}.png'.format(file_name[:-4]), img[:4000, :4000, :])
        # if index > 30:
        #     break


def txt_vis(pred_dir, img_dir):
    objects = []

    label_list = os.listdir(pred_dir)
    print(len(label_list))
    for label_txt in label_list:
        file_path = os.path.join(pred_dir, label_txt)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            splitlines = [x.strip().split(',') for x in lines]
            for splitline in splitlines:
                if len(splitline) < 3:
                    # print(splitline)
                    continue
                object_struct = {}
                object_struct['name'] = label_txt[:-4]#categroy#
                object_struct['categroy'] = int(splitline[4])#categroy#
                object_struct['bbox'] = np.array(splitline[:4]).astype(float)#.tolist()
                objects.append(object_struct)

    file_list = os.listdir(img_dir)
    file_list.sort()
    print(len(file_list))
    # file_name = file_list[0]#'P2181.png'#
    for index, file_name in enumerate(file_list):
        print(file_name)

        img_path =os.path.join(img_dir, file_name)
        img = cv2.imread(img_path)
        for obj in objects:
            # print(obj["name"])
            # print(obj["name"], file_name[:-4])
            if obj["name"]!= file_name[:-4]:
                continue
            bbox = obj["bbox"]
            txt_ = mininame_3[obj["categroy"]]
            vis_bbox(img, [bbox[0], bbox[1], bbox[2] - bbox[1], bbox[3] - bbox[0]], _GREEN)
            img = vis_class(img, (bbox[0], bbox[1] - 2), txt_, _GREEN)
            # box = np.reshape(bbox, (1, -1, 2)).astype(np.int32)
            # print(box)
            # cv2.polylines(img, box.astype(np.int32), 1, _GREEN, 2)
            # cv2.circle(img, tuple(bbox[:2].astype(np.int32)), 1, _RED, 2)
            # cv2.circle(img, tuple(bbox[2:4].astype(np.int32)), 1, _WHITE, 2)
        # cv2.imwrite('./{}_0.png'.format(file_name[:-4]), img[:4000, :4000, :])
        # cv2.imwrite('./{}_1.png'.format(file_name[:-4]), img[4000:8000, :4000, :])
        # cv2.imwrite('./{}.png'.format(file_name[:-4]), img[:4000, :4000, :])
        cv2.imwrite('./{}.png'.format(file_name[:-4]), img)
        # if index > 10:
        #     break


def ori_label_vis(label_dir, img_dir):

    file_list = os.listdir(img_dir)
    file_list.sort()
    file_name = file_list[8][:-4]
    file_name = 'P0696__1__739___600'
    print(file_name)

    objects = []
    file_path = os.path.join(label_dir, file_name+'.txt')
    print(file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            if len(splitline) < 9:
                # print(splitline)
                continue
            object_struct = {}
            object_struct['category'] = splitline[8]
            object_struct['poly'] = np.array(splitline[:8]).astype(float)#.tolist()
            objects.append(object_struct)


    img_path =os.path.join(img_dir, file_name+'.png')
    img = cv2.imread(img_path)
    for obj in objects:
        # print(obj["name"])
        # if obj["category"]!= file_name[:-4]:
        #     continue
        bbox = obj["poly"]
        box = np.reshape(bbox, (1, -1, 2)).astype(np.int32)
        # print(box)
        cv2.polylines(img, box.astype(np.int32), 1, _GREEN, 3)
        cv2.circle(img, tuple(bbox[:2].astype(np.int32)), 1, _RED, 3)
    cv2.imwrite('./{}.png'.format(file_name), img)


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
    return np.array([[box[:2], Rotate1, Rotate2, Rotate3]], dtype=np.int)


def rbox_label_vis(anno_path, img_dir, save_dir):
    with open(anno_path, 'r') as load_f:
        load_dict = json.load(load_f)

    anno_dict = load_dict["annotations"]
    img_index_dict = load_dict["images"]
    ins_color = _GREEN
    for img_ins in img_index_dict:
        index = img_ins["id"]
        if index ==1:
            continue
        img_name = img_ins["file_name"]
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        annos = [anno for anno in anno_dict if anno["image_id"]==index]
        # print('num:', len(annos))
        # print('name:', img_name)
        if len(annos)==0:
            continue
        for instance in annos:
            bbox = instance["urbox"][:4]
            theta = instance["urbox"][4]

            # img = vis_bbox(img, bbox, ins_color)
            # print(instance["urbox"])
            poly = Nrotate(bbox, theta)
            # print(poly)
            # class_index = instance["category_id"]-1
            # txt = '{}'.format(wordname_18[class_index])  # .lstrip('0')
        #     img = vis_class(img, (bbox[0], bbox[1] - 2), txt, ins_color)
            cv2.polylines(img, poly.astype(np.int32), 1, ins_color, 3)
            cv2.circle(img, tuple(poly[0, 0].astype(np.int32)), 1, _RED, 3)
            # break
        cv2.imwrite('./{}/{}.png'.format(save_dir, img_name[:-4]), img)
        break

def vis_xml():
    dom = xml.dom.minidom.parse('P0000.xml')
    root = dom.documentElement
    img = cv2.imread('P0000.png')
    itemlist = root.getElementsByTagName('bndbox')
    for item in itemlist:
        x1 = int(item.getElementsByTagName("x1")[0].firstChild.data)
        y1 = int(item.getElementsByTagName("y1")[0].firstChild.data)
        x2 = int(item.getElementsByTagName("x2")[0].firstChild.data)
        y2 = int(item.getElementsByTagName("y2")[0].firstChild.data)
        x3 = int(item.getElementsByTagName("x3")[0].firstChild.data)
        y3 = int(item.getElementsByTagName("y3")[0].firstChild.data)
        x4 = int(item.getElementsByTagName("x4")[0].firstChild.data)
        y4 = int(item.getElementsByTagName("y4")[0].firstChild.data)
        pts_list = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        pts_array = np.array(pts_list, np.int32)
        pts = pts_array.reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (255, 0, 0))
    cv2.imwrite('P0000_vis.png', img)

if __name__ == '__main__':
    # img_dir = 'data/test_2019/images'
    # anno_dir = 'data/annotations'
    # anno_name = 'sub1024_test2019.json'
    # anno_path = os.path.join(anno_dir, anno_name)
    # json_vis(anno_path)

    # imgname_index = 'data/annotations/sub1024_val2019.json'
    # pred_path = 'data/result/val_results.json'
    # img_dir = 'data/val_2019/images'
    # pred_vis(imgname_index, pred_path, img_dir)

    # pred_dir = 'data/result/test_p'
    # img_dir = 'data/DOTA/test/images'
    # result_vis(pred_dir, img_dir)

    # pred_dir = 'data/result/3c_split'
    # img_dir = 'data/DOTA/data_ori/val/images'
    # result_vis(pred_dir, img_dir, mininame_3)

    pred_dir = 'data/result/test_800'
    img_dir = 'data/DOTA/data_ori/test/images'
    result_vis(pred_dir, img_dir, mininame_3)
    # pred_dir = 'data/result/test_p'
    # img_dir = 'data/DOTA/test/images'
    # pred_dir = 'data/result/val_p'
    # img_dir = 'data/DOTA/val/images'

    # pred_dir = '../../PytorchEveryThing/results/dota1'
    # img_dir = 'data/DOTA/minival_800/images'
    # txt_vis(pred_dir, img_dir)

    # label_dir = 'data/DOTA/val_800/labelTxt'
    # img_dir = 'data/DOTA/val_800/images'
    # ori_label_vis(label_dir, img_dir)

    # img_dir = 'data/DOTA/val_800/images'
    # # label_path = 'data/DOTA/annotations/sub800r_val2019.json'
    # label_path = 'data/DOTA/annotations/sub800_3c_val2019.json'
    # save_dir = 'last_vis'
    # rbox_label_vis(label_path, img_dir, save_dir)

    # img_dir = 'data/DOTA/val_800/images'
    # anno_path = 'data/DOTA/annotations/sub800_3c_val2019.json'
    # json_vis(anno_path, img_dir)