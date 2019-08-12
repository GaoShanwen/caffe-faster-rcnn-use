import pickle as pkl
# import os
#
# file_dir = 'data/result'
# file_name = 'val_detection.pkl'
# file_path =os.path.join(file_dir, file_name)
#
#
# def show_boxes(boxes):
#     for category_box in boxes:
#         for instance_index in category_box:
#             if instance_index.all():
#                 print('b:', instance_index)
#                 return True
#     return False
#
#
# if __name__ =='__main__':
#     pkl_dict = pkl.load(open(file_path, 'rb'))
#     # print(type(pkl_dict))
#     # print('inf', inf[0])
#     # for key in pkl_dict.keys():
#     #     print(key)
#     txt = pkl_dict["txt_all"]
#     boxes = pkl_dict["all_boxes"]
#     segms = pkl_dict["all_segms"]
#     uvs = pkl_dict["all_uvs"]
#     keyps = pkl_dict["all_keyps"]
#     parss = pkl_dict["all_parss"]
#     cfg = pkl_dict["cfg"]
#     print('t', txt)# []
#     print('l:', len(boxes[1]))
#     show_boxes(boxes)
#
#     # print('s:', len(segms))
#     # print('u:', len(uvs))
#     # print('k:', len(keyps))
#     # print('p:', len(parss))
#     # print('c:', cfg)

import os
import json
import numpy as np


wordname_18 = ['__background__',
            'airport', 'baseball-diamond', 'basketball-court', 'bridge', 'container-crane',
            'ground-track-field', 'harbor', 'helicopter', 'helipad', 'large-vehicle',
            'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
            'storage-tank', 'swimming-pool', 'tennis-court']
mininame_3 = ['__background__', 'container-crane', 'helicopter', 'helipad']


def dots4ToRec8(poly):
    xmin, ymin, xmax, ymax = poly[0], poly[1], poly[2], poly[3],
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]


def COCO2DOTA(json_file_path, img_index_path, txt_file_dir):
    """
    :param json_file_path:
    :param img_index_path:
    :param txt_file_dir:
    :return:
    :step
        1 create a dict for image index.
        2 create txt for dota Datasets' result
    """
    img_index_dict = {}
    with open(img_index_path, 'r') as load_f:
        index_dict = json.load(load_f)["images"]
    for img_ins in index_dict:
        # print(img_ins)
        img_index_dict[img_ins["id"]] = img_ins["file_name"][:-4]#.split('_')[0]
        # break
    # print(img_index_dict)
    for category_index, category_name in enumerate(wordname_18):
        if category_name =='__background__':
            continue
        with open(json_file_path, 'r') as load_f:
            anno_list = json.load(load_f)
        # 87060 {'score': 0.16897061467170715, 'image_id': 13, 'category_id': 1,
        # 'bbox': [548.188232421875, 960.3355102539062, 440.392333984375, 63.0531005859375]}
        txt_file_path = os.path.join(txt_file_dir, category_name + '.txt')
        with open(txt_file_path, "w") as save_f:
            for anno_ins in anno_list:
                if anno_ins["category_id"] != category_index:
                    continue
                score = anno_ins["score"]
                img_name = img_index_dict[anno_ins["image_id"]]
                # bbox = list(np.array(anno_ins["bbox"]).astype(int))
                bbox = anno_ins["bbox"]
                # print(bbox)
                rbox = dots4ToRec8([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                line = '{} {} {}'.format(img_name, score, rbox)
                line = line.replace('[', '').replace(',', '').replace(']', '')
                # print(line)
                # 13 0.16897061467170715
                # 548.188232421875 960.3355102539062 988.58056640625 960.3355102539062
                # 988.58056640625 1023.3886108398438 548.188232421875 1023.3886108398438
                # print(line)
                # break
                save_f.writelines(line+'\n')
        save_f.close()
        # break
        # txt_file_path = os.path.join(txt_file_dir, category_name)



# def dots4ToRec4(poly):
#     xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
#                             max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
#                              min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
#                              max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
#     return xmin, ymin, xmax, ymax


def dots4ToRec8(poly):
    xmin, ymin, xmax, ymax = poly[0], poly[1], poly[2], poly[3]
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]


def multi_mkdir(path):
    if not os.path.isdir(path):
        multi_mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)


def test2DOTA(pred_file_dir, txt_file_dir, categroy_list):
    """
    :param json_file_path:
    :param img_index_path:
    :param txt_file_dir:
    :return:
    :step
        1 create a dict for image index.
        2 create txt for dota Datasets' result
    """
    multi_mkdir(txt_file_dir)
    img_list = os.listdir(pred_file_dir)
    for category_index, category_name in enumerate(categroy_list):
        if category_name =='__background__':
            continue

        txt_file_path = os.path.join(txt_file_dir, category_name + '.txt')
        with open(txt_file_path, "w") as save_f:
            for file_ins in img_list:
                file_ins_path = os.path.join(pred_file_dir, file_ins)
                with open(file_ins_path, "r") as read_f:
                    lines = read_f.readlines()
                splitlines = [x.strip().split(',') for x in lines]
                for splitline in splitlines:
                    if len(splitline)!=6:
                        continue
                    if int(splitline[4])!= category_index:
                        continue
                    bbox = np.array(splitline[:4]).astype(float).tolist()

                    score = float(splitline[5])

                    # print(bbox)
                    rbox = dots4ToRec8(bbox)
                    line = '{} {} {}'.format(file_ins[:-4], score, rbox)
                    line = line.replace('[', '').replace(',', '').replace(']', '')
                    # print(line)
                    # 13 0.16897061467170715
                    # 548.188232421875 960.3355102539062 988.58056640625 960.3355102539062
                    # 988.58056640625 1023.3886108398438 548.188232421875 1023.3886108398438
                    # print(line)
                    # break
                    save_f.writelines(line+'\n')
        save_f.close()
        # break
        # txt_file_path = os.path.join(txt_file_dir, category_name)


if __name__ =='__main__':
    # COCO2DOTA('data/result/val_results.json', 'data/DOTA/annotations/sub800_val2019.json', 'data/result/val_p_split')
    # '/home/user/workspace/PytorchEveryThing/results/mask_dota'
    # test2DOTA('../../PytorchEveryThing/results/dota1', 'data/result/3c_split', mininame_3)
    # test2DOTA('../../PytorchEveryThing/results/dota2', 'data/result/3c_split2', mininame_3)
    # test2DOTA('../../PytorchEveryThing/results/dota', 'data/result/test_HR50_split', wordname_18)
    test2DOTA('../../PytorchEveryThing/results/mask_dota', 'data/result/test_MHR502_split', wordname_18)
    # test2DOTA('../../PytorchEveryThing/results/mask_dota2', 'data/result/test_MHR50_split', wordname_18)