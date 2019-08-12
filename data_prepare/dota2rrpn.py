import numpy as np
import os
import cv2
from dota_utils import wordname_18
# from xml.dom.minidom import parse
# import xml.dom.minidom
# from PIL import Image
import pickle
import json


def get_dota800_pkl2019(img_dir, label_dir, save_path):
    img_file_type = "png"
    label_file_type = "txt"

    cls_list = {'__background__': 0 ,
            'airport': 1 , 'baseball-diamond': 2 , 'basketball-court': 3 , 'bridge': 4 , 'container-crane': 5 ,
            'ground-track-field': 6 , 'harbor': 7 , 'helicopter': 8 , 'helipad': 9 , 'large-vehicle': 10 ,
            'plane': 11 , 'roundabout': 12 , 'ship': 13 , 'small-vehicle': 14 , 'soccer-ball-field': 15 ,
            'storage-tank': 16 , 'swimming-pool': 17 , 'tennis-court': 18 }
    file_list = os.listdir(img_dir)

    img_list = []

    for file_name in file_list:
        split = file_name.split(".")
        if split[len(split) - 1] == img_file_type:
            img_list.append(file_name)
    img_list.sort()
    gt_list = []
    for img_ind in range(len(img_list)):
        split = img_list[img_ind].split(".")
        gt_list.append(split[0] + "." + label_file_type)

    im_infos = []
    for index in range(len(img_list)):
        img_name = os.path.join(img_dir, img_list[index])
        gt_name = os.path.join(label_dir, gt_list[index])
        boxes = []

        # print gt_name
        gt_obj = open(gt_name, 'r')
        gt_txt = gt_obj.read()

        gt_split = gt_txt.split('\n')
        # print('len(gt_txt:', len(gt_txt))
        # print('len(gt_split:', len(gt_split))
        if not len(gt_split)-1:
            # print(gt_name)
            continue

        img = cv2.imread(img_name)

        for gt_line in gt_split:
            gt_ind = gt_line.split(' ')
            if len(gt_ind) > 3:
                # condinate_list = gt_ind[2].split(' ')
                pt1 = (int(float(gt_ind[0])), int(float(gt_ind[1])))
                pt2 = (int(float(gt_ind[2])), int(float(gt_ind[3])))
                pt3 = (int(float(gt_ind[4])), int(float(gt_ind[5])))
                pt4 = (int(float(gt_ind[6])), int(float(gt_ind[7])))

                edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                angle = 0

                if edge1 > edge2:
                    width = edge1
                    height = edge2
                    if pt1[0] - pt2[0] != 0:
                        angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                elif edge2 >= edge1:
                    width = edge2
                    height = edge1
                    # print pt2[0], pt3[0]
                    if pt2[0] - pt3[0] != 0:
                        angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                if angle < -45.0:
                    angle = angle + 180

                x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                boxes.append([x_ctr, y_ctr, height, width, angle, gt_ind[8]])

        cls_num = len(cls_list.keys())

        gt_boxes = []  # np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = []  # np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = []  # np.zeros((len_of_bboxes, cls_num), dtype=np.float32) #text or non-text
        seg_areas = []  # np.zeros((len_of_bboxes), dtype=np.float32)

        # print ("boxes_size:", len(boxes))
        for idx in range(len(boxes)):
            if not boxes[idx][5] in cls_list:
                print (boxes[idx][5] + " not in list")
                continue
            gt_classes.append(cls_list[boxes[idx][5]])  # cls_text
            overlap = np.zeros((cls_num))
            overlap[cls_list[boxes[idx][5]]] = 1.0  # prob
            overlaps.append(overlap)
            seg_areas.append((boxes[idx][2]) * (boxes[idx][3]))
            gt_boxes.append([boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]])

        gt_classes = np.array(gt_classes)
        overlaps = np.array(overlaps)
        seg_areas = np.array(seg_areas)
        gt_boxes = np.array(gt_boxes)

        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        # print(gt_name, 'box shape:', gt_boxes.shape)

        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': img_name,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': img.shape[0],
            'width': img.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        # print('bbox', im_info["boxes"].shape)
        im_infos.append(im_info)
        # if index>=1500:
        #     break
        # print('index:{}'.fromat(index), len(img_list))
    f_save_pkl = open(save_path, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    return im_infos


def PKL2JSON(pkl_file_path, json_file_path):
    pkl_file = open(pkl_file_path, 'rb')
    img_infos = pickle.load(pkl_file)
    pkl_file.close()

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
    with open(json_file_path, 'w') as f_out:
        print(len(img_infos))
        for img_info in img_infos:
            single_image = {}
            single_image["width"] = img_info["width"]
            single_image["height"] = img_info["height"]
            single_image["file_name"] = img_info["image"].split('/')[-1]
            single_image["id"] = image_id
            data_dict['images'].append(single_image)
            # print('img:',single_image)
            objs_boxes, objs_areas, objs_classes = img_info["boxes"][:].tolist(),\
                                                   img_info["seg_areas"][:].tolist(),\
                                                   img_info["gt_classes"][:].tolist()
            for single_box, single_area, single_class in zip(objs_boxes, objs_areas, objs_classes):
                single_obj = {}
                single_obj['area'] = single_area
                single_obj['category_id'] = single_class#wordname_18.index(obj['name'])
                single_obj['iscrowd'] = 0
                single_obj['image_id'] = image_id
                single_obj['bbox'] = single_box
                single_obj['id'] = inst_count

                data_dict['annotations'].append(single_obj)
                inst_count += 1

            image_id += 1
            # break
        json.dump(data_dict, f_out)


def vis_image(image_path, boxes):
    img = cv2.imread(image_path)
    #cv2.namedWindow("image")
    #cv2.setMouseCallback("image", trigger)
    for idx in range(len(boxes)):
        cx, cy, h, w, angle = boxes[idx]
        lt = [cx - w/2, cy - h/2,1]
        rt = [cx + w/2, cy - h/2,1]
        lb = [cx - w/2, cy + h/2,1]
        rb = [cx + w/2, cy + h/2,1]

        pts = []

        #angle = angle * 0.45

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        #if angle != 0:
        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        #else :
        #	cos_cita = 1
        #	sin_cita = 0

        M0 = np.array([[1,0,0],[0,1,0],[-cx,-cy,1]])
        M1 = np.array([[cos_cita, sin_cita,0], [-sin_cita, cos_cita,0],[0,0,1]])
        M2 = np.array([[1,0,0],[0,1,0],[cx,cy,1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)


        #print im
        #print im.shape
        #			im = im.transpose(2,0,1)

        cv2.line(img, (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (0, 0, 255),3)
        cv2.line(img, (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (0, 0, 255),3)

    # img = cv2.resize(img, (1024, 768))
    # cv2.imshow("image", img)
    cv2.imwrite('./last_vis/{}.png'.format(image_path.split('/')[-1][:-4]), img)


def change_1809(img_dir):
    file_list = [filename for filename in os.listdir(img_dir) if 'P1809' in filename]
    print(len(file_list))
    for filename in file_list:
        file_path = os.path.join(img_dir, filename)
        with open(file_path, 'r') as read_f:
            lines = read_f.readlines()
            txt_write_lines = []
            for line in lines:
                splitline = line.strip().split(' ')
                if len(splitline)<9 or splitline[8] != 'helicopter':
                    # print(splitline)
                    txt_write_lines.append(line)
                    continue
                line = line.replace('helicopter', 'helipad')
                txt_write_lines.append(line)
                # print(line)

        # print(txt_write_lines)
        with open(file_path, 'w') as write_f:
            for txt_write_line in txt_write_lines:
                write_f.writelines(txt_write_line)


if __name__ == '__main__':
    # pass
    img_dir = "data/DOTA/data_800/val_800/images"
    label_dir = "data/DOTA/data_800/val_800/labelTxt"
    save_path = 'data/DOTA/annotations/sub800r_val2019.pkl'
    img_infos = get_dota800_pkl2019(img_dir, label_dir, save_path)

    # img_dir = "data/DOTA/data_800/train_merge/images"
    # label_dir = "data/DOTA/data_800/train_merge/labelTxt"
    # save_path = 'data/DOTA/annotations/sub800mr_train2019.pkl'
    # get_dota800_pkl2019(img_dir, label_dir, save_path)

    # PKL2JSON('data/DOTA/annotations/sub800r_val2019.pkl', 'data/DOTA/annotations/sub800r_val2019.json')
    # PKL2JSON('data/DOTA/annotations/sub800mr_train2019.pkl', 'data/DOTA/annotations/sub800mr_train2019.json')

    # change_1809('./data/DOTA/data_800/val_800/labelTxt')
    # change_1809('./data/DOTA/data_800/train_merge/labelTxt')
    # change_1809('./data/DOTA/data_ori/val/labelTxt')
