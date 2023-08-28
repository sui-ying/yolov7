# encoding=utf-8
import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse
import cv2
import os
import shutil
import random
import argparse
import sys
import numpy as np
import glob
import re

from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance


sys.path.append("../../myscripts")
from my_io import JsonHandler, YamlHandler
from figure import draw_bar
from scripts_for_image import filter_different_suffix_with_the_same_name_of_2dirs, rgb2gray_for_imgset_and_save
from tools import not_exists_path_make_dir, not_exists_path_make_dirs
from file_operate import Count_the_number_of_directory, read_txt_and_return_list, save_file_according_suffixes
from augment import contrastEnhancement, brightnessEnhancement, colorEnhancement, sharp_enhance, noise_enhance
from tools_module.basic_process_file import Process_file
from convert import xyxy_four_point_convert_xywh, xywh_convert_xxyy_four_point

class Generate_dataset_v5_to_v7(Process_file):
    def __init__(self, args):
        self.opt = args

    def _init_path(self):
        self.wdir = self.opt.wdir
        self.imgpath = os.path.join(self.wdir, 'img')  # img path
        self.xmlpath = os.path.join(self.wdir, 'xml')  # xml path
        not_exists_path_make_dirs([self.imgpath, self.xmlpath])
        self.sets = ['train', 'val', 'test']
        self.project = self.opt.project_type

        self.infopath = os.path.join(self.wdir, 'info')  # xml 2 info: [cls, x, y, w, h]
        self.all_visual_path = os.path.join(self.wdir, 'all_visualization')  # visual path
        self.error_img_path = os.path.join(os.path.join(self.wdir, "error/img"))
        self.error_xml_path = os.path.join(os.path.join(self.wdir, "error/xml"))
        self.dataset = os.path.join(self.wdir, 'dataset')
        self.train_set = os.path.join(self.dataset, "train")
        self.val_set = os.path.join(self.dataset, "val")
        self.test_set = os.path.join(self.dataset, "test")
        not_exists_path_make_dirs([self.infopath, self.all_visual_path, self.error_img_path,
                                   self.error_xml_path, self.train_set, self.val_set, self.test_set])
        if self.opt.enhance:
            self.enhance_path = os.path.join(self.wdir, "enhance")
            not_exists_path_make_dir(self.enhance_path)

    def _load_labelmap(self):
        # load labelmap
        jsonhander = JsonHandler(os.path.join(self.opt.wdir, "labelmap.json"))
        print("labelmap path: ", os.path.join(self.opt.wdir, "labelmap.json"))
        classes = jsonhander.load_json()['label']
        print(classes)
        print("nc:", len(classes))
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
        self.classes = classes

    def _xml2info(self):
        """
        xml to info:
        up_left, up_right, down_right, down_left: [label x y w h  pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y]
        """
        with tqdm(total=len(os.listdir(self.imgpath))) as p_bar:
            p_bar.set_description('xml2info')
            for file in os.listdir(self.imgpath):
                bbox_xywh = []
                name, suffix = os.path.splitext(file)
                # 1. read xml to info
                if os.path.exists(os.path.join(self.xmlpath, name + ".xml")):
                    bbox_xyxy = self._read_xml_to_lst(os.path.join(self.xmlpath, name + ".xml"))
                else:
                    shutil.move(os.path.join(self.imgpath, file), self.error_img_path)
                    print(name + ".xml not exist")
                    p_bar.update(1)
                    continue
                if len(bbox_xyxy) == 0:
                    print("The len(info)=0: ", file)
                    shutil.move(os.path.join(self.imgpath, file), self.error_img_path)
                    shutil.move(os.path.join(self.xmlpath, name + ".xml"), self.error_xml_path)
                    p_bar.update(1)
                    continue

                # dataset format: v5 to v7
                for box in bbox_xyxy:
                    cls, xmin, ymin, xmax, ymax = box
                    box += [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

                img = cv2.imread(os.path.join(self.imgpath, file))
                height, width, channel = img.shape

                # 2. img and box resize
                if self.opt.label_box:
                    # 对图像进行缩放并且进行长和宽的扭曲
                    resize_size = self.opt.resize_size
                    img = cv2.resize(img, (resize_size[0], resize_size[1]), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(self.new_img_path, file), img)
                    # INTER_NEAREST - 最邻近插值
                    # INTER_LINEAR - 双线性插值，如果最后一个参数你不指定，默认使用这种方法
                    # INTER_CUBIC - 4x4像素邻域内的双立方插值
                    # INTER_LANCZOS4 - 8x8像素邻域内的Lanczos插值
                    bbox_xyxy = self._bbox_resize_nopad(width, height, bbox_xyxy, resize_size[0],
                                                        resize_size[1])  # np padding
                else:
                    w = width
                    h = height

                # 3. write label info to text
                out_file = open(os.path.join(self.infopath, name + '.txt'), 'w')
                for i in range(len(bbox_xyxy)):
                    newx, newy, neww, newh, pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y = \
                        xyxy_four_point_convert_xywh((w, h), bbox_xyxy[i])
                    line = str(bbox_xyxy[i][0]) + " " \
                           + str(newx) + " " \
                           + str(newy) + " " \
                           + str(neww) + " " \
                           + str(newh) + " " \
                           + str(pt0_x) + " " \
                           + str(pt0_y) + " " \
                           + str(pt1_x) + " " \
                           + str(pt1_y) + " " \
                           + str(pt2_x) + " " \
                           + str(pt2_y) + " " \
                           + str(pt3_x) + " " \
                           + str(pt3_y) + "\n"
                    out_file.write(line)
                out_file.close()

                # 3. visual
                if self.opt.all_visual and self.opt.visual_cls:
                    visual_cls_path = os.path.join(os.path.join(self.wdir, "visualizaition_cls"))
                    for j in range(len(self.classes)):
                        not_exists_path_make_dir(os.path.join(visual_cls_path, self.classes[j]))

                    labelled = img
                    for i in range(len(bbox_xyxy)):
                        cls, xmin, ymin, xmax, ymax, pt0x, pt0y, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y = bbox_xyxy[i]
                        labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                 self.colors[int(cls)], 2)
                        labelled = cv2.putText(labelled, self.classes[int(cls)], (int(xmin), int(ymin) - 2),
                                               cv2.FONT_HERSHEY_PLAIN,
                                               2, self.colors[int(cls)], 1)  # font scale, thickness
                        cv2.circle(labelled, (pt0x, pt0y), 7, (0, 255, 255), -1)
                        cv2.circle(labelled, (pt1x, pt1y), 7, (0, 255, 255), -1)
                        cv2.circle(labelled, (pt2x, pt2y), 7, (0, 255, 255), -1)
                        cv2.circle(labelled, (pt3x, pt3y), 7, (0, 255, 255), -1)
                    cv2.imwrite(os.path.join(self.all_visual_path, name + '.jpg'), labelled, [int(cv2.IMWRITE_JPEG_QUALITY), 40])  # 0-100, 越小图片质量越低
                    cv2.imwrite(os.path.join(visual_cls_path, self.classes[cls], name + '.jpg'), labelled, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                else:
                    labelled = img
                    for i in range(len(bbox_xyxy)):
                        cls, xmin, ymin, xmax, ymax, pt0x, pt0y, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y = bbox_xyxy[i]
                        labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                 self.colors[int(cls)], 2)
                        labelled = cv2.putText(labelled, self.classes[int(cls)], (int(xmin), int(ymin) - 2),
                                               cv2.FONT_HERSHEY_PLAIN,
                                               2, self.colors[int(cls)], 2)  # font scale, thickness
                        cv2.circle(labelled, (pt0x, pt0y), 7, (0, 255, 255), -1)
                        cv2.circle(labelled, (pt1x, pt1y), 7, (0, 255, 255), -1)
                        cv2.circle(labelled, (pt2x, pt2y), 7, (0, 255, 255), -1)
                        cv2.circle(labelled, (pt3x, pt3y), 7, (0, 255, 255), -1)
                    cv2.imwrite(os.path.join(self.all_visual_path, name + '.jpg'), labelled, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                p_bar.update(1)

    def _generate_dataset(self):
        """
        making dataset: train, val, test = 0.9, 0.1 * 0.9, 0.01
        :return:
        """
        val_ratio = (1 - self.opt.train_ratio) * self.opt.train_ratio
        # test_ratio = 1 - self.trin_ratio - self.val_ratio

        # 制作训练集、验证集和测试集
        img_lst = os.listdir(self.imgpath)
        total_len = len(img_lst)
        train_len = int(total_len * self.opt.train_ratio)
        val_len = int(total_len * val_ratio)
        # test_len = total_len - train_len - val_len

        index_list = list(range(total_len))
        random.shuffle(index_list)

        train_index = index_list[:train_len]
        val_index = index_list[train_len:(train_len + val_len)]
        # test_index = index_list[(train_len + val_len):]

        with tqdm(total=total_len) as p_bar:
            p_bar.set_description("generate dataset")
            for i in index_list:
                filename, suffix = os.path.splitext(img_lst[i])

                if i in train_index:
                    shutil.copy(os.path.join(self.imgpath, img_lst[i]), self.train_set)
                    shutil.copy(os.path.join(self.infopath, filename + ".txt"), self.train_set)
                    p_bar.update(1)
                elif i in val_index:
                    shutil.copy(os.path.join(self.imgpath, img_lst[i]), self.val_set)
                    shutil.copy(os.path.join(self.infopath, filename + ".txt"), self.val_set)
                    p_bar.update(1)
                else:  # testset
                    shutil.copy(os.path.join(self.imgpath, img_lst[i]), self.test_set)
                    shutil.copy(os.path.join(self.infopath, filename + ".txt"), self.test_set)
                    p_bar.update(1)

    def _get_images_and_labels(self, dataset):
        """
        .jpg and .txt
        Iterate through the file names to find images and labels that have the same file name but different extensions,
        and store them in two lists, respectively
        :return:
        """
        image_list = []
        label_list = []
        # 获取所有文件的文件名（不含扩展名）
        file_names = set(os.path.splitext(f)[0] for f in os.listdir(dataset))
        # 遍历文件名，找出含有相同文件名但不同扩展名的图片和 label，并分别存储到两个列表中
        for name in file_names:
            image_path = os.path.join(dataset, f"{name}.jpg")
            label_path = os.path.join(dataset, f"{name}.txt")

            if os.path.exists(image_path) and os.path.exists(label_path):
                file_size = os.stat(label_path).st_size  # file size: xx byte
                if file_size > 0:
                    image_list.append(f"{name}.jpg")
                    label_list.append(f"{name}.txt")
        return image_list, label_list

    def _visual_dataset(self):
        visual_path = os.path.join(self.wdir, "visualization")
        not_exists_path_make_dir(visual_path)

        test_img_lst, test_label_lst = self._get_images_and_labels(self.test_set)
        for file in test_img_lst:
            filename, suffix = os.path.splitext(file)
            img = cv2.imread(os.path.join(self.test_set, file))
            height, width, _ = img.shape
            labelled = np.copy(img)

            with open(os.path.join(self.test_set, filename + ".txt"), 'r') as fn:
                lines = fn.readlines()
                for i in range(len(lines)):
                    if lines[i] == "\n":
                        continue
                    # 将字符串按照空格分割
                    label_info = re.split(r'\s+', lines[i])
                    if len(label_info) == 14 and label_info[-1] == "":
                        label_info = label_info[:13]
                    cls, xmin, ymin, xmax, ymax, pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y = \
                        xywh_convert_xxyy_four_point((width, height), label_info)
                    cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 4)
                    cv2.putText(labelled, cls, (int(xmin), int(ymin) - 2), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
                    cv2.circle(labelled, (pt0_x, pt0_y), 7, (0, 255, 255), -1)
                    cv2.circle(labelled, (pt1_x, pt1_y), 7, (0, 255, 255), -1)
                    cv2.circle(labelled, (pt2_x, pt2_y), 7, (0, 255, 255), -1)
                    cv2.circle(labelled, (pt3_x, pt3_y), 7, (0, 255, 255), -1)
            fn.close()

            cv2.imwrite(os.path.join(visual_path, file), labelled, [int(cv2.IMWRITE_JPEG_QUALITY), 40])

    def run(self):
        print("start")
        # 0. init
        self._init_path()

        # 1. split
        if len(os.listdir(self.imgpath)) == 0 and len(os.listdir(self.xmlpath)) == 0:
            save_file_according_suffixes(self.wdir)

        # 2. show label and save labelmap.json
        if not os.path.exists(os.path.join(self.wdir, "labelmap.json")):
            self._show_label_and_count_class_frequence()
            self._load_labelmap()
        else:
            self._load_labelmap()

        # 3 xml2info
        if self.opt.xml2info:
            self._xml2info()

        # 4 generate dataset
        if self.opt.gene_dataset:
            self._generate_dataset()

        # 5 visual
        if self.opt.visual:
            self._visual_dataset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--project_type', type=str, default=None, help='detection type, for xml2info')
    parser.add_argument('--wdir', type=str,
                        default="/cv/all_training_data/control/yard",
                        help="Dataset working directory")
    parser.add_argument('--xml2info', type=bool, default=True, help="Parse dataset annotation file")
    parser.add_argument('--all_visual', type=bool, default=True, help="visual for all dataset")
    parser.add_argument('--visual_cls', type=bool, default=False, help="visual for all class of dataset")
    parser.add_argument('--gene_dataset', type=bool, default=True, help="generate dataset")
    parser.add_argument('--enhance', type=bool, default=False, help="Enhance fewer data of class")
    parser.add_argument('--label_box', type=bool, default=False, help="label box transformation")
    parser.add_argument('--train_ratio', type=float, default=0.9, help="train set ratio")
    parser.add_argument('--resize_size', type=int, nargs='+', default=[1080, 1920],
                        help="resize size: width, height")
    parser.add_argument('--visual', type=bool, default=True, help="visual for train-val-test dataset")
    args = parser.parse_args()

    generate_dataset_v5_to_v7 = Generate_dataset_v5_to_v7(args)
    generate_dataset_v5_to_v7.run()