# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cgi import test
import cv2
import json
import os
import os.path as osp
import shutil
import numpy as np
import PIL.ImageDraw
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from utils import path_normalization, MyEncoder, is_pic, get_encoding


class X2COCO(object):
    def __init__(self):
        self.images_list = []
        self.categories_list = []
        self.annotations_list = []

    def generate_categories_field(self, label, labels_list):
        category = {}
        category["supercategory"] = "component"
        category["id"] = len(labels_list) + 1
        category["name"] = label
        return category

    def generate_rectangle_anns_field(self, points, label, image_id, object_id,
                                      label_to_num):
        annotation = {}
        seg_points = np.asarray(points).copy()
        seg_points[1, :] = np.asarray(points)[2, :]
        seg_points[2, :] = np.asarray(points)[1, :]
        annotation["segmentation"] = [list(seg_points.flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0],
                points[1][1] - points[0][1]
            ]))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    def create_json_list(self, new_image_dir_detail, json_dir, train, val, test):
        json_list_path = glob.glob("%s/*.json"%json_dir)
        train_path, val_path = train_test_split(json_list_path, train_size=train)
        val_path, test_path = train_test_split(val_path, train_size=val/(test+val))

        json_list = [train_path, val_path, test_path]

        for i in range(3):
            for img_name in json_list[i]:
                shutil.copy(
                    img_name.replace("json", "png"),
                    new_image_dir_detail[i])

    def convert(self, image_dir, json_dir, dataset_save_dir, train, val, test):
        """转换。
        Args:
            image_dir (str): 图像文件存放的路径。
            json_dir (str): 与每张图像对应的json文件的存放路径。
            dataset_save_dir (str): 转换后数据集存放路径。
        """
        assert osp.exists(image_dir), "The image folder does not exist!"
        assert osp.exists(json_dir), "The json folder does not exist!"
        if not osp.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)
        # Convert the image files.
        new_image_dir = osp.join(dataset_save_dir, "images")
        if osp.exists(new_image_dir):
            raise Exception(
                "The directory {} is already exist, please remove the directory first".
                format(new_image_dir))
        os.makedirs(new_image_dir)
        os.makedirs(osp.join(dataset_save_dir, "annotations"))

        coco_category = ['train2017', 'val2017', 'test2017']
        new_image_dir_detail = []
        for i in coco_category:
            new_image_dir_detail.append(osp.join(new_image_dir, i))
            os.makedirs(osp.join(new_image_dir, i))

        self.create_json_list(new_image_dir_detail, json_dir, train, val, test)

        # Convert the json files.
        for i in range(3):
            self.images_list = []
            self.categories_list = []
            self.annotations_list = []
            self.parse_json(new_image_dir_detail[i], json_dir)
            coco_data = {}
            coco_data["images"] = self.images_list
            coco_data["categories"] = self.categories_list
            coco_data["annotations"] = self.annotations_list
            json_path = osp.join(dataset_save_dir, 'annotations', "instances_%s.json"%coco_category[i])
            f = open(json_path, "w")
            json.dump(coco_data, f, indent=4, cls=MyEncoder)
            f.close()
        # labels.txt
        with open(dataset_save_dir + '/labels.txt', 'w') as f:
            for i in self.categories_list:
                f.writelines(i['name'])

#planthopper_135
class Planthopper135_2coco(X2COCO):
    def __init__(self):
        super(Planthopper135_2coco, self).__init__()

    def create_json_list(self, new_image_dir_detail, json_dir, train, val, test):
        json_list_path = glob.glob("%s/*.json"%json_dir)
        # print(json_list_path)
        train_path, val_path = train_test_split(json_list_path, train_size=train)
        val_path, test_path = train_test_split(val_path, train_size=val/(test+val))

        json_list = [train_path, val_path, test_path]

        for i in range(3):
            for img_name in json_list[i]:
                if os.path.exists(img_name.replace("json", "jpeg")):
                    shutil.copy(
                        img_name.replace("json", "jpeg"),
                        new_image_dir_detail[i])
                elif os.path.exists(img_name.replace("json", "jpg")):
                    shutil.copy(
                        img_name.replace("json", "jpg"),
                        new_image_dir_detail[i])
                elif os.path.exists(img_name.replace("json", "png")):
                    shutil.copy(
                        img_name.replace("json", "png"),
                        new_image_dir_detail[i])
                elif os.path.exists(img_name.replace("json", "bmp")):
                    shutil.copy(
                        img_name.replace("json", "bmp"),
                        new_image_dir_detail[i])
                else:
                    print('!!!please copy the [{}] to {}'.format(img_name, new_image_dir_detail[i]))
    
    def generate_rectangle_anns_field(self, points, label, image_id, object_id,
                                      label_to_num):
        annotation = {}
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0],
                points[1][1] - points[0][1]
            ]))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    
    def generate_images_field(self, image_file, image_id):
        image = {}
        image["id"] = image_id + 1
        image["file_name"] = image_file
        return image


    def parse_json(self, img_dir, json_dir):
        image_id = -1
        object_id = -1
        labels_list = []
        label_to_num = {}
        for img_file in os.listdir(img_dir):
            print(img_file)
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                print("!!!can't find the json file: "+json_file)
                # os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                      encoding=get_encoding(json_file)) as j:
                
                # 图片基本信息
                image = cv2.imread(osp.join(img_dir, img_file))
                size = image.shape

                img_info = {
                    "id": image_id,
                    "file_name": img_file,
                    "height": size[0],
                    "width": size[1]
                }

                self.images_list.append(img_info)

                json_info = json.load(j)
                json_info = json_info['labels']
                
                for labelInfo in json_info:
                    object_id = object_id + 1
                    label = labelInfo['name']
                    if label not in labels_list:
                        self.categories_list.append( \
                            self.generate_categories_field(label, labels_list))
                        labels_list.append(label)
                        label_to_num[label] = len(labels_list)
                    
                    points = []
                    points.append([labelInfo['x1'], labelInfo['y1']])
                    points.append([labelInfo['x2'], labelInfo['y2']])
                    self.annotations_list.append(
                        self.generate_rectangle_anns_field(
                            points, label, image_id, object_id,
                            label_to_num))

    def convert(self, dataset_dir, dataset_save_dir, train, val, test):
        assert osp.exists(dataset_dir), "The json folder does not exist!"
        # if not osp.exists('origin'):
        #     os.makedirs('origin/')
        if not osp.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)

            # for dir in os.listdir(dataset_dir):
            #     dir_name = os.path.join(dataset_dir, dir)
            #     for file in os.listdir(dir_name):
            #         if file.split('.')[1] == 'txt':
            #             shutil.copy(os.path.join(dir_name, file), os.path.join('origin/', file))
            #         else:
            #             shutil.copy(os.path.join(dir_name, file), os.path.join('origin/', file.split('.')[0]+'.jpg'))
        
        # Convert the image files.
        # new_image_dir = osp.join(dataset_save_dir, "images")
        # if osp.exists(new_image_dir):
        #     raise Exception(
        #         "The directory {} is already exist, please remove the directory first".
        #         format(new_image_dir))
        # os.makedirs(new_image_dir)
        os.makedirs(osp.join(dataset_save_dir, "annotations"))
        

        coco_category = ['train2017', 'val2017', 'test2017']
        new_image_dir_detail = []
        for i in coco_category:
            new_image_dir_detail.append(osp.join(dataset_save_dir, i))
            new_path = osp.join(dataset_save_dir, i)
            if not osp.exists(new_path):
                os.makedirs(new_path)

        self.create_json_list(new_image_dir_detail, dataset_dir, train, val, test)

        # Convert the json files.
        for i in range(3):
            self.images_list = []
            self.categories_list = []
            self.annotations_list = []
            self.parse_json(new_image_dir_detail[i], dataset_dir)
            coco_data = {}
            coco_data["images"] = self.images_list
            coco_data["categories"] = self.categories_list
            coco_data["annotations"] = self.annotations_list
            json_path = osp.join(dataset_save_dir, 'annotations', "instances_%s.json"%coco_category[i])
            f = open(json_path, "w")
            json.dump(coco_data, f, indent=4, cls=MyEncoder)
            f.close()
        # labels.txt
        with open(dataset_save_dir + '/labels.txt', 'w') as f:
            for i in self.categories_list:
                f.writelines(i['name']+'\n')

# planthopper2417
class Planthopper2417_2coco(X2COCO):
    def __init__(self):
        super(Planthopper2417_2coco, self).__init__()

    def create_json_list(self, new_image_dir_detail, json_dir, train, val, test):
        json_list_path = glob.glob("%s/*.txt"%json_dir)
        # print(json_list_path)
        train_path, val_path = train_test_split(json_list_path, train_size=train)
        val_path, test_path = train_test_split(val_path, train_size=val/(test+val))

        json_list = [train_path, val_path, test_path]

        for i in range(3):
            for img_name in json_list[i]:
                shutil.copy(
                    img_name.replace("txt", "jpg"),
                    new_image_dir_detail[i])
    
    def generate_rectangle_anns_field(self, points, label, image_id, object_id,
                                      label_to_num):
        annotation = {}
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0],
                points[1][1] - points[0][1]
            ]))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    
    def generate_images_field(self, json_info, image_file, image_id):
        image = {}
        image["id"] = image_id + 1
        image["file_name"] = image_file
        return image


    def parse_json(self, img_dir, json_dir):
        image_id = -1
        object_id = -1
        labels_list = []
        label_to_num = {}
        for img_file in os.listdir(img_dir):
            print(img_file)
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".txt")
            if not osp.exists(json_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                      encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                img_info = self.generate_images_field(json_info, img_file,
                                                      image_id)
                
                image = cv2.imread(osp.join(img_dir, img_file))
                size = image.shape
                img_info["height"] = size[0]
                img_info["width"] = size[1]
                self.images_list.append(img_info)
                for shapes in json_info["regions"]:
                    object_id = object_id + 1
                    label = "planthopper"
                    if label not in labels_list:
                        self.categories_list.append( \
                            self.generate_categories_field(label, labels_list))
                        labels_list.append(label)
                        label_to_num[label] = len(labels_list)
                    
                    points = []
                    points.append([shapes['region'][0], shapes['region'][1]])
                    points.append([shapes['region'][2], shapes['region'][3]])
                    self.annotations_list.append(
                        self.generate_rectangle_anns_field(
                            points, label, image_id, object_id,
                            label_to_num))

    def convert(self, dataset_dir, dataset_save_dir, train, val, test):
        assert osp.exists(dataset_dir), "The json folder does not exist!"
        if not osp.exists('origin'):
            os.makedirs('origin/')
        if not osp.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)

            for dir in os.listdir(dataset_dir):
                dir_name = os.path.join(dataset_dir, dir)
                for file in os.listdir(dir_name):
                    if file.split('.')[1] == 'txt':
                        shutil.copy(os.path.join(dir_name, file), os.path.join('origin/', file))
                    else:
                        shutil.copy(os.path.join(dir_name, file), os.path.join('origin/', file.split('.')[0]+'.jpg'))
        
        # Convert the image files.
        new_image_dir = osp.join(dataset_save_dir, "images")
        if osp.exists(new_image_dir):
            raise Exception(
                "The directory {} is already exist, please remove the directory first".
                format(new_image_dir))
        os.makedirs(new_image_dir)
        os.makedirs(osp.join(dataset_save_dir, "annotations"))
        

        coco_category = ['train2017', 'val2017', 'test2017']
        new_image_dir_detail = []
        for i in coco_category:
            new_image_dir_detail.append(osp.join(new_image_dir, i))
            new_path = osp.join(new_image_dir, i)
            if not osp.exists(new_path):
                os.makedirs(new_path)

        self.create_json_list(new_image_dir_detail, 'origin', train, val, test)

        # Convert the json files.
        for i in range(3):
            self.images_list = []
            self.categories_list = []
            self.annotations_list = []
            self.parse_json(new_image_dir_detail[i], 'origin')
            coco_data = {}
            coco_data["images"] = self.images_list
            coco_data["categories"] = self.categories_list
            coco_data["annotations"] = self.annotations_list
            json_path = osp.join(dataset_save_dir, 'annotations', "instances_%s.json"%coco_category[i])
            f = open(json_path, "w")
            json.dump(coco_data, f, indent=4, cls=MyEncoder)
            f.close()
        # labels.txt
        with open(dataset_save_dir + '/labels.txt', 'w') as f:
            for i in self.categories_list:
                f.writelines(i['name'])

    




class LabelMe2COCO(X2COCO):
    """将使用LabelMe标注的数据集转换为COCO数据集。
    """

    def __init__(self):
        super(LabelMe2COCO, self).__init__()

    def generate_images_field(self, json_info, image_file, image_id):
        image = {}
        image["height"] = json_info["imageHeight"]
        image["width"] = json_info["imageWidth"]
        image["id"] = image_id + 1
        json_img_path = path_normalization(json_info["imagePath"])
        json_info["imagePath"] = osp.join(
            osp.split(json_img_path)[0], image_file)
        image["file_name"] = osp.split(json_info["imagePath"])[-1]
        return image

    def generate_polygon_anns_field(self, height, width, points, label,
                                    image_id, object_id, label_to_num):
        annotation = {}
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, self.get_bbox(height, width, points)))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    def get_bbox(self, height, width, points):
        polygons = points
        mask = np.zeros([height, width], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        left_top_r = np.min(rows)
        left_top_c = np.min(clos)
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
        return [
            left_top_c, left_top_r, right_bottom_c - left_top_c,
            right_bottom_r - left_top_r
        ]

    def parse_json(self, img_dir, json_dir):
        image_id = -1
        object_id = -1
        labels_list = []
        label_to_num = {}
        for img_file in os.listdir(img_dir):
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                      encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                img_info = self.generate_images_field(json_info, img_file,
                                                      image_id)
                self.images_list.append(img_info)
                for shapes in json_info["shapes"]:
                    object_id = object_id + 1
                    label = shapes["label"]
                    if label not in labels_list:
                        self.categories_list.append( \
                            self.generate_categories_field(label, labels_list))
                        labels_list.append(label)
                        label_to_num[label] = len(labels_list)
                    points = shapes["points"]
                    p_type = shapes["shape_type"]
                    if p_type == "polygon":
                        self.annotations_list.append(
                            self.generate_polygon_anns_field(
                                json_info["imageHeight"], json_info[
                                    "imageWidth"], points, label, image_id,
                                object_id, label_to_num))
                    if p_type == "rectangle":
                        points.append([points[0][0], points[1][1]])
                        points.append([points[1][0], points[0][1]])
                        self.annotations_list.append(
                            self.generate_rectangle_anns_field(
                                points, label, image_id, object_id,
                                label_to_num))

class LabelImg2COCO(X2COCO):
    """将使用LabelMe标注的数据集转换为COCO数据集。
    """

    def __init__(self):
        super(LabelImg2COCO, self).__init__()

    def get_elements(self, root, childElementName):
        elements = root.findall(childElementName)
        return elements
    
    def get_element(self, root, childElementName):
        element = root.find(childElementName)
        return element

    def create_json_list(self, new_image_dir_detail, json_dir, train, val, test):
        json_list_path = glob.glob("%s/*.xml"%json_dir)
        train_path, val_path = train_test_split(json_list_path, train_size=train)
        val_path, test_path = train_test_split(val_path, train_size=val/(test+val))

        json_list = [train_path, val_path, test_path]

        for i in range(3):
            for img_name in json_list[i]:
                shutil.copy(
                    img_name.replace("xml", "jpg"),
                    new_image_dir_detail[i])

    def generate_rectangle_anns_field(self, points, label, image_id, object_id,
                                      label_to_num):
        annotation = {}
        
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0],
                points[1][1] - points[0][1]
            ]))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation
    
    def generate_images_field(self, root, image_file, image_id):
        image = {}
        size = self.get_element(root, 'size')
        
        image["height"] = int(self.get_element(size, 'width').text)
        image["width"] = int(self.get_element(size, 'width').text)
        image["id"] = image_id + 1
        image["file_name"] = self.get_element(root, 'filename').text.split('.')[0] + '.jpg'
        return image

    def generate_polygon_anns_field(self, height, width, points, label,
                                    image_id, object_id, label_to_num):
        annotation = {}
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, self.get_bbox(height, width, points)))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    def get_bbox(self, height, width, points):
        polygons = points
        mask = np.zeros([height, width], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        left_top_r = np.min(rows)
        left_top_c = np.min(clos)
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
        return [
            left_top_c, left_top_r, right_bottom_c - left_top_c,
            right_bottom_r - left_top_r
        ]

    def parse_json(self, img_dir, json_dir):
        image_id = -1
        object_id = -1
        labels_list = []
        label_to_num = {}
        for img_file in os.listdir(img_dir):
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".xml")
            if not osp.exists(json_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1

            tree = ET.parse(json_file)
            root = tree.getroot()

            img_info = self.generate_images_field(root, img_file,
                                                    image_id)
            self.images_list.append(img_info)
            for shapes in self.get_elements(root, 'object'):
                object_id = object_id + 1
                label = self.get_element(shapes, 'name').text
                if label not in labels_list:
                    self.categories_list.append( \
                        self.generate_categories_field(label, labels_list))
                    labels_list.append(label)
                    label_to_num[label] = len(labels_list)
                # points = shapes["points"]
                # p_type = shapes["shape_type"]
                # if p_type == "polygon":
                #     self.annotations_list.append(
                #         self.generate_polygon_anns_field(
                #             json_info["imageHeight"], json_info[
                #                 "imageWidth"], points, label, image_id,
                #             object_id, label_to_num))
                # if p_type == "rectangle":
                bndbox = self.get_element(shapes, 'bndbox')
                xmin = int(self.get_element(bndbox, 'xmin').text)
                ymin = int(self.get_element(bndbox, 'ymin').text)
                xmax = int(self.get_element(bndbox, 'xmax').text)
                ymax = int(self.get_element(bndbox, 'ymax').text)
                points = []
                points.append([xmin, ymax])
                points.append([xmax, ymin])
                self.annotations_list.append(
                    self.generate_rectangle_anns_field(
                        points, label, image_id, object_id,
                        label_to_num))


class EasyData2COCO(X2COCO):
    """将使用EasyData标注的检测或分割数据集转换为COCO数据集。
    """

    def __init__(self):
        super(EasyData2COCO, self).__init__()

    def generate_images_field(self, img_path, image_id):
        image = {}
        img = cv2.imread(img_path)
        image["height"] = img.shape[0]
        image["width"] = img.shape[1]
        image["id"] = image_id + 1
        img_path = path_normalization(img_path)
        image["file_name"] = osp.split(img_path)[-1]
        return image

    def generate_polygon_anns_field(self, points, segmentation, label,
                                    image_id, object_id, label_to_num):
        annotation = {}
        annotation["segmentation"] = segmentation
        annotation["iscrowd"] = 1 if len(segmentation) > 1 else 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, [
                points[0][0], points[0][1], points[1][0] - points[0][0],
                points[1][1] - points[0][1]
            ]))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    def parse_json(self, img_dir, json_dir):
        from pycocotools.mask import decode
        image_id = -1
        object_id = -1
        labels_list = []
        label_to_num = {}
        for img_file in os.listdir(img_dir):
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                      encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                img_info = self.generate_images_field(
                    osp.join(img_dir, img_file), image_id)
                self.images_list.append(img_info)
                for shapes in json_info["labels"]:
                    object_id = object_id + 1
                    label = shapes["name"]
                    if label not in labels_list:
                        self.categories_list.append( \
                            self.generate_categories_field(label, labels_list))
                        labels_list.append(label)
                        label_to_num[label] = len(labels_list)
                    points = [[shapes["x1"], shapes["y1"]],
                              [shapes["x2"], shapes["y2"]]]
                    if "mask" not in shapes:
                        points.append([points[0][0], points[1][1]])
                        points.append([points[1][0], points[0][1]])
                        self.annotations_list.append(
                            self.generate_rectangle_anns_field(
                                points, label, image_id, object_id,
                                label_to_num))
                    else:
                        mask_dict = {}
                        mask_dict[
                            'size'] = [img_info["height"], img_info["width"]]
                        mask_dict['counts'] = shapes['mask'].encode()
                        mask = decode(mask_dict)
                        contours, hierarchy = cv2.findContours(
                            (mask).astype(np.uint8), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
                        segmentation = []
                        for contour in contours:
                            contour_list = contour.flatten().tolist()
                            if len(contour_list) > 4:
                                segmentation.append(contour_list)
                        self.annotations_list.append(
                            self.generate_polygon_anns_field(
                                points, segmentation, label, image_id,
                                object_id, label_to_num))


class JingLing2COCO(X2COCO):
    """将使用EasyData标注的检测或分割数据集转换为COCO数据集。
    """

    def __init__(self):
        super(JingLing2COCO, self).__init__()

    def generate_images_field(self, json_info, image_id):
        image = {}
        image["height"] = json_info["size"]["height"]
        image["width"] = json_info["size"]["width"]
        image["id"] = image_id + 1
        json_info["path"] = path_normalization(json_info["path"])
        image["file_name"] = osp.split(json_info["path"])[-1]
        return image

    def generate_polygon_anns_field(self, height, width, points, label,
                                    image_id, object_id, label_to_num):
        annotation = {}
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image_id + 1
        annotation["bbox"] = list(
            map(float, self.get_bbox(height, width, points)))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["category_id"] = label_to_num[label]
        annotation["id"] = object_id + 1
        return annotation

    def get_bbox(self, height, width, points):
        polygons = points
        mask = np.zeros([height, width], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        left_top_r = np.min(rows)
        left_top_c = np.min(clos)
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
        return [
            left_top_c, left_top_r, right_bottom_c - left_top_c,
            right_bottom_r - left_top_r
        ]

    def parse_json(self, img_dir, json_dir):
        image_id = -1
        object_id = -1
        labels_list = []
        label_to_num = {}
        for img_file in os.listdir(img_dir):
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            image_id = image_id + 1
            with open(json_file, mode='r', \
                      encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                img_info = self.generate_images_field(json_info, image_id)
                self.images_list.append(img_info)
                anns_type = "bndbox"
                for i, obj in enumerate(json_info["outputs"]["object"]):
                    if i == 0:
                        if "polygon" in obj:
                            anns_type = "polygon"
                    else:
                        if anns_type not in obj:
                            continue
                    object_id = object_id + 1
                    label = obj["name"]
                    if label not in labels_list:
                        self.categories_list.append( \
                            self.generate_categories_field(label, labels_list))
                        labels_list.append(label)
                        label_to_num[label] = len(labels_list)
                    if anns_type == "polygon":
                        points = []
                        for j in range(int(len(obj["polygon"]) / 2.0)):
                            points.append([
                                obj["polygon"]["x" + str(j + 1)],
                                obj["polygon"]["y" + str(j + 1)]
                            ])
                        self.annotations_list.append(
                            self.generate_polygon_anns_field(
                                json_info["size"]["height"], json_info["size"][
                                    "width"], points, label, image_id,
                                object_id, label_to_num))
                    if anns_type == "bndbox":
                        points = []
                        points.append(
                            [obj["bndbox"]["xmin"], obj["bndbox"]["ymin"]])
                        points.append(
                            [obj["bndbox"]["xmax"], obj["bndbox"]["ymax"]])
                        points.append(
                            [obj["bndbox"]["xmin"], obj["bndbox"]["ymax"]])
                        points.append(
                            [obj["bndbox"]["xmax"], obj["bndbox"]["ymin"]])
                        self.annotations_list.append(
                            self.generate_rectangle_anns_field(
                                points, label, image_id, object_id,
                                label_to_num))
