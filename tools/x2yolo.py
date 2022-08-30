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


class X2YOLO(object):
    def __init__(self):
        self.images_list = []
        self.categories_list = []
        self.annotations_list = []
        self.labelList = []

# planthopper_135
class Planthopper135_2yolo(X2YOLO):
    def __init__(self):
        super(Planthopper135_2yolo, self).__init__()

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

    
    def parse_json(self, img_dir, json_dir):
        for img_file in os.listdir(img_dir):
            # print(img_file)
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            new_json_file = osp.join(img_dir, img_name_part + ".txt")
            if not osp.exists(json_file):
                print("!!can't find "+ img_name_part)
                # os.remove(osp.join(img_dir, img_file))
                continue
            
            with open(json_file, mode='r', \
                      encoding=get_encoding(json_file)) as j:
                
                image = cv2.imread(osp.join(img_dir, img_file))
                size = image.shape

                json_info = json.load(j)
                json_info = json_info['labels']

                img_info = []
                for region in json_info:
                    if region['name'] not in self.labelList:
                        self.labelList.append(region['name'])
                    labelId = self.labelList.index(region['name'])
                    
                    img_info.append([
                        str(labelId), 
                        str(min((region['x1']+region['x2'])/2/size[1], 1)), str(min(1, (region['y1']+region['y1'])/2/size[0])), str(min(1, ((region['x2']-region['x1'])/size[1]))), str(min(1, ((region['y2']-region['y1'])/size[0])))])

                f = open(new_json_file, "w+")
                for i in img_info:
                    # print(i)
                    f.writelines(' '.join(i)+'\n')
                f.close()

    def convert(self, dataset_dir, dataset_save_dir, train, val, test):
        assert osp.exists(dataset_dir), "The json folder does not exist!"

        if not osp.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)
        
        # Convert the image files.
        coco_category = ['train', 'val', 'test']
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

        with open(dataset_save_dir + '/labels.txt', 'w') as f:
            f.writelines('\n'.join(self.labelList))

        
# planthopper_2417
class Planthopper2417_2yolo(X2YOLO):
    def __init__(self):
        super(Planthopper2417_2yolo, self).__init__()

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

    def generate_images_field(self, json_info, picSize):
        image = []
        for region in json_info['regions']:
            regionOrigin = region['region']
            
            image.append([str(region['cls']-1), str((regionOrigin[0]+regionOrigin[2])/2/picSize[0]), str((regionOrigin[1]+regionOrigin[3])/2/picSize[1]), str((regionOrigin[2]-regionOrigin[0])/picSize[0]), str((regionOrigin[3]-regionOrigin[1])/picSize[1])])

        return image
    
    def parse_json(self, img_dir, json_dir):
        for img_file in os.listdir(img_dir):
            print(img_file)
            img_name_part = osp.splitext(img_file)[0]
            json_file = osp.join(json_dir, img_name_part + ".txt")
            new_json_file = osp.join(img_dir, img_name_part + ".txt")
            if not osp.exists(json_file):
                os.remove(osp.join(img_dir, img_file))
                continue
            
            with open(json_file, mode='r', \
                      encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                
                
                image = cv2.imread(osp.join(img_dir, img_file))
                size = image.shape

                img_info = self.generate_images_field(json_info, [size[1], size[0]])
                f = open(new_json_file, "w+")
                for i in img_info:
                    # print(i)
                    f.writelines(' '.join(i)+'\n')
                f.close()

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
        coco_category = ['train', 'val', 'test']
        new_image_dir_detail = []
        for i in coco_category:
            new_image_dir_detail.append(osp.join(dataset_save_dir, i))
            new_path = osp.join(dataset_save_dir, i)
            if not osp.exists(new_path):
                os.makedirs(new_path)

        self.create_json_list(new_image_dir_detail, 'origin', train, val, test)

        # Convert the json files.
        for i in range(3):
            self.images_list = []
            self.categories_list = []
            self.annotations_list = []
            self.parse_json(new_image_dir_detail[i], 'origin')

    