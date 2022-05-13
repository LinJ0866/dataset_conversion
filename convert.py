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

import argparse
from tools import *

easydata2imagenet = EasyData2ImageNet().convert
jingling2imagenet = JingLing2ImageNet().convert
jingling2seg = JingLing2Seg().convert
labelme2seg = LabelMe2Seg().convert
easydata2seg = EasyData2Seg().convert
labelme2voc = LabelMe2VOC().convert
easydata2voc = EasyData2VOC().convert
labelme2coco = LabelMe2COCO().convert
easydata2coco = EasyData2COCO().convert
jingling2coco = JingLing2COCO().convert
labelimg2coco = LabelImg2COCO().convert
planthopper = planthopper().convert


def dataset_conversion(source, to, pics, anns, save_dir, train, val, test):
    if source.lower() == 'easydata' and to.lower() == 'imagenet':
        easydata2imagenet(pics, anns, save_dir)
    elif source.lower() == 'jingling' and to.lower() == 'imagenet':
        jingling2imagenet(pics, anns, save_dir, train, val, test)
    elif source.lower() == 'jingling' and to.lower() == 'seg':
        jingling2seg(pics, anns, save_dir, train, val, test)
    elif source.lower() == 'labelme' and to.lower() == 'seg':
        labelme2seg(pics, anns, save_dir)
    elif source.lower() == 'easydata' and to.lower() == 'seg':
        easydata2seg(pics, anns, save_dir, train, val, test)
    elif source.lower() == 'labelme' and to.lower() == 'pascalvoc':
        labelme2voc(pics, anns, save_dir, train, val, test)
    elif source.lower() == 'easydata' and to.lower() == 'pascalvoc':
        easydata2voc(pics, anns, save_dir, train, val, test)
    elif source.lower() == 'labelme' and to.lower() == 'coco':
        labelme2coco(pics, anns, save_dir, train, val, test)
    elif source.lower() == 'labelimg' and to.lower() == 'coco':
        labelimg2coco(pics, anns, save_dir, train, val, test)
    elif source.lower() == 'easydata' and to.lower() == 'coco':
        easydata2coco(pics, anns, save_dir, train, val, test)
    elif source.lower() == 'jingling' and to.lower() == 'coco':
        jingling2coco(pics, anns, save_dir, train, val, test)
    elif source.lower() == 'planthopper' and to.lower() == 'coco':
        planthopper(pics, save_dir, train, val, test)
    else:
        raise Exception("Converting from {} to {} is not supported.".format(
            source, to))

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-se",
        default=None,
        help="define dataset format before the conversion")
    parser.add_argument(
        "--to",
        "-to",
        default=None,
        help="define dataset format after the conversion")
    parser.add_argument(
        "--pics",
        "-p",
        default=None,
        help="define pictures directory path")
    parser.add_argument(
        "--annotations",
        "-a",
        default=None,
        help="define annotations directory path")
    parser.add_argument(
        "--save_dir",
        "-s",
        default=None,
        help="path to save inference model")
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="define the value of train dataset(E.g 0.7)")
    parser.add_argument(
        "--val",
        type=float,
        default=0.2,
        help="define the value of validation dataset(E.g 0.2)")
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="define the value of test dataset(E.g 0.1)")

    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()

    assert args.source is not None, "--source should be defined while converting dataset"
    assert args.to is not None, "--to should be defined to confirm the taregt dataset format"
    assert args.pics is not None, "--pics should be defined to confirm the pictures path"
    assert args.save_dir is not None, "--save_dir should be defined to store taregt dataset"
    # assert args.train+args.val+args.test != 1.0, "train + val + test should be 1"

    dataset_conversion(
        args.source,
        args.to,
        args.pics,
        args.annotations,
        args.save_dir,
        args.train,
        args.val,
        args.test
    )

if __name__ == "__main__":
    main()