# 数据集格式转换工具

基于PaddleX完善的数据集格式转换工具

## Start

### 安装依赖

```bash
pip install -r requirement.txt
```

### 运行程序

```bash
python convert.py --source [source_dataset] --to [to_dataset] --pics [pics_dir] --annotations [annotations_dir] --save_dir [output_dir]
# 可选--train 0.7 --val 0.2 --test 0.1
```

> 参数列表
> 
> - source  转化前数据集类型 **必填**
> - to   转化后数据集类型 **必填**
> - pics 转化前数据集图片存储位置 **必填**
> - annotations 转化前数据集标签存储位置（个别任务无需提供）
> - save_dir 转化后数据集保存位置 **必填**
> - train 训练集划分比例（默认0.7）
> - val  测试集划分比例（默认0.2）
> - test（默认0.1）

## 支持列表

### X2COCO

- [x] labelme
- [x] labelImg
- [x] planthopper_135
- [x] planthopper_2417
- [ ] easydata
- [ ] jingling

### X2yolo

- [x] planthopper_135
- [x] planthopper_2417

### X2Seg

- [ ] jingling
- [ ] labelme
- [ ] easydata

### X2VOC

- [ ] labelme
- [ ] easydata

### X2ImageNet

- [ ] easydata
- [ ] jingling

## 待优化

- [ ] 图片仅支持jpg
