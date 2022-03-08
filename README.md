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

# 例： python convert.py --source labelimg --to coco --pics shootDataset --annotations shootDataset --save_dir coco
```

> 可选参数
> 
> - train 训练集划分比例（默认0.7）
> - val  测试集划分比例（默认0.2）
> - test（默认0.1）

## 支持列表

### X2COCO

- [x] labelme
- [x] labelImg
- [ ] easydata
- [ ] jingling

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
