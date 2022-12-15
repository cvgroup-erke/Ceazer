# YOLOv3 SPP
## 该代码来自于https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
## 该项目源自[ultralytics/yolov3](https://github.com/ultralytics/yolov3)
## 1 环境配置：
* Python3.6或者3.7
* Pytorch1.7.1(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)
* pycocotools(Linux: `pip install pycocotools`;   
  Windows: `pip install pycocotools-windows`(不需要额外安装vs))
* 更多环境配置信息，请查看`requirements.txt`文件

## 2 文件结构：
```
  ├── cfg: 配置文件目录
  │    ├── hyp.yaml: 训练网络的相关超参数
  │    └── yolov3-spp.cfg: yolov3-spp网络结构配置 
  │ 
  ├── data: 存储训练时数据集相关信息缓存
  │    └── pascal_voc_classes.json: pascal voc数据集标签
  │ 
  ├── runs: 保存训练过程中生成的所有tensorboard相关文件
  ├── build_utils: 搭建训练网络时使用到的工具
  │     ├── datasets.py: 数据读取以及预处理方法
  │     ├── img_utils.py: 部分图像处理方法
  │     ├── layers.py: 实现的一些基础层结构
  │     ├── parse_config.py: 解析yolov3-spp.cfg文件
  │     ├── torch_utils.py: 使用pytorch实现的一些工具
  │     └── utils.py: 训练网络过程中使用到的一些方法
  │
  ├── train_utils: 训练验证网络时使用到的工具(包括多GPU训练以及使用cocotools)
  ├── weights: 所有相关预训练权重(下面会给出百度云的下载地址)
  ├── model.py: 模型搭建文件
  ├── train.py: 针对单GPU或者CPU的用户使用
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── trans_voc2yolo.py: 将voc数据集标注信息(.xml)转为yolo标注格式(.txt)
  ├── calculate_dataset.py: 1)统计训练集和验证集的数据并生成相应.txt文件
  │                         2)创建data.data文件
  │                         3)根据yolov3-spp.cfg结合数据集类别数创建my_yolov3.cfg文件
  ├── yolo_kmeans.py：聚类生成512尺度的bboxes的聚类中心
  ├── split_train_val.py：按照labels划分数据集，生成train.txt和val.txt
  └── predict_test.py: 简易的预测脚本，使用训练好的权重进行预测测试
```

## 3 训练数据的准备以及目录结构
├── data 利用数据集生成的一系列相关准备文件目录
│    ├── my_train_data.txt:  该文件里存储的是所有训练图片的路径地址
│    ├── my_val_data.txt:  该文件里存储的是所有验证图片的路径地址
│    ├── my_data_label.names:  该文件里存储的是所有类别的名称，一个类别对应一行(这里会根据`.json`文件自动生成)
│    └── my_data.data:  该文件里记录的是类别数类别信息、train以及valid对应的txt文件
```

## 4 预训练权重下载地址（下载后放入weights文件夹中）：
* `yolov3-spp-ultralytics-416.pt`: 链接: https://pan.baidu.com/s/1cK3USHKxDx-d5dONij52lA  密码: r3vm
* `yolov3-spp-ultralytics-512.pt`: 链接: https://pan.baidu.com/s/1k5yeTZZNv8Xqf0uBXnUK-g  密码: e3k1
* `yolov3-spp-ultralytics-608.pt`: 链接: https://pan.baidu.com/s/1GI8BA0wxeWMC0cjrC01G7Q  密码: ma3t
