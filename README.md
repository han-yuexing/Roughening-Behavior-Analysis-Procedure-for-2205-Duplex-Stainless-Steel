# Roughening-Behavior-Analysis-Procedure-for-2205-Duplex-Stainless-Steel
# 工程文件说明

本工程包含以下文件：

## 文件列表

- `CCNet_loss.py`: CCNet中包含的损失函数文件。
- `CCNet.py`: CCNet的网络结构代码。
- `Data.py`: 对我们格式的标注数据进行读取和处理的代码，标注格式为npy。
- `FCN8s.py`: FCN8s模型结构的代码。
- `find_contours.py`: 对标注图进行轮廓查找的代码，以确定各个沉淀物的位置、尺寸信息。
- `histogram.py`: 对所有沉淀物的半径进行直方图统计的代码。
- `main.py`: 用来训练模型的主要代码。
- `radius.py`: 对图像的比例尺进行读取，并根据比例尺和沉淀物尺寸的像素距离转化为真实的尺寸的代码。
- `SegNet.py`: SegNet模型的代码。
- `show_result.py`: 将npy文件彩色化显示在原图像上的代码。
- `test.py`: 使用训练好的模型对特定文件夹中的图像进行测试，并保存结果为npy文件的代码。
- `train.py`: 模型训练的细节，供main.py文件使用的代码。
- `Unet.py`: Unet模型的代码。

## 运行环境和所需函数库

- 操作系统：Windows 7.0 以上，Linux系统。
- 运行语言：Python。
- 运行函数库：
  - `sklearn`
  - `pandas`
  - `numpy`
  - `opencv_python`
  - `pytorch`
  - `Pillow`
  - `matplotlib`
