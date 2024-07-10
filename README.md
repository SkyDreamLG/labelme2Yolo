# labelme2Yolo
将labelme数据标注格式转换为Yolo语义分割数据集，并可自动划分训练集和验证集

## 使用

直接运行releace内的exe文件或源码内的python文件即可。脚本根据文件名对图片-标注进行匹配。

```shell
python main.py
```

示例：

```shell
python main.py
```


## 训练

参照[Yolo官方文档](https://docs.ultralytics.com/tasks/segment/)


示例：

```python
from ultralytics import YOLO
from ultralytics import settings

settings.update({'datasets_dir': './'})
model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

if __name__ == '__main__':
    # Train the model
    results = model.train(data='./datasets/yolo.yaml', epochs=100, imgsz=640)
```


## 疑难解答

YOLO找不到训练集和测试集文件

YOLO在查找路径时，会将三个路径拼接到一起：
- setting中的datasets_dir
- 数据集yaml中的path
- 数据集yaml中的train、value

可以通过以下方式来修改 datasets_dir：
```python
from ultralytics import settings
settings.update({'datasets_dir': './'})
```
