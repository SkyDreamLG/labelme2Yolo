import sys
import json
import random
import yaml
import shutil
from pathlib import Path
from collections import defaultdict
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QLineEdit, QSlider, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# 设置随机数种子以确保结果的可重复性
random.seed(114514)

# 定义YOLO支持的图像格式列表
image_formats = ["jpg", "jpeg", "png", "bmp", "webp", "tif", ".dng", ".mpo", ".pfm"]

# 复制带有标签的图像到目标文件夹
def copy_labled_img(json_path: Path, target_folder: Path, task: str):
    # 遍历所有支持的图像格式
    for format in image_formats:
        image_path = json_path.with_suffix("." + format)
        # 如果找到对应的图像文件，则复制到目标位置
        if image_path.exists():
            target_path = target_folder / "images" / task / image_path.name
            shutil.copy(image_path, target_path)

# 将LabelMe的JSON文件转换为YOLO格式的文本文件
def json_to_yolo(json_path: Path, sorted_keys: list):
    with open(json_path, "r") as f:
        labelme_data = json.load(f)  # 加载LabelMe的JSON数据

    width = labelme_data["imageWidth"]  # 图像宽度
    height = labelme_data["imageHeight"]  # 图像高度
    yolo_lines = []  # 用于存储YOLO格式的标注行

    # 遍历所有的形状（即标注）
    for shape in labelme_data["shapes"]:
        label = shape["label"]  # 标签名
        points = shape["points"]  # 形状的点坐标
        class_idx = sorted_keys.index(label)  # 根据标签名获取类别的索引
        txt_string = f"{class_idx} "  # 开始构建YOLO格式的字符串

        # 遍历点坐标并标准化
        for x, y in points:
            x /= width
            y /= height
            txt_string += f"{x} {y} "  # 添加标准化后的坐标到字符串中

        yolo_lines.append(txt_string.strip() + "\n")  # 添加一行到列表中

    return yolo_lines  # 返回YOLO格式的标注行列表

# 创建目录如果它不存在
def create_directory_if_not_exists(directory_path):
    directory_path.mkdir(parents=True, exist_ok=True)

# 创建YOLO使用的yaml配置文件
def create_yaml(output_folder: Path, sorted_keys: list):
    # 定义训练和验证集的路径
    train_img_path = Path("images") / "train"
    val_img_path = Path("images") / "val"
    train_label_path = Path("labels") / "train"
    val_label_path = Path("labels") / "val"

    # 创建所有需要的目录
    for path in [train_img_path, val_img_path, train_label_path, val_label_path]:
        create_directory_if_not_exists(output_folder / path)

    # 构建类别名字典
    names_dict = {idx: name for idx, name in enumerate(sorted_keys)}
    # 构建yaml配置字典
    yaml_dict = {
        "path": output_folder.as_posix(),
        "train": train_img_path.as_posix(),
        "val": val_img_path.as_posix(),
        "names": names_dict,
    }

    # 写入yaml配置文件
    yaml_file_path = output_folder / "yolo.yaml"
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(yaml_dict, yaml_file, default_flow_style=False, sort_keys=False)

    # 输出创建成功的提示信息
    print(f"yaml created in {yaml_file_path.as_posix()}")

# 获取标签和JSON文件路径
def get_labels_and_json_path(input_folder: Path):
    json_file_paths = list(input_folder.rglob("*.json"))  # 找到所有的JSON文件
    label_counts = defaultdict(int)  # 初始化标签计数字典

    # 遍历所有JSON文件并统计标签数量
    for json_file_path in json_file_paths:
        with open(json_file_path, "r") as f:
            labelme_data = json.load(f)
        for shape in labelme_data["shapes"]:
            label = shape["label"]
            label_counts[label] += 1

    # 按标签出现次数排序
    sorted_keys = sorted(label_counts, key=lambda k: label_counts[k], reverse=True)
    return sorted_keys, json_file_paths  # 返回排序后的标签列表和JSON文件路径列表

# 将LabelMe的JSON文件转换为YOLO格式，并分割训练和验证集
def labelme_to_yolo(
    json_file_paths: list, output_folder: Path, sorted_keys: list, split_rate: float, progress_callback
):
    random.shuffle(json_file_paths)  # 打乱JSON文件列表顺序

    # 计算分割点
    split_point = int(split_rate * len(json_file_paths))
    train_set = json_file_paths[:split_point]  # 分割训练集
    val_set = json_file_paths[split_point:]  # 分割验证集

    total_files = len(train_set) + len(val_set)  # 总文件数
    processed_files = 0  # 已处理文件数

    # 处理训练集
    for json_file_path in train_set:
        txt_name = json_file_path.with_suffix(".txt").name
        yolo_lines = json_to_yolo(json_file_path, sorted_keys)
        output_json_path = Path(output_folder / "labels" / "train" / txt_name)
        with open(output_json_path, "w") as f:
            f.writelines(yolo_lines)
        copy_labled_img(json_file_path, output_folder, task="train")
        processed_files += 1
        progress_callback.emit(int(processed_files / total_files * 100))  # 更新进度条

    # 处理验证集
    for json_file_path in val_set:
        txt_name = json_file_path.with_suffix(".txt").name
        yolo_lines = json_to_yolo(json_file_path, sorted_keys)
        output_json_path = Path(output_folder / "labels" / "val" / txt_name)
        with open(output_json_path, "w") as f:
            f.writelines(yolo_lines)
        copy_labled_img(json_file_path, output_folder, task="val")
        processed_files += 1
        progress_callback.emit(int(processed_files / total_files * 100))  # 更新进度条

# 转换线程类
class ConvertThread(QThread):
    progress = pyqtSignal(int)  # 进度信号
    error = pyqtSignal(str)  # 错误信号
    finished = pyqtSignal()  # 完成信号

    def __init__(self, input_folder, output_folder, split_rate):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.split_rate = split_rate

    def run(self):
        try:
            # 获取标签和JSON文件路径，创建yaml配置文件，进行转换
            sorted_keys, json_file_paths = get_labels_and_json_path(self.input_folder)
            create_yaml(self.output_folder, sorted_keys)
            labelme_to_yolo(json_file_paths, self.output_folder, sorted_keys, self.split_rate, self.progress)
        except Exception as e:
            self.error.emit(str(e))  # 发出错误信号
        else:
            self.finished.emit()  # 发出完成信号

# 主应用程序窗口类
class LabelMe2YoloApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  # 初始化用户界面

    def initUI(self):
        self.setWindowTitle("LabelMe 转 YOLO 转换器")  # 设置窗口标题

        self.input_folder_path = ""  # 输入文件夹路径
        self.output_folder_path = ""  # 输出文件夹路径
        self.split_rate = 0.8  # 默认分割率

        layout = QVBoxLayout()  # 主布局

        # 输入文件夹布局
        input_layout = QHBoxLayout()
        self.input_label = QLabel("输入文件夹:")
        self.input_path_display = QLineEdit()
        self.input_browse_button = QPushButton("浏览")
        self.input_browse_button.clicked.connect(self.browse_input_folder)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path_display)
        input_layout.addWidget(self.input_browse_button)
        layout.addLayout(input_layout)

        # 输出文件夹布局
        output_layout = QHBoxLayout()
        self.output_label = QLabel("输出文件夹:")
        self.output_path_display = QLineEdit()
        self.output_browse_button = QPushButton("浏览")
        self.output_browse_button.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_display)
        output_layout.addWidget(self.output_browse_button)
        layout.addLayout(output_layout)

        # 分割比例布局
        split_layout = QHBoxLayout()
        self.split_label = QLabel("分割比例 (训练/验证):")
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setMinimum(0)
        self.split_slider.setMaximum(100)
        self.split_slider.setValue(80)
        self.split_slider.valueChanged.connect(self.update_split_rate)
        self.split_rate_display = QLabel("0.8")
        split_layout.addWidget(self.split_label)
        split_layout.addWidget(self.split_slider)
        split_layout.addWidget(self.split_rate_display)
        layout.addLayout(split_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 转换按钮
        self.convert_button = QPushButton("转换")
        self.convert_button.clicked.connect(self.convert)
        layout.addWidget(self.convert_button)

        self.setLayout(layout)

    # 浏览输入文件夹
    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder:
            self.input_folder_path = folder
            self.input_path_display.setText(folder)

    # 浏览输出文件夹
    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder_path = folder
            self.output_path_display.setText(folder)

    # 更新分割比例
    def update_split_rate(self):
        self.split_rate = self.split_slider.value() / 100.0
        self.split_rate_display.setText(f"{self.split_rate:.2f}")

    # 转换操作
    def convert(self):
        input_folder = Path(self.input_folder_path)
        output_folder = Path(self.output_folder_path)
        split_rate = self.split_rate

        # 创建转换线程
        self.thread = ConvertThread(input_folder, output_folder, split_rate)
        self.thread.progress.connect(self.update_progress)
        self.thread.error.connect(self.show_error_message)
        self.thread.finished.connect(self.show_success_message)
        self.thread.start()

    # 更新进度条
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    # 显示错误信息弹窗
    def show_error_message(self, message):
        QMessageBox.critical(self, "错误", f"程序出错。\n请检查输入输出文件夹是否设置正确且输入文件夹内部文件是否合法且一一对应。\n错误详情: {message}")

    # 显示成功信息弹窗
    def show_success_message(self):
        QMessageBox.information(self, "成功", "转换成功完成！")

# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelMe2YoloApp()
    window.show()
    sys.exit(app.exec())
