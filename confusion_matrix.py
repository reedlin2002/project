import cv2
import numpy as np
import os
from detectron2.engine import DefaultPredictor
from centermask.config import get_cfg
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 加载模型
cfg = get_cfg()
cfg.merge_from_file(r"檔案位置")
cfg.MODEL.WEIGHTS = r"檔案位置"
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)

# 加载训练集的 JSON
with open(r"檔案位置") as f:
    val_data = json.load(f)

# 构建 image_id 到 file_name 的映射表
image_id_to_file = {image['id']: image['file_name'] for image in val_data['images']}

# 预测结果与实际类别的存储
y_true = []
y_pred = []

# 可信度高于 0.5
confidence_threshold = 0.5

# 类别映射
category_id_to_label = {
    1: 'Float',
    2: 'Wood',
    3: 'Styrofoam',
    4: 'Bottle',
    5: 'Buoy'
}

# 图像文件夹路径
image_folder = r'檔案位置'

# 处理每张图片
for image_id, file_name in image_id_to_file.items():
    # 获取正确的图像路径
    image_path = os.path.join(image_folder, file_name)
    
    # 读取图片
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error loading image {image_path}")
        continue
    
    # 进行预测
    outputs = predictor(im)
    
    # 获取预测类别和可信度
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_scores = outputs["instances"].scores.cpu().numpy()
    
    # 根据可信度筛选预测结果
    high_conf_indices = np.where(pred_scores >= confidence_threshold)[0]
    pred_classes = pred_classes[high_conf_indices]
    
    # 将预测类别从0基础调整到1基础
    pred_classes = pred_classes + 1
    
    # 找到对应的标注数据
    true_classes = [ann['category_id'] for ann in val_data['annotations'] if ann['image_id'] == image_id]
    
    # 添加到真实标签和预测标签列表中
    y_true.extend(true_classes)
    
    # 如果预测多于真实标签，裁剪预测
    if len(pred_classes) > len(true_classes):
        pred_classes = pred_classes[:len(true_classes)]
    
    # 如果预测少于真实标签，补齐标签
    if len(pred_classes) < len(true_classes):
        pred_classes = np.pad(pred_classes, (0, len(true_classes) - len(pred_classes)), 'constant', constant_values=-1)
    
    # 添加到预测标签列表中
    y_pred.extend(pred_classes)

# 计算混淆矩阵和分类报告
labels = [1, 2, 3, 4, 5]
label_names = ['Float', 'Wood', 'Styrofoam', 'Bottle', 'Buoy']

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 计算准确率
accuracy = np.trace(cm) / np.sum(cm)
print(f'Accuracy: {accuracy:.4f}')  # 打印准确率

# 显示混淆矩阵
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

# 添加标题和轴标签
ax.set_title('Train Confusion Matrix')
ax.set_xticks(np.arange(len(label_names)))
ax.set_yticks(np.arange(len(label_names)))
ax.set_xticklabels(label_names)
ax.set_yticklabels(label_names)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

# 显示数值
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")

plt.tight_layout()
plt.show()

# 计算并显示分类报告
report = classification_report(y_true, y_pred, labels=labels, target_names=label_names, zero_division=0)
print("Classification Report:\n", report)
