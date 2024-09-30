import cv2
import numpy as np
import os
from detectron2.engine import DefaultPredictor
from centermask.config import get_cfg
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# 加载模型
cfg = get_cfg()
cfg.merge_from_file(r"C:\Users\JerryLin\Desktop\0911_pth\beach_V_39_eSE_FPN_ms_3x.yaml")
cfg.MODEL.WEIGHTS = r"C:\Users\JerryLin\Desktop\0911_pth\model_final.pth"
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)

# 加载训练集的 JSON
with open(r"C:\Users\JerryLin\Desktop\0911_pth\beach\annotations\beach_train.json") as f:
    val_data = json.load(f)

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

# 处理每张图片
for image_info in val_data['images']:
    image_id = image_info['id']
    image_filename = f"{image_id:06d}.bmp"
    image_folder = r'C:\Users\JerryLin\Desktop\0911_pth\beach\train\beach_train'
    image_path = os.path.join(image_folder, image_filename)
    
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
    
    # 打印真实标签和预测标签的数量
    print(f"Image ID: {image_id}")
    print(f"True Classes (Count: {len(true_classes)}): {true_classes}")
    print(f"Pred Classes (Count: {len(pred_classes)}): {pred_classes}")
    
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

# 打印所有图片的真实标签和预测标签
print(f"Length of y_true: {len(y_true)}")
print(f"Length of y_pred: {len(y_pred)}")

# 定义标签和名称
labels = [1, 2, 3, 4, 5]  # 根据你的类别编号
label_names = ['Float', 'Wood', 'Styrofoam', 'Bottle', 'Buoy']  # 根据你的类别名称

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 显示混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 计算并显示分类报告
report = classification_report(y_true, y_pred, labels=labels, target_names=label_names, zero_division=0)
print("Classification Report:\n", report)
