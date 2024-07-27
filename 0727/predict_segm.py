import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from centermask.config import get_cfg
import json

# 初始化模型配置和預測器
def setup_predictor(config_path, weights_path, device="cuda"):
    """
    配置和初始化 CenterMask2 模型預測器。
    
    :param config_path: 配置文件路徑
    :param weights_path: 預訓練權重文件路徑
    :param device: 使用的設備，默認為 "cuda" (GPU)
    :return: 配置好的 DefaultPredictor 對象
    """
    cfg = get_cfg()  # 獲取配置對象
    cfg.merge_from_file(config_path)  # 從配置文件中加載配置
    cfg.MODEL.WEIGHTS = weights_path  # 設置模型權重路徑
    cfg.MODEL.DEVICE = device  # 設置設備
    return DefaultPredictor(cfg)  # 返回配置好的預測器

# 執行預測並保存結果
def classify_and_save(image, predictor, output_image_path, output_json_path):
    """
    對圖像進行預測，並將結果保存到文件中。
    
    :param image: 輸入圖像，NumPy 陣列格式
    :param predictor: 已配置的 DefaultPredictor 對象
    :param output_image_path: 輸出圖像的保存路徑
    :param output_json_path: 輸出 JSON 文件的保存路徑
    """
    outputs = predictor(image)  # 使用預測器對圖像進行預測
    instances = outputs["instances"].to("cpu")  # 獲取預測實例並轉移到 CPU

    confidence_threshold = 0.5  # 置信度閾值
    high_confidence_idxs = instances.scores >= confidence_threshold  # 篩選出置信度高的實例
    instances = instances[high_confidence_idxs]  # 保留高置信度的實例

    # 類別名稱
    class_names = [
        'Plastic bucket',
        'Iron bucket',
        'Plastic basket',
        'Rope',
        'Fishing Net',
        'Float',
        'Wood',
        'Lifebuoy',
        'Styrofoam',
        'Pipe',
        'Bottle',
        'Tire'
    ]

    # 類別顏色（用於可視化）
    class_colors = {
        'Plastic bucket': (255/255, 0/255, 0/255),
        'Iron bucket': (0/255, 255/255, 0/255),
        'Plastic basket': (0/255, 0/255, 255/255),
        'Rope': (255/255, 255/255, 0/255),
        'Fishing Net': (0/255, 255/255, 255/255),
        'Float': (255/255, 0/255, 128/255),
        'Wood': (0/255, 0/255, 0/255),
        'Lifebuoy': (255/255, 255/255, 128/255),
        'Styrofoam': (255/255, 0/255, 255/255),
        'Pipe': (255/255, 255/255, 255/255),
        'Bottle': (0/255, 0/255, 128/255),
        'Tire': (0/255, 255/255, 128/255),
    }

    # 創建標籤文本
    labels = [
        f"{class_names[cls]} {score:.0%}"
        for cls, score in zip(instances.pred_classes, instances.scores)
    ]

    # 使用 Visualizer 對圖像進行可視化
    v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(predictor.cfg.DATASETS.TEST[0]), instance_mode=ColorMode.SEGMENTATION)
    out = v.overlay_instances(
        labels=labels,  # 預測標籤
        boxes=instances.pred_boxes if instances.has("pred_boxes") else None,  # 預測框
        masks=instances.pred_masks,  # 預測掩碼
        assigned_colors=[class_colors[class_names[cls]] for cls in instances.pred_classes]  # 類別顏色
    )

    # 初始化類別信息
    category_info = {class_name: {"count": 0, "instances": []} for class_name in class_names}

    # 記錄每個類別的實例信息
    for i in range(len(instances)):
        class_id = instances.pred_classes[i]
        class_name = class_names[class_id]
        bbox = instances.pred_boxes[i].tensor.numpy()[0]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        mask = instances.pred_masks[i].numpy()
        area = int(np.sum(mask))
        category_info[class_name]["count"] += 1
        category_info[class_name]["instances"].append({"x": x, "y": y, "width": w, "height": h, "area": area})

    # 打印每個類別的數量
    for class_name, info in category_info.items():
        print(f"{class_name}: 數量 = {info['count']}")

    # 顯示預測結果（可選）
    # cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)  # 等待用戶按鍵
    cv2.destroyAllWindows()  # 關閉所有 OpenCV 窗口
    
    # 保存預測結果圖像
    cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])

    # 保存類別信息到 JSON 文件
    category_info_json = json.dumps(category_info, indent=4)
    with open(output_json_path, "w") as f:
        f.write(category_info_json)

    print(f"垃圾類別已儲存到 {output_json_path}")
