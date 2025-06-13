import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from centermask.config import get_cfg
import pandas as pd

# 初始化模型配置和預測器
def setup_predictor(config_path, weights_path, device="cuda"):
    cfg = get_cfg()  # 獲取配置對象
    cfg.merge_from_file(config_path)  # 從配置文件中加載配置
    cfg.MODEL.WEIGHTS = weights_path  # 設置模型權重路徑
    cfg.MODEL.DEVICE = device  # 設置設備
    return DefaultPredictor(cfg)  # 返回配置好的預測器

# 判斷是否在邊界上的物體
def is_on_boundary(mask, boundary_threshold=1):
    h, w = mask.shape
    top_boundary = np.sum(mask[:boundary_threshold, :]) > 0
    bottom_boundary = np.sum(mask[-boundary_threshold:, :]) > 0
    left_boundary = np.sum(mask[:, :boundary_threshold]) > 0
    right_boundary = np.sum(mask[:, -boundary_threshold:]) > 0
    return top_boundary or bottom_boundary or left_boundary or right_boundary

# 執行預測並保存結果
def classify_and_save(image, predictor, output_image_path, output_json_path, image_index, image_data):
    outputs = predictor(image)  # 使用預測器對圖像進行預測
    instances = outputs["instances"].to("cpu")  # 獲取預測實例並轉移到 CPU

    confidence_threshold = 0.5  # 置信度閾值
    high_confidence_idxs = instances.scores >= confidence_threshold  # 篩選出置信度高的實例
    instances = instances[high_confidence_idxs]  # 保留高置信度的實例

    # 類別名稱
    class_names = ['Float', 'Wood', 'Styrofoam', 'Bottle', 'Buoy']
    # 類別顏色（用於可視化）
    class_colors = {
     'Float': (255/255, 0/255, 128/255),
     'Wood':  (255/255, 128/255, 0/255),
     'Styrofoam': (255/255, 255/255, 128/255),
     'Bottle': (0/255, 0/255, 128/255),
     'Buoy': (0/255, 255/255, 128/255),
    }

     # 創建標籤文本
    labels = [
         f"{class_names[cls]} {score:.0%}"
         for cls, score in zip(instances.pred_classes, instances.scores)
     ]
    
    # 初始化每個原圖的類別信息
    if "category_info" not in image_data[image_index]:
        image_data[image_index]["category_info"] = {class_name: {"count": 0, "total_area": 0, "max_area": None, "min_area": None, "boundary_count": 0} for class_name in class_names}

    category_info = image_data[image_index]["category_info"]

    # 記錄每個類別的實例信息
    for i in range(len(instances)):
        class_id = instances.pred_classes[i]
        class_name = class_names[class_id]
        mask = instances.pred_masks[i].numpy()
        area = int(np.sum(mask))
        is_boundary = is_on_boundary(mask)

        if is_boundary:
            category_info[class_name]["boundary_count"] += 1  # 邊界上的物體記錄

        # 更新類別信息
        category_info[class_name]["count"] += 1
        category_info[class_name]["total_area"] += area
        if category_info[class_name]["max_area"] is None or area > category_info[class_name]["max_area"]:
            category_info[class_name]["max_area"] = area
        if category_info[class_name]["min_area"] is None or area < category_info[class_name]["min_area"]:
            category_info[class_name]["min_area"] = area

    # 保存預測結果圖像
    v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(predictor.cfg.DATASETS.TEST[0]), instance_mode=ColorMode.SEGMENTATION)
    out = v.overlay_instances(
        labels=labels,
        #boxes=instances.pred_boxes if instances.has("pred_boxes") else None,
        masks=instances.pred_masks,  
        assigned_colors=[class_colors[class_names[cls]] for cls in instances.pred_classes]
    )

    cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])
    
    return instances
C = 0.24

# 最終將所有數據保存到 Excel 表格
def save_excel(image_data, output_excel_path):
    # 将每张原图的数据信息整理成一个表格
    excel_data = {
        "圖像編號": [],
        "類別": [],
        "數量": [],
        "總面積(cm²)": [],
        "最大面積(cm²)": [],
        "最小面積(cm²)": []
    }

    for image_name, data in image_data.items():
        for class_name, class_info in data["category_info"].items():
            if class_info["count"] > 1:
                total_count = class_info["count"] - class_info["boundary_count"]  # 減去邊界上的物體數量
            else:
                total_count = class_info["count"]

            # 乘以 C 進行面積轉換，先檢查是否為 None
            total_area_cm2 = class_info["total_area"] * C if class_info["total_area"] is not None else None
            max_area_cm2 = class_info["max_area"] * C if class_info["max_area"] is not None else None
            # 更新 min_area_cm2 的計算，過濾掉 None 和 0 面積的情況
            min_area_cm2 = class_info["min_area"] * C if class_info["max_area"] is not None else None
            if min_area_cm2 is not None and min_area_cm2 <= 0: min_area_cm2=1
            
            
            excel_data["圖像編號"].append(image_name)
            excel_data["類別"].append(class_name)
            excel_data["數量"].append(total_count)
            excel_data["總面積(cm²)"].append(total_area_cm2)
            excel_data["最大面積(cm²)"].append(max_area_cm2)
            excel_data["最小面積(cm²)"].append(min_area_cm2)

    # 创建 DataFrame
    df = pd.DataFrame(excel_data)

    # 計算總結信息
    summary_data = {
        "類別": [],
        "數量": [],
        "總面積(cm²)": [],
        "最大面積(cm²)": [],
        "最小面積(cm²)": []
    }

    class_names = ['Float', 'Wood', 'Styrofoam', 'Bottle', 'Buoy']

    for class_name in class_names:
        class_df = df[df["類別"] == class_name]
        count_sum = class_df["數量"].sum()
        total_area_sum = class_df["總面積(cm²)"].sum()
        max_area = class_df["最大面積(cm²)"].max() if not class_df["最大面積(cm²)"].isnull().all() else None
        # 更新最小面積計算，過濾掉0面積的情況
        min_area = class_df["最小面積(cm²)"][class_df["最小面積(cm²)"] > 1].min() if not class_df["最小面積(cm²)"].isnull().all() else None

        summary_data["類別"].append(class_name)
        summary_data["數量"].append(count_sum)
        summary_data["總面積(cm²)"].append(total_area_sum)
        summary_data["最大面積(cm²)"].append(max_area)
        summary_data["最小面積(cm²)"].append(min_area)

    summary_df = pd.DataFrame(summary_data)

    # 將原始數據和總結信息寫入 Excel 文件
    with pd.ExcelWriter(output_excel_path) as writer:
        df.to_excel(writer, sheet_name="個別圖像數據", index=False)
        summary_df.to_excel(writer, sheet_name="總數據", index=False)

    print("辨識完成")
    print(f"所有偵測出的類別已儲存到 Excel 表格： {output_excel_path}")
