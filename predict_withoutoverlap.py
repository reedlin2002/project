import os
import cv2
import numpy as np
import pandas as pd
from detectron2.engine import DefaultPredictor
from centermask.config import get_cfg

# 設置模型預測器
def setup_predictor(config_path, weights_path, device="cuda"):
    cfg = get_cfg()  # 獲取配置對象
    cfg.merge_from_file(config_path)  # 從配置文件中加載配置
    cfg.MODEL.WEIGHTS = weights_path  # 設置模型權重路徑
    cfg.MODEL.DEVICE = device  # 設置設備
    return DefaultPredictor(cfg)  # 返回配置好的預測器

# 圖片切割與過濾功能
def process_images(input_folder, output_excel_path, predictor):
    image_data = {}
    
    # 遍歷資料夾中的圖片
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        
        # 初始化每個原圖的類別信息
        class_names = ['Float', 'Wood', 'Styrofoam', 'Bottle', 'Buoy']
        image_data[image_name] = {"category_info": {class_name: {"count": 0, "total_area": 0, "max_area": None, "min_area": None, "boundary_count": 0} for class_name in class_names}}
        category_info = image_data[image_name]["category_info"]
        
        # 切割圖片為 640x480 的小圖片
        height, width, _ = image.shape
        for y in range(0, height, 480):
            for x in range(0, width, 640):
                sub_image = image[y:y+480, x:x+640]
                
                # 計算黑色像素比例
                black_pixel_count = np.sum(np.all(sub_image == [0, 0, 0], axis=-1))
                total_pixel_count = sub_image.shape[0] * sub_image.shape[1]
                black_ratio = black_pixel_count / total_pixel_count
                
                # 如果黑色像素比例 >= 40%，則跳過該圖片
                if black_ratio >= 0.40:
                    continue
                
                # 進行物件辨識
                outputs = predictor(sub_image)
                instances = outputs["instances"].to("cpu")
                confidence_threshold = 0.5
                high_confidence_idxs = instances.scores >= confidence_threshold
                instances = instances[high_confidence_idxs]
                
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

    # 儲存數據到 Excel
    save_excel(image_data, output_excel_path)

# 判斷是否在邊界上的物體
def is_on_boundary(mask, boundary_threshold=1):
    h, w = mask.shape
    top_boundary = np.sum(mask[:boundary_threshold, :]) > 0
    bottom_boundary = np.sum(mask[-boundary_threshold:, :]) > 0
    left_boundary = np.sum(mask[:, :boundary_threshold]) > 0
    right_boundary = np.sum(mask[:, -boundary_threshold:]) > 0
    return top_boundary or bottom_boundary or left_boundary or right_boundary

# 最終將所有數據保存到 Excel 表格
def save_excel(image_data, output_excel_path):
    C = 0.24  # 假設面積轉換係數
    excel_data = {
        "圖像編號": [], "類別": [], "數量": [], "總面積(cm²)": [], "最大面積(cm²)": [], "最小面積(cm²)": []
    }

    for image_name, data in image_data.items():
        for class_name, class_info in data["category_info"].items():
            total_count = class_info["count"] - class_info["boundary_count"] if class_info["count"] > 1 else class_info["count"]
            total_area_cm2 = class_info["total_area"] * C if class_info["total_area"] is not None else None
            max_area_cm2 = class_info["max_area"] * C if class_info["max_area"] is not None else None
            min_area_cm2 = class_info["min_area"] * C if class_info["max_area"] is not None else None
            if min_area_cm2 is not None and min_area_cm2 <= 0: min_area_cm2 = 1
            
            excel_data["圖像編號"].append(image_name)
            excel_data["類別"].append(class_name)
            excel_data["數量"].append(total_count)
            excel_data["總面積(cm²)"].append(total_area_cm2)
            excel_data["最大面積(cm²)"].append(max_area_cm2)
            excel_data["最小面積(cm²)"].append(min_area_cm2)

    # 创建 DataFrame
    df = pd.DataFrame(excel_data)

    # 計算每個類別的總結信息
    summary_data = {
        "類別": [], "數量": [], "總面積(cm²)": [], "最大面積(cm²)": [], "最小面積(cm²)": []
    }
    class_names = ['Float', 'Wood', 'Styrofoam', 'Bottle', 'Buoy']

    for class_name in class_names:
        class_df = df[df["類別"] == class_name]
        count_sum = class_df["數量"].sum()
        total_area_sum = class_df["總面積(cm²)"].sum()
        max_area = class_df["最大面積(cm²)"].max()
        min_area = class_df["最小面積(cm²)"][class_df["最小面積(cm²)"] > 1].min()

        summary_data["類別"].append(class_name)
        summary_data["數量"].append(count_sum)
        summary_data["總面積(cm²)"].append(total_area_sum)
        summary_data["最大面積(cm²)"].append(max_area)
        summary_data["最小面積(cm²)"].append(min_area)

    summary_df = pd.DataFrame(summary_data)

    # 寫入 Excel
    with pd.ExcelWriter(output_excel_path) as writer:
        df.to_excel(writer, sheet_name="個別圖像數據", index=False)
        summary_df.to_excel(writer, sheet_name="總數據", index=False)

    print(f"所有數據已保存到 {output_excel_path}")
