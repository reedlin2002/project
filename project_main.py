import os
import cv2
import numpy as np
import json
import pandas as pd
import tkinter as tk
from tkinter import ttk

from gui import gui  # 匯入 GUI
from predict_segm import setup_predictor, classify_and_save, save_excel  # 匯入模型預測器和分類儲存函數
from stitching import stitch_images_in_folder  # 匯入圖像拼接函數
# 匯入 predict_withoutoverlap 模組
from predict_withoutoverlap import process_images as process_images_without_overlap, setup_predictor as setup_predictor_without_overlap

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 圖像切割函數
def split_image(image, output_folder, base_filename, width=640, height=480):
    h, w = image.shape[:2]
    sub_images = []
    index = 0
    
    # 進行切割
    for y in range(0, h - height + 1, height):
        for x in range(0, w - width + 1, width):
            sub_image = image[y:y + height, x:x + width]
            sub_image_path = os.path.join(output_folder, f"{base_filename}_{index}.jpg")
            cv2.imwrite(sub_image_path, sub_image)
            sub_images.append(sub_image_path)
            index += 1
    
    return sub_images

# 計算 bbox 中心點
def calculate_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2  # 浮點數計算
    center_y = (y_min + y_max) / 2  # 浮點數計算
    return center_x, center_y

# 保存中心點數據到 JSON
def save_json_center_points(center_points, output_json_path):
    with open(output_json_path, 'w') as json_file:
        json.dump(center_points, json_file, indent=4)

# 預測圖像資料夾中的圖像
def predict_images_in_folder(folder_path, predictor, output_excel_path, temp_folder):
    image_data = {}
    json_data = {}

    # 取得所有 JPG 圖片
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]

    # 對每張圖像進行預測
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if image_name not in image_data:
            image_data[image_name] = {
                "圖像編號": image_name,
                "類別": [],
                "數量": [],
                "總面積(cm²)": [],
                "最大面積(cm²)": [],
                "最小面積(cm²)": []
            }

        # 創建臨時資料夾保存切割圖像
        image_split_folder = os.path.join(temp_folder, image_name)
        os.makedirs(image_split_folder, exist_ok=True)

        # 切割圖像
        sub_image_paths = split_image(image, image_split_folder, image_name)

        json_data[image_name] = []

        for j, sub_image_path in enumerate(sub_image_paths):
            sub_image = cv2.imread(sub_image_path)
            output_image_path = os.path.join(image_split_folder, f"output_{j}.bmp")
            output_json_path = os.path.join(image_split_folder, f"category_info_{j}.json")

            instances = classify_and_save(sub_image, predictor, output_image_path, output_json_path, image_name, image_data)

            # 如果有預測結果，計算 bbox 和中心點
            if instances is not None and hasattr(instances, 'pred_boxes'):
                sub_h, sub_w = sub_image.shape[:2]
                column = j % (image.shape[1] // 640)
                row = j // (image.shape[1] // 640)
                for bbox in instances.pred_boxes.tensor.numpy():
                    x_min, y_min, x_max, y_max = bbox
                    
                    x_min_full = max(0, x_min + column * 640)
                    y_min_full = max(0, y_min + row * 480)
                    x_max_full = min(image.shape[1], x_max + column * 640)
                    y_max_full = min(image.shape[0], y_max + row * 480)

                    center_x, center_y = calculate_center((x_min_full, y_min_full, x_max_full, y_max_full))
                    json_data[image_name].append({
                        "bbox": [x_min_full, y_min_full, x_max_full, y_max_full],
                        "center": [center_x, center_y]
                    })

        save_json_center_points(json_data, os.path.join(folder_path, f"centers_{image_name}.json"))

        # 拼接小圖回原圖
        reconstructed_image = reconstruct_image(image_split_folder, sub_image_paths, image.shape[1], image.shape[0])
        reconstructed_image_path = os.path.join(folder_path, f"reconstructed_{image_name}.bmp")
        cv2.imwrite(reconstructed_image_path, reconstructed_image)

    save_excel(image_data, output_excel_path)

    final_json_path = os.path.join(folder_path, "all_centers.json")
    save_json_center_points(json_data, final_json_path)

# 拼接回原圖的函數
def reconstruct_image(image_split_folder, sub_image_paths, original_width, original_height, sub_image_width=640, sub_image_height=480):
    num_columns = original_width // sub_image_width
    num_rows = original_height // sub_image_height

    reconstructed_image = np.zeros((sub_image_height * num_rows, sub_image_width * num_columns, 3), dtype=np.uint8)

    for i, sub_image_path in enumerate(sub_image_paths):
        sub_image_path = os.path.join(image_split_folder, f"output_{i}.bmp")
        sub_image = cv2.imread(sub_image_path)
        row = i // num_columns
        col = i % num_columns

        y_start = row * sub_image_height
        y_end = y_start + sub_image_height
        x_start = col * sub_image_width
        x_end = x_start + sub_image_width

        reconstructed_image[y_start:y_end, x_start:x_end] = sub_image

    return reconstructed_image

# 顯示 Excel 數據
def show_excel_data(excel_file):
    root = tk.Tk()
    root.title(f"辨識成果資訊 : {os.path.basename(excel_file)}")

    df_summary = pd.read_excel(excel_file, sheet_name="總數據")
    df_individual = pd.read_excel(excel_file, sheet_name="個別圖像數據")

    df_summary["總面積(cm²)"] = df_summary["總面積(cm²)"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else x)
    df_summary["最大面積(cm²)"] = df_summary["最大面積(cm²)"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else x)
    df_summary["最小面積(cm²)"] = df_summary["最小面積(cm²)"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else x)

    df_individual["總面積(cm²)"] = df_individual["總面積(cm²)"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else x)
    df_individual["最大面積(cm²)"] = df_individual["最大面積(cm²)"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else x)
    df_individual["最小面積(cm²)"] = df_individual["最小面積(cm²)"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else x)

    tab_control = ttk.Notebook(root)

    summary_tab = ttk.Frame(tab_control)
    tab_control.add(summary_tab, text='總數據')

    tree_summary = ttk.Treeview(summary_tab, columns=list(df_summary.columns), show="headings")
    for col in df_summary.columns:
        tree_summary.heading(col, text=col)

    for _, row in df_summary.iterrows():
        formatted_row = [f"{x}" if isinstance(x, str) else x for x in row]
        tree_summary.insert("", "end", values=formatted_row)

    tree_summary.pack(expand=True, fill=tk.BOTH)
    scroll_bar_summary = ttk.Scrollbar(summary_tab, orient="vertical", command=tree_summary.yview)
    scroll_bar_summary.pack(side=tk.RIGHT, fill=tk.Y)
    tree_summary.config(yscroll=scroll_bar_summary.set)

    individual_tab = ttk.Frame(tab_control)
    tab_control.add(individual_tab, text='個別圖像數據')

    tree_individual = ttk.Treeview(individual_tab, columns=list(df_individual.columns), show="headings")
    for col in df_individual.columns:
        tree_individual.heading(col, text=col)

    for _, row in df_individual.iterrows():
        formatted_row = [f"{x}" if isinstance(x, str) else x for x in row]
        tree_individual.insert("", "end", values=formatted_row)

    tree_individual.pack(expand=True, fill=tk.BOTH)
    scroll_bar_individual = ttk.Scrollbar(individual_tab, orient="vertical", command=tree_individual.yview)
    scroll_bar_individual.pack(side=tk.RIGHT, fill=tk.Y)
    tree_individual.config(yscroll=scroll_bar_individual.set)

    tab_control.pack(expand=True, fill=tk.BOTH)
    root.mainloop()


if __name__ == "__main__":
    # 啟動 GUI，並等待使用者操作 GUI
    gui()

    # 當 GUI 關閉或特定條件達成後，執行以下流程
    folder_path = "frames_output"
    temp_folder = "temp_output"
    withoutoverlap_folder = "withoutoverlap"
    #output_excel_path = "result.xlsx"
    output_excel_path = "output_data.xlsx"

    # 第一步：加載模型並執行 predict_images_in_folder
    print('加載模型預測器...')
    predictor = setup_predictor(
        #config_path=r"檔案位置",
        #weights_path=r"檔案位置"
        config_path=r"檔案位置",
        weights_path=r"檔案位置"
    )

    # 執行圖像拼接和預測
    stitch_images_in_folder(folder_path)
    
    print('圖像辨識中...')
    predict_images_in_folder(folder_path, predictor, output_excel_path, temp_folder)

    # 確保 `predict_images_in_folder` 完成之後，再進行 withoutoverlap 處理
    print('處理 重疊區域(withoutoverlap)的數量問題...')
    predictor_without_overlap = setup_predictor_without_overlap(
        config_path=r"檔案位置",
        weights_path=r"檔案位置"
    )
    
    # 第二步：處理 withoutoverlap 資料夾中的圖像
    process_images_without_overlap(withoutoverlap_folder, output_excel_path, predictor_without_overlap)

   # 顯示 Excel 數據
    show_excel_data(output_excel_path)