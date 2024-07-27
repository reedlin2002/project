import os
import cv2
from gui import gui  # 匯入 GUI 
from predict_segm import setup_predictor, classify_and_save  # 匯入模型預測器和分類儲存函數
from stitching import stitch_images_in_folder  # 匯入圖像拼接函數

def predict_images_in_folder(folder_path, predictor):
    """
    對指定資料夾中的每張圖片進行預測，並將預測結果保存為圖片和 JSON 文件。
    
    :param folder_path: 圖片所在資料夾的路徑
    :param predictor: 已配置的模型預測器
    """
    # 獲取所有 JPEG 圖片的路徑
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    
    # 對每張圖片進行預測
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)  # 讀取圖片
        output_image_path = f"output_{i}.bmp"  # 設置預測結果圖像的保存路徑
        output_json_path = f"category_info_{i}.json"  # 設置類別信息的 JSON 文件保存路徑
        
        # 使用預測器對圖片進行預測並保存結果
        classify_and_save(image, predictor, output_image_path, output_json_path)

if __name__ == "__main__":
    gui()  # 啟動 GUI
    
    folder_path = "frames_output"  # 設置圖像所在資料夾的路徑

    # 加載模型預測器
    print('加載模型預測器...')
    predictor = setup_predictor(
        config_path=r"C:\Users\JerryLin\Desktop\test\0721\beach_V_39_eSE_FPN_ms_3x.yaml",  # 模型配置文件路徑
        weights_path=r"C:\Users\JerryLin\Desktop\test\0721\model_final.pth"  # 模型權重文件路徑
    )

    # 對每張圖片進行預測
    predict_images_in_folder(folder_path, predictor)
    
    # 進行圖像拼接
    stitch_images_in_folder(folder_path)  # 對指定資料夾中的圖片進行拼接
