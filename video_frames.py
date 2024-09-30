import cv2
import os
import shutil

def process_video(file_path, frame_cut, start_point, end_point, resolution_label, total_frames_label, fps_label, completion_label, progress_bar):
    """
    處理影片，將指定範圍內的幀按間隔儲存為圖片，並更新進度條和完成標籤。
    #影片分割
    :param file_path: 影片檔案路徑
    :param frame_cut: 每幀間隔（例如每 10 幀儲存一張圖片）
    :param start_point: 起始點（以秒為單位）
    :param end_point: 終點（以秒為單位）
    :param resolution_label: 顯示解析度的標籤
    :param total_frames_label: 顯示總幀數的標籤
    :param fps_label: 顯示幀率的標籤
    :param completion_label: 顯示完成信息的標籤
    :param progress_bar: 進度條
    """
    def inner_process_video():
        # 開啟影片檔案
        video_capture = cv2.VideoCapture(file_path)
        
        # 獲取影片的總幀數
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # 獲取影片的幀率（每秒幾幀）
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))

        # 計算起始和結束幀數
        start_frame = int(start_point * fps)
        end_frame = int(end_point * fps)

        # 獲取影片的尺寸（寬度和高度）
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = (width, height)

        print("成功讀取影片")

        # 設定進度條的最大值為剪輯後的幀數
        progress_bar["maximum"] = end_frame - start_frame - 1

        count = 0
        success, image = video_capture.read()  # 讀取第一幀

        output_folder = "frames_output"

        # 如果已經存在輸出資料夾，則刪除舊資料夾
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            print(f"刪除原有資料夾 {output_folder}")

        # 創建新資料夾
        os.makedirs(output_folder)
        print(f"正在創建新資料夾 {output_folder}")

        num_images = 0  # 計數器，用於跟蹤儲存的圖片數量

        # 循環處理每一幀
        while success and count <= end_frame:
            # 如果當前幀在起始點和結束點之間，並且符合 frame_cut 的間隔條件，則儲存圖片
            if (count >= start_frame and (count - start_frame) % frame_cut == 0) or count == end_frame:
                image_file_name = os.path.join(output_folder, f"{num_images:06d}.jpg")
                cv2.imwrite(image_file_name, image)  # 儲存圖片
                num_images += 1

                # 更新進度條的值
                progress_bar["value"] = count - start_frame

            # 讀取下一幀
            success, image = video_capture.read()
            count += 1

        # 釋放影片資源
        video_capture.release()

        # 更新完成標籤顯示圖片數量
        completion_label.config(text=f"共有 {num_images} 張圖片")
        print(f"完成，共 {num_images} 張圖片")

    # 呼叫內部處理函數
    inner_process_video()
