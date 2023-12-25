import cv2
import os
import shutil

def process_video(file_path, frame_cut, start_point, resolution_label,total_frames_label, fps_label, completion_label, progress_bar):
    def inner_process_video():
        video_capture = cv2.VideoCapture(file_path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 總幀數
        
        fps = int(video_capture.get(cv2.CAP_PROP_FPS)) + 1  # 幀率
        
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))#尺寸
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))#尺寸
        
        resolution = (width, height)
        
        
        #resolution_label.config(text=f"影片解析度: {resolution}")
        #total_frames_label.config(text=f"總幀數: {total_frames}")
        #fps_label.config(text=f"幀率: {fps}")

        print("成功讀取影片")
        #print(f"總幀數: {total_frames}, 幀率: {fps}")

        progress_bar["maximum"] = total_frames  # 進度條最大值是total frames的數

        count = 0
        success, image = video_capture.read()

        output_folder = "frames_output"

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            print(f"刪除原有資料夾 {output_folder}")

        os.makedirs(output_folder)
        print(f"正在創建新資料夾 {output_folder}")

        num_images = 0

        while success:
            if count >= start_point * fps and (count - start_point * fps) % (frame_cut * fps ) == 0:
                image_file_name = os.path.join(output_folder, f"{num_images}.jpg")
                cv2.imwrite(image_file_name, image)
                num_images += 1

                progress_bar["value"] = count  # 更新進度條

            success, image = video_capture.read()
            count += 1

        video_capture.release()

        completion_label.config(text=f"共有 {num_images} 張圖片")
        print(f"完成，共 {num_images} 張圖片")

    inner_process_video()