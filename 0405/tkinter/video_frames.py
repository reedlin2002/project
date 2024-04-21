import cv2
import os
import shutil

def process_video(file_path, frame_cut, start_point, end_point, resolution_label, total_frames_label, fps_label, completion_label, progress_bar):
    def inner_process_video():
        video_capture = cv2.VideoCapture(file_path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 總幀數
        
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))  # 幀率

        start_frame = int(start_point * fps)  # 起始幀數
        end_frame = int(end_point * fps)  # 結束幀數

        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 尺寸
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 尺寸

        resolution = (width, height)

        print("成功讀取影片")

        progress_bar["maximum"] = end_frame - start_frame-1  # 進度條最大值是剪輯後的幀數

        count = 0
        success, image = video_capture.read()

        output_folder = "frames_output"

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            print(f"刪除原有資料夾 {output_folder}")

        os.makedirs(output_folder)
        print(f"正在創建新資料夾 {output_folder}")

        num_images = 0

        while success and count <= end_frame:
            if count >= start_frame and (count - start_frame) % frame_cut == 0 and start_frame - end_frame <=count:
                image_file_name = os.path.join(output_folder, f"{num_images}.jpg")
                cv2.imwrite(image_file_name, image)
                num_images += 1

                progress_bar["value"] = count - start_frame  # 更新進度條

            success, image = video_capture.read()
            count += 1

        video_capture.release()

        completion_label.config(text=f"共有 {num_images} 張圖片")
        print(f"完成，共 {num_images} 張圖片")

    inner_process_video()
