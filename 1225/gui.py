import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from video_frames import process_video
import cv2
import threading
def gui():
    def open_file():
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if file_path:
            file_var.set(file_path)
        
    def start_video_processing():
        file_path = file_var.get()
        frame_cut = entry.get()
        start_point = start_point_entry.get()
        completion_label.config(text="切割中...")
        
        if not file_path:
            # 尚未選擇影片
            completion_label.config(text="請選擇影片")
            return
    
        if not start_point or not frame_cut or not start_point.isdigit() or not frame_cut.isdigit():
            # 尚未自訂切割方式
            completion_label.config(text="請選擇決定方式")
            return
        

        
        video_capture = cv2.VideoCapture(file_path)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS)) +1
        total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start_point = int(start_point)
        
        frame_cut = int(frame_cut) * fps  #改成使用秒數判斷
        
        
        if start_point >= total_frame:
            completion_label.config(text="起點超過總幀數.")
            return

        if frame_cut <= 0:
            completion_label.config(text="每幀間隔需大於0.")
            return

        if start_point + frame_cut > total_frame:
            completion_label.config(text="每幀間隔超過總幀數.")
            return
        
        threading.Thread(target=process_video, args=(
            file_var.get(), int(entry.get()), int(start_point_entry.get()), resolution_label, total_frames_label, fps_label, completion_label, progress_bar
            )).start()
    #====顯示瀏覽影片視窗====
    def play_video():
        file_path = file_var.get()
        if file_path:
            cap = cv2.VideoCapture(file_path)
            cv2.namedWindow('Video Player', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video Player', 1080, 720)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cv2.imshow('Video Player', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
    
                # 處理視窗關閉
                if cv2.getWindowProperty('Video Player', cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        #====顯示影片資訊====
    def get_video_info():
        file_path = file_var.get()
        if file_path:
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) + 1 #29.99666 
            resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            
            seconds = int(total_frames/fps)
            
            total_frames_label.config(text=f"影片時間: {seconds} (seconds)")
            resolution_label.config(text=f"影片解析度: {resolution}")
            fps_label.config(text=f"幀率: {fps}")
                    
    root = tk.Tk()
    root.title("影格拼接程式")
    
    file_var = tk.StringVar()  # 檔案路徑
    # ===選擇影片===
    label = tk.Label(root, text="選擇影片：")
    label.grid(row=0, column=0, padx=(10, 5), pady=10, sticky='w')
    
    file_entry = tk.Entry(root, textvariable=file_var, state='readonly', width=50)
    file_entry.grid(row=0, column=1, padx=10, pady=10)
    
    #瀏覽button(找影片)
    browse_button = tk.Button(root, text="瀏覽", command=open_file)
    browse_button.grid(row=0, column=2, padx=10, pady=10)
    
    #播放選擇的影片(button)
    play_button = tk.Button(root, text="播放影片", command=play_video)
    play_button.grid(row=1, column=1, padx=10, pady=10, sticky='w')
    
    # ===決定起點、幾幀切===
    #起點
    label = tk.Label(root, text="   起點 (seconds)")
    label.grid(row=2, column=1, padx=10, pady=10, sticky='w')

    start_point_entry = tk.Entry(root)
    start_point_entry.grid(row=2, column=1, padx=10, pady=10)
    
    #要用幾幀切
    label = tk.Label(root, text="影格間距 (seconds)：")
    label.grid(row=3, column=1, padx=5, pady=10, sticky='w')
    
    entry = tk.Entry(root)
    entry.grid(row=3, column=1, padx=10, pady=10)
    
    # ===顯示資訊的===
    info_button = tk.Button(root, text="獲取影片資訊", command=get_video_info)
    info_button.grid(row=1, column=1, padx=10, pady=10)
    
    #總秒數
    total_frames_label = tk.Label(root, text="")
    total_frames_label.grid(row=4, columnspan=3, padx=10, pady=5)
    
    # fps
    fps_label = tk.Label(root, text="")
    fps_label.grid(row=5, columnspan=3, padx=10, pady=5)
    # 解析度
    resolution_label = tk.Label(root, text="")
    resolution_label.grid(row=6, columnspan=3, padx=10, pady=5)
    
    # 完成有幾張
    completion_label = tk.Label(root, text="")
    completion_label.grid(row=7, columnspan=3, padx=10, pady=5)
    
    #=======開始=====
    # 開始切
    confirm_button = tk.Button(root, text="確定", command=start_video_processing)
    confirm_button.grid(row=9, columnspan=3, padx=10, pady=10)
    
    # 進度條
    progress_bar = ttk.Progressbar(root, length=300)
    progress_bar.grid(row=8, columnspan=3, padx=10, pady=10)
    
    root.mainloop()