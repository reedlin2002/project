import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2

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
        cap.release()
        cv2.destroyAllWindows()

def get_video_info():
    file_path = file_var.get()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        
        resolution_label.config(text=f"Resolution: {resolution[0]}x{resolution[1]}")
        total_frames_label.config(text=f"Total Frames: {total_frames}")
        fps_label.config(text=f"FPS: {fps}")

def cut_video():
    file_path = file_var.get()
    start_point = start_point_entry.get()
    frame_interval = frame_interval_entry.get()
    
    if not file_path:
        completion_label.config(text="Please select a video.")
        return
    
    if not start_point or not frame_interval or not start_point.isdigit() or not frame_interval.isdigit():
        completion_label.config(text="Please enter valid cut parameters.")
        return
    
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_point = int(start_point)
    frame_interval = int(frame_interval)

    if start_point >= total_frames:
        completion_label.config(text="Start point exceeds total frames.")
        return

    if frame_interval <= 0:
        completion_label.config(text="Frame interval should be greater than zero.")
        return

    if start_point + frame_interval > total_frames:
        completion_label.config(text="Frame interval exceeds total frames.")
        return

    cut_frames = (total_frames - start_point) // frame_interval
    completion_label.config(text=f"Cut frames: {cut_frames}")

root = tk.Tk()
root.title("Video Player & Cutter")

file_var = tk.StringVar()

label = tk.Label(root, text="Select Video:")
label.grid(row=0, column=0, padx=10, pady=10)

file_entry = tk.Entry(root, textvariable=file_var, state='readonly', width=50)
file_entry.grid(row=0, column=1, padx=10, pady=10)

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        file_var.set(file_path)

browse_button = tk.Button(root, text="Browse", command=open_file)
browse_button.grid(row=0, column=2, padx=10, pady=10)

play_button = tk.Button(root, text="Play Video", command=play_video)
play_button.grid(row=1, column=0, padx=10, pady=10)

info_button = tk.Button(root, text="Get Video Info", command=get_video_info)
info_button.grid(row=1, column=1, padx=10, pady=10)

label = tk.Label(root, text="Start Point (seconds):")
label.grid(row=2, column=0, padx=10, pady=10)

start_point_entry = tk.Entry(root)
start_point_entry.grid(row=2, column=1, padx=10, pady=10)

label = tk.Label(root, text="Frame Interval:")
label.grid(row=3, column=0, padx=10, pady=10)

frame_interval_entry = tk.Entry(root)
frame_interval_entry.grid(row=3, column=1, padx=10, pady=10)

cut_button = tk.Button(root, text="Cut Video", command=cut_video)
cut_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

resolution_label = tk.Label(root, text="")
resolution_label.grid(row=5, columnspan=2, padx=10, pady=5)

total_frames_label = tk.Label(root, text="")
total_frames_label.grid(row=6, columnspan=2, padx=10, pady=5)

fps_label = tk.Label(root, text="")
fps_label.grid(row=7, columnspan=2, padx=10, pady=5)

completion_label = tk.Label(root, text="")
completion_label.grid(row=8, columnspan=2, padx=10, pady=5)

root.mainloop()
