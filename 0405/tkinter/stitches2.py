import cv2
import os
from gui import gui

def resize_images(images, scale_percent=100):  # 將圖片縮小為原來的 scale_percent/100
    resized_images = []
    for img in images:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
    return resized_images

def stitch_images_in_folder(folder_path):# 獲取資料夾中所有的 .jpg 檔案路徑
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    
    # 讀取並縮小所有圖片
    images = [cv2.imread(path) for path in image_paths]
    resized_images = resize_images(images)
    stitch(resized_images)
    
def stitch(image):
    print('圖像拼接中...')
    # 圖像拼接
    stitcher = cv2.Stitcher_create()
    status, scans = stitcher.stitch(image)
    # 黑邊處理...
    if status == cv2.Stitcher_OK:
        print('黑邊處理...')
        stitched = cv2.copyMakeBorder(scans, 20, 20, 20, 20, cv2.BORDER_CONSTANT, (0, 0, 0, 0))
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        #cv2.imwrite('image.jpg', stitched)
    
        # 尋找最大輪廓
        max_contour = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(thresh, (x, y), (x + w, y + h), 255, 2)
    
        # 提取最大輪廓並裁減
        stitched = stitched[y:y + h, x:x + w]
        print('完成')
        cv2.imwrite('stitched.jpg', stitched)
    else:
        print('圖像匹配的特徵點不足')
        

if __name__ == "__main__":
    gui()
    folder_path = "frames_output"  # 使用當前資料夾的路徑
    stitch_images_in_folder(folder_path)
