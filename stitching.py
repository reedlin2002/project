import cv2
import os

def resize_images(images, scale_percent=100):
    """
    將圖像列表按指定的縮放百分比進行縮放。
    #影像拼接
    :param images: 圖像的列表，每個圖像是一個 NumPy 陣列
    :param scale_percent: 縮放百分比，預設為 50%
    :return: 縮放後的圖像列表
    """
    resized_images = []  # 用來儲存縮放後的圖像
    for img in images:
        # 計算新的尺寸
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # 縮放圖像
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        
        # 將縮放後的圖像加入列表
        resized_images.append(resized)
        
    return resized_images

def stitch(images):
    """
    對一組圖像進行拼接，並處理黑邊。
    
    :param images: 要拼接的圖像列表，每個圖像是一個 NumPy 陣列
    """
    print('圖像拼接中...')
    
    # 創建一個 Stitcher 物件，用於圖像拼接
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    
    # 使用 Stitcher 物件進行拼接
    status, stitched = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        print('黑邊處理...')
        
        # 對拼接後的圖像加上黑邊
        stitched = cv2.copyMakeBorder(stitched, 20, 20, 20, 20, cv2.BORDER_CONSTANT, (0, 0, 0, 0))
        
        # 將拼接後的圖像轉為灰階
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        
        # 對灰階圖像進行二值化處理
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        
        # 查找二值化圖像中的輪廓
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # 找到面積最大的輪廓
        max_contour = max(cnts, key=cv2.contourArea)
        
        # 計算最大輪廓的邊界框
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 在二值化圖像上畫出最大輪廓的邊界框
        cv2.rectangle(thresh, (x, y), (x + w, y + h), 255, 2)

        # 根據邊界框裁剪拼接後的圖像
        stitched = stitched[y:y + h, x:x + w]
        
        print('拼接完成')
        
        # 儲存拼接後的圖像
        cv2.imwrite('stitched.jpg', stitched)
    else:
        # 如果圖像匹配失敗，輸出錯誤信息
        print('拼接失敗，錯誤')

def stitch_images_in_folder(folder_path):
    """
    從指定資料夾中讀取所有圖像，進行縮放，然後拼接這些圖像。
    
    :param folder_path: 包含圖像的資料夾路徑
    """
    # 獲取資料夾中所有的圖像路徑
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    
    # 讀取所有圖像
    images = [cv2.imread(path) for path in image_paths]
    
    # 縮放圖像
    resized_images = resize_images(images)
    
    # 拼接縮放後的圖像
    stitch(resized_images)
