from ultralytics.data.dataset import ClassificationDataset
from ultralytics.data.augment import classify_augmentations
import cv2
import numpy as np
from PIL import Image

class CustomChannelDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=""):
        super().__init__(root, args, augment, prefix)
        
        # 保存原始的transforms用于后续处理
        self.original_transforms = self.torch_transforms
    
    def __getitem__(self, i):
        # 获取原始图像
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        
        # 加载图像
        if self.cache_ram:
            if im is None:
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:
            im = cv2.imread(f)  # BGR格式
        
        # 执行通道替换处理
        im = self.custom_channel_processing(im)
        
        # 转换为PIL图像并应用标准增强
        im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.original_transforms(im_pil)
        
        return {"img": sample, "cls": j}
    
    # def custom_channel_processing(self, image):
    #     # 分离通道 (BGR格式)
    #     b, g, r = cv2.split(image)
        
    #     # 对G通道进行多阈值分割
    #     # 使用Otsu算法自动确定阈值
    #     _, g_thresh = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    #     # 对R通道进行边缘检测
    #     r_edges = cv2.Canny(r, 100, 200)
        
    #     # 合并处理后的通道
    #     processed_image = cv2.merge([b, g_thresh, r_edges])
        
    #     return processed_image
    def custom_channel_processing(self, image):
    # 分离通道
        b, g, r = cv2.split(image)
        
        # 多阈值分割 - 使用多个阈值
        ret1, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2, th2 = cv2.threshold(g, ret1-30, 255, cv2.THRESH_BINARY)
        ret3, th3 = cv2.threshold(g, ret1+30, 255, cv2.THRESH_BINARY)
        g_multi_thresh = cv2.bitwise_or(cv2.bitwise_or(th1, th2), th3)
        
        # 高级边缘检测 - 使用Sobel算子
        sobelx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=3)
        r_edges = cv2.magnitude(sobelx, sobely)
        r_edges = cv2.normalize(r_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 合并处理后的通道
        processed_image = cv2.merge([b, g_multi_thresh, r_edges])
        
        return processed_image