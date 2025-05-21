from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.detect import DetectionPredictor
from custom_dataset import CustomChannelDataset
import cv2
import numpy as np


class CustomPredictor(DetectionPredictor):

    def __init__(self, args=None, _callbacks=None):
        # 完整继承父类初始化参数
        super().__init__(args, _callbacks)
        self.args.custom_preprocess = args.get('custom_preprocess', False) if args else False

    def preprocess(self, im):
        if self.args.custom_preprocess:
            im = CustomChannelDataset.custom_channel_processing(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return super().preprocess(im)


# 修改模型类关联方式
model = YOLO(model=r'E:\Project_Files\ultralytics\runs\detect\train8\weights\best.pt', task='detect')
model.predictor = CustomPredictor(args=model.args)  # 传递模型参数

# 使用示例
if __name__ == '__main__':
    model = YOLO(r'E:\Project_Files\ultralytics\runs\detect\train8\weights\best.pt')
    model.predictor = CustomPredictor
    results = model.predict(
        source=r'E:\Project_Files\ImageProcess\welds',
        custom_preprocess=True,  # 启用自定义预处理
        show=True,
        save=True,
        conf=0.5
    )