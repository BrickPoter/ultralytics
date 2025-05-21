from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer
from custom_dataset import CustomChannelDataset

class CustomTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return CustomChannelDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

# 关键修改：添加if __name__ == '__main__'条件
if __name__ == '__main__':
    # 在Windows上添加多进程支持
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 使用自定义训练器
    model = YOLO('ultralytics/cfg/models/11/yolo11n_cbam.yaml')
    model.trainer = CustomTrainer
    results = model.train(data="mydata.yaml", epochs=100, device=0)