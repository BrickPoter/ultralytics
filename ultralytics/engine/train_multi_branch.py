from ultralytics import YOLO
from multi_branch_trainer import MultiBranchDetectionTrainer
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 确保在Windows上添加多进程支持
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 创建YOLO模型
    model = YOLO("multi_branch_yolo.yaml")
    
    # 设置自定义训练器
    results = model.train(
        data="coco8.yaml",  # 使用coco8数据集进行测试，您可以替换为自己的数据集
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,  # 使用GPU，如果没有GPU可以设置为'cpu'
        trainer=MultiBranchDetectionTrainer
    )
    
    # 打印训练结果
    print(f"训练完成，最佳模型保存在: {results.best}")