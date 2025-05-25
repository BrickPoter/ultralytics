from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
import torch
import sys
import os

# 添加tmp目录到系统路径，以便导入multi_branch_yolo模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multi_branch_yolo import MultiBranchYOLO11, create_multi_branch_yolo11


class MultiBranchDetectionModel(DetectionModel):
    """多分支YOLO检测模型"""
    
    def __init__(self, cfg="multi_branch_yolo.yaml", ch=3, nc=None, verbose=True):
        super(DetectionModel, self).__init__()
        # 从YAML加载配置
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            from ultralytics.utils.checks import check_yaml
            self.yaml_file = check_yaml(cfg)
            with open(self.yaml_file, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)
        
        # 设置参数
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        
        # 创建多分支YOLO模型
        self.model = create_multi_branch_yolo11(
            num_branches=self.yaml.get('num_branches', 3),
            num_classes=self.yaml['nc']
        )
        
        # 设置模型属性
        self.names = {i: f"{i}" for i in range(self.yaml['nc'])}
        self.stride = torch.Tensor([32, 16, 8])  # P5, P4, P3 strides
    
    def forward(self, x):
        """前向传播"""
        if isinstance(x, dict):
            # 训练模式
            return self.loss(x)
        # 推理模式
        return self.model(x)
    
    def loss(self, batch):
        """计算损失"""
        # 这里需要实现损失计算逻辑
        # 简化版本，实际应用中需要根据模型输出格式调整
        device = batch["img"].device
        loss = torch.zeros(3, device=device)  # box, cls, dfl losses
        
        # 获取模型预测
        preds = self.model(batch["img"])
        
        # 这里应该实现完整的损失计算逻辑
        # 为简化示例，我们返回一个基本的损失结构
        return {"loss": loss.sum(), "loss_items": loss, "preds": preds}


class MultiBranchDetectionTrainer(DetectionTrainer):
    """多分支YOLO检测训练器"""
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回多分支YOLO检测模型"""
        model = MultiBranchDetectionModel(cfg=cfg or "multi_branch_yolo.yaml", nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model
    
    def preprocess_batch(self, batch):
        """预处理批次数据"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch