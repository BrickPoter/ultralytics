# distill_train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import de_parallel
from pathlib import Path
import yaml

# ----------------------------
# 1. 配置参数
# ----------------------------
TEACHER_WEIGHTS = "yolov8s.pt"      # 教师模型路径
STUDENT_CFG = "yolov8n.yaml"        # 学生模型结构（可替换为 yolov8n_pruned.yaml）
DATA_YAML = "data.yaml"             # 数据集配置
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
LR = 0.001
TEMPERATURE = 4.0                   # 蒸馏温度
LAMBDA_KD = 5.0                     # 蒸馏损失权重
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 2. 修改 DetectionModel 以支持返回中间特征
# ----------------------------
class DistillDetectionModel(DetectionModel):
    def forward(self, x, augment=False, profile=False, visualize=False, return_features=False):
        if augment:
            return self._forward_augment(x)
        features = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = features[m.f] if isinstance(m.f, int) else [x if j == -1 else features[j] for j in m.f]
            x = m(x)
            features.append(x)
            if hasattr(m, 'i') and m.i in [6, 8, 10]:  # YOLOv8 的 Detect 层前的三个特征层索引（P3, P4, P5）
                if return_features:
                    pass
        if return_features:
            # 返回 backbone/neck 的最后三层特征（在 Detect 前）
            neck_features = [features[6], features[8], features[10]]  # 根据实际模型结构调整
            return x, neck_features
        return x

def load_distill_model(cfg, weights=None, device="cpu"):
    """加载支持特征输出的学生或教师模型"""
    model = DistillDetectionModel(cfg).to(device)
    if weights:
        ckpt = torch.load(weights, map_location=device)
        model.load_state_dict(ckpt["model"].float().state_dict(), strict=False)
    return model

# ----------------------------
# 3. 蒸馏训练主循环
# ----------------------------
def main():
    # 加载教师模型（冻结）
    teacher = load_distill_model(TEACHER_WEIGHTS, TEACHER_WEIGHTS, DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 加载学生模型
    student = load_distill_model(STUDENT_CFG, None, DEVICE)
    student.train()

    # 使用 YOLO 训练器获取 dataloader（复用 Ultralytics 的 pipeline）
    yolo_base = YOLO(STUDENT_CFG)
    trainer = yolo_base._smart_load("trainer")(overrides={
        "data": DATA_YAML,
        "imgsz": IMG_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        "epochs": EPOCHS,
        "lr0": LR,
        "name": "distill_yolov8"
    })
    trainer.model = student  # 替换模型
    trainer.train()  # 初始化 dataloader 等

    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=0.937, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 获取 dataloader
    train_loader = trainer.train_loader

    # 损失函数（使用 Ultralytics 的 ComputeLoss）
    compute_loss = trainer.get_validator().loss

    print(f"🚀 开始知识蒸馏训练 | Teacher: {Path(TEACHER_WEIGHTS).stem} | Student: {Path(STUDENT_CFG).stem}")

    for epoch in range(EPOCHS):
        student.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            imgs = batch["img"]

            # 教师推理（不更新梯度）
            with torch.no_grad():
                t_pred, t_feats = teacher(imgs, return_features=True)

            # 学生推理
            s_pred, s_feats = student(imgs, return_features=True)

            # 1. 原始检测损失
            loss, loss_items = compute_loss(s_pred, batch)

            # 2. 特征蒸馏损失（MSE）
            feat_loss = 0
            for ft, fs in zip(t_feats, s_feats):
                # 对齐通道（如果需要）
                if fs.shape[1] != ft.shape[1]:
                    proj = nn.Conv2d(fs.shape[1], ft.shape[1], kernel_size=1).to(DEVICE)
                    fs = proj(fs)
                feat_loss += F.mse_loss(fs, ft)

            # 3. 输出蒸馏损失（可选，YOLO 检测头较复杂，此处简化为特征蒸馏为主）
            total_loss = loss + LAMBDA_KD * feat_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch}/{EPOCHS}, Iter {i}, Loss: {total_loss:.4f}, FeatKD: {feat_loss:.4f}")

        scheduler.step()
        print(f"✅ Epoch {epoch} finished. Avg Loss: {epoch_loss / len(train_loader):.4f}")

        # 保存检查点
        ckpt = {
            "model": de_parallel(student).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(ckpt, f"runs/distill_yolov8/weights/student_epoch{epoch}.pt")

    print("🎉 蒸馏训练完成！")

if __name__ == "__main__":
    main()