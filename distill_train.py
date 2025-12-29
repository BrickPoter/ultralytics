import argparse
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import copy

from ultralytics.data.utils import check_det_dataset
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.cfg import get_cfg, DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionValidator


class KDAdaptors(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self._device = device

    def set_device(self, device):
        self._device = device

    def get(self, in_c, out_c):
        for m in self.layers:
            if isinstance(m, nn.Conv2d) and m.in_channels == in_c and m.out_channels == out_c:
                return m
        m = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        if self._device is not None:
            m = m.to(self._device)
        self.layers.append(m)
        return m


def get_head_indices(model):
    m = model.model[-1]
    if isinstance(m, Detect):
        return m.f if isinstance(m.f, list) else [m.f]
    return []


def hook_features(model, indices):
    outputs = {}
    hooks = []
    modules = {i: mod for i, mod in enumerate(model.model)}
    for i in indices:
        if i in modules:
            def _hook(idx):
                def fn(module, inp, out):
                    outputs[idx] = out
                return fn
            h = modules[i].register_forward_hook(_hook(i))
            hooks.append(h)
    return outputs, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def split_head_logits(x, nc, reg_max):
    box = x[:, : reg_max * 4]
    cls = x[:, reg_max * 4 : reg_max * 4 + nc]
    return box, cls


def distill_train(
    teacher_path,
    student_path,
    data_yaml,
    kd_mode="all",
    epochs=10,
    batch_size=16,
    imgsz=640,
    lr=1e-3,
    device_str="",
    alpha=1.0,
    beta=1.0,
    gamma=0.5,
    temperature=1.0,
    save_path="distilled_student.pt",
    workers=4,
):
    device = select_device(device_str, batch_size)
    data = check_det_dataset(data_yaml)
    cfg = get_cfg(
        DEFAULT_CFG,
        {
            "imgsz": imgsz,
            "rect": False,
            "cache": None,
            "single_cls": False,
            "task": "detect",
            "classes": None,
            "fraction": 1.0,
            "workers": workers,
        },
    )

    if str(student_path).endswith(".pt"):
        w_s, _ = attempt_load_one_weight(student_path)
        student = DetectionModel(cfg=w_s.yaml, nc=data["nc"], ch=data["channels"], verbose=False)
        student.load(w_s)
    else:
        student = DetectionModel(cfg=student_path, nc=data["nc"], ch=data["channels"], verbose=False)
    student = student.to(device)
    student.args = cfg
    student.names = data["names"]
    student.nc = data["nc"]

    w_t, _ = attempt_load_one_weight(teacher_path)
    teacher = DetectionModel(cfg=w_t.yaml, nc=data["nc"], ch=data["channels"], verbose=False)
    teacher.load(w_t)
    teacher = teacher.to(device)
    teacher.eval()
    teacher.args = cfg
    teacher.names = data["names"]
    teacher.nc = data["nc"]

    train_ds = build_yolo_dataset(cfg, data["train"], batch_size, data, mode="train", rect=False, stride=32)
    train_loader: DataLoader = build_dataloader(train_ds, batch_size, workers, shuffle=True, rank=-1)
    val_path = data.get("val") or data.get("test")
    val_ds = build_yolo_dataset(cfg, val_path, batch_size * 2, data, mode="val", rect=True, stride=32)
    val_loader: DataLoader = build_dataloader(val_ds, batch_size * 2, workers * 2, shuffle=False, rank=-1)
    val_args = copy(cfg)
    setattr(val_args, "model", student_path)
    setattr(val_args, "data", data_yaml)
    setattr(val_args, "split", "val")
    validator = DetectionValidator(dataloader=val_loader, save_dir=None, args=val_args)

    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=lr)
    adaptors = KDAdaptors(device=device).to(device)
    optimizer.add_param_group({"params": adaptors.parameters(), "lr": lr})

    def add_missing_params_to_optimizer(opt, module, lr_value):
        existing = set(id(p) for g in opt.param_groups for p in g["params"])
        new_params = [p for p in module.parameters() if id(p) not in existing]
        if new_params:
            opt.add_param_group({"params": new_params, "lr": lr_value})

    reg_max = int(getattr(student.model[-1], "reg_max", 16))
    nc = student.yaml["nc"]
    kd_mid_weight = alpha if kd_mode in {"mid", "all"} else 0.0
    kd_out_weight_cls = beta if kd_mode in {"out", "all"} else 0.0
    kd_out_weight_box = gamma if kd_mode in {"out", "all"} else 0.0

    student.train()
    indices_s = get_head_indices(student)
    indices_t = get_head_indices(teacher)
    iters = math.ceil(len(train_ds) / batch_size) * epochs
    kldiv = nn.KLDivLoss(reduction="batchmean")

    if getattr(student, "criterion", None) is None:
        student.criterion = student.init_criterion()

    for epoch in range(epochs):
        for batch in train_loader:
            img = batch["img"].to(device).float() / 255.0
            batch["img"] = img
            with torch.no_grad():
                feats_t, hooks_t = hook_features(teacher, indices_t)
                pred_t = teacher.predict(img)
                remove_hooks(hooks_t)
                raw_t = pred_t[1]

            feats_s, hooks_s = hook_features(student, indices_s)
            preds_s = student.forward(img)
            remove_hooks(hooks_s)

            det_out = student.criterion(preds_s, batch)
            if isinstance(det_out, tuple):
                det_vec, loss_items = det_out
            else:
                det_vec, loss_items = det_out, None
            loss_det = det_vec.sum() if torch.is_tensor(det_vec) and det_vec.ndim > 0 else det_vec

            loss_kd_mid = torch.tensor(0.0, device=device)
            if kd_mid_weight > 0.0 and feats_s and feats_t:
                keys_s = sorted(feats_s.keys())
                keys_t = sorted(feats_t.keys())
                n = min(len(keys_s), len(keys_t))
                for i in range(n):
                    fs = feats_s[keys_s[i]]
                    ft = feats_t[keys_t[i]].detach()
                    if fs.shape[-2:] != ft.shape[-2:]:
                        fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
                    if fs.shape[1] != ft.shape[1]:
                        proj = adaptors.get(fs.shape[1], ft.shape[1])
                        add_missing_params_to_optimizer(optimizer, adaptors, lr)
                        fs = proj(fs)
                    loss_kd_mid = loss_kd_mid + F.mse_loss(fs, ft)
                loss_kd_mid = loss_kd_mid / max(n, 1)

            loss_kd_out_cls = torch.tensor(0.0, device=device)
            loss_kd_out_box = torch.tensor(0.0, device=device)
            if kd_out_weight_cls > 0.0 or kd_out_weight_box > 0.0:
                raw_s = preds_s
                nls = min(len(raw_s), len(raw_t))
                for i in range(nls):
                    rs = raw_s[i]
                    rt = raw_t[i].detach()
                    if rs.shape[-2:] != rt.shape[-2:]:
                        rs = F.interpolate(rs, size=rt.shape[-2:], mode="nearest")
                    bs = rs.shape[0]
                    rs_flat = rs.view(bs, rs.shape[1], -1)
                    rt_flat = rt.view(bs, rt.shape[1], -1)
                    box_s, cls_s = split_head_logits(rs_flat, nc, reg_max)
                    box_t, cls_t = split_head_logits(rt_flat, nc, reg_max)
                    if kd_out_weight_box > 0.0:
                        loss_kd_out_box = loss_kd_out_box + F.mse_loss(box_s, box_t)
                    if kd_out_weight_cls > 0.0:
                        ps = torch.sigmoid(cls_s / temperature)
                        pt = torch.sigmoid(cls_t / temperature)
                        loss_kd_out_cls = loss_kd_out_cls + kldiv(torch.log(ps + 1e-9), pt) * (temperature ** 2)
                loss_kd_out_cls = loss_kd_out_cls / max(nls, 1)
                loss_kd_out_box = loss_kd_out_box / max(nls, 1)

            loss_total = (
                loss_det
                + kd_mid_weight * loss_kd_mid
                + kd_out_weight_cls * loss_kd_out_cls
                + kd_out_weight_box * loss_kd_out_box
            )
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
            optimizer.step()
        with torch.no_grad():
            metrics = validator(model=student)
            if isinstance(metrics, dict):
                print({k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()})

    torch.save({"model": student}, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--student", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all", choices=["mid", "out", "all"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save", type=str, default="distilled_student.pt")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    distill_train(
        teacher_path=args.teacher,
        student_path=args.student,
        data_yaml=args.data,
        kd_mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        device_str=args.device,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temperature,
        save_path=args.save,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
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
