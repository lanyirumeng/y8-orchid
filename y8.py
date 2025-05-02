import os
import torch
from ultralytics import YOLO

def main():
    # 配置参数
    model_name = "yolov8l.pt"  
    data_yaml = "Plants/data.yaml"  # 数据集配置文件
    epochs = 50  # 训练轮数
    imgsz = 640  # 输入图像大小
    batch_size = 128  
    device = 0  # GPU 设备（0 为 GPU，-1 为 CPU）
    project = "runs/train"  # 训练结果保存目录
    name = "orchid_binary_exp"  # 实验名称
    workers = 2  # 数据加载线程数（降低显存占用）

    # 清理 GPU 内存
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    
    # 加载模型
    model = YOLO(model_name)

    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        augment=True,  # 启用数据增强
        mosaic=1.0,  # 启用 Mosaic 增强（可设为 0.0 降低显存）
        patience=10,  # 早停耐心
        save=True,  # 保存模型权重
        save_period=10,  # 每 10 个 epoch 保存一次
        workers=workers,
        amp=True,  # 启用混合精度训练
        lr0=0.01,  # 初始学习率
        optimizer="AdamW",  # 优化器
        cos_lr=True  # 余弦学习率调度
    )

    # 打印训练结果
    print("训练完成！结果保存在:", os.path.join(project, name))

    

if __name__ == '__main__':
    main()
