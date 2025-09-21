import os                           # 标准库：文件/目录、路径拼接、创建目录等
import cv2                          # OpenCV：图像读写、绘制可视化（在这里用来画旋转框、多边形）
import numpy as np                  # 数值计算库（这里基本用于数组类型转换）
import matplotlib.pyplot as plt     # 可视化：绘制训练曲线（Loss / mAP）
import pandas as pd                 # 读取results.csv并处理数据帧
from ultralytics import YOLO        # Ultralytics YOLO 类（v8系列，含OBB）

# ========== 1. 数据转换：DOTA -> YOLOv8-OBB ==========
def convert_dota_to_yoloobb(txt_path, save_path, class2id, img_width=256, img_height=256):
    os.makedirs(save_path, exist_ok=True)                    # 如果输出目录不存在则创建
    for file in os.listdir(txt_path):                        # 遍历标注目录中的所有文件
        if not file.endswith('.txt'):                        # 只处理txt标注文件
            continue
        with open(os.path.join(txt_path, file), 'r') as f:   # 读入单个标注文件
            lines = f.readlines()
        out = []                                             # 准备输出的YOLO-OBB行（字符串列表）
        for line in lines:                                   # 遍历每一行目标标注
            vals = line.strip().split()                      # 去空白并按空格分列
            if len(vals) < 9:                                # DOTA格式至少应为8个点坐标+类别名
                continue
            x = list(map(float, vals[:8]))                   # 前8列是四个点(x1,y1,...,x4,y4)
            cls = vals[8]                                    # 第9列是类别字符串
            if cls not in class2id:                          # 若类别不在映射字典中则跳过
                continue

            # 归一化坐标到0-1范围（偶数索引是x，用宽度除；奇数索引是y，用高度除）
            normalized_x = [coord / img_width if i % 2 == 0 else coord / img_height
                            for i, coord in enumerate(x)]

            # YOLOv8-OBB格式：class_id x1 y1 x2 y2 x3 y3 x4 y4（全部0~1）
            out.append(
                f"{class2id[cls]} {normalized_x[0]:.6f} {normalized_x[1]:.6f} {normalized_x[2]:.6f} {normalized_x[3]:.6f} {normalized_x[4]:.6f} {normalized_x[5]:.6f} {normalized_x[6]:.6f} {normalized_x[7]:.6f}\n")

        with open(os.path.join(save_path, file), 'w') as f:  # 将该标注文件转换后的所有行写回新目录
            f.writelines(out)
    print(f"数据转换完成: {txt_path} -> {save_path}")        # 打印转换完成信息

# ========== 2. 生成数据配置文件 ==========
def write_yaml(save_path="rsar.yaml"):
    # 使用绝对路径确保YOLO能找到数据
    current_dir = os.getcwd().replace('\\', '/')             # 获取当前工作目录并统一分隔符

    yaml_str = f"""# RSAR SAR目标检测数据集 (YOLOv8-OBB)
path: {current_dir}
train: train/images
val: val/images

names:
  0: ship
  1: aircraft
  2: car
  3: tank
  4: bridge
  5: harbor
"""
    with open(save_path, "w") as f:                          # 将上面的多行字符串写为数据配置yaml
        f.write(yaml_str)
    print(f"YAML配置文件已生成: {save_path}")                 # 提示生成成功
    print(f"数据集路径: {current_dir}")                      # 打印数据根路径

# ========== 3. 检查标注文件格式 ==========
def check_label_format(label_dir):
    """检查标注文件格式是否正确"""
    print("检查标注文件格式...")
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]  # 找到所有txt标注

    if not label_files:                                      # 若目录下没有标注，直接返回失败
        print("没有找到标注文件！")
        return False

    for label_file in label_files[:3]:                        # 只抽查前三个文件
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            if lines:                                        # 若文件非空
                first_line = lines[0].strip()                # 取第一行做检查
                values = first_line.split()
                print(f"文件: {label_file}, 第一行: {first_line}")
                print(f"列数: {len(values)}")

                # 检查是否为9列（class + 8个坐标）
                if len(values) != 9:
                    print(f"错误：应该有9列，实际有{len(values)}列")
                    return False

                # 检查坐标值是否在0-1范围内（此处默认已是YOLO归一化格式）
                try:
                    coords = list(map(float, values[1:]))    # 跳过class_id，仅检查8个坐标值
                    for i, coord in enumerate(coords):
                        if coord < 0 or coord > 1:
                            print(f"警告：坐标 {coord} 不在0-1范围内 (位置 {i + 1})")
                    print("坐标范围检查完成")
                except ValueError:                           # 若转换浮点失败，则格式不正确
                    print("错误：坐标值不是有效的浮点数")
                    return False

    return True                                              # 通过抽查则认为基本正确

# ========== 4. 可视化旋转框 ==========
def visualize_predictions(results, class_names, save_dir="runs/detect/vis"):
    os.makedirs(save_dir, exist_ok=True)                     # 确保输出目录存在
    for r in results:                                        # 遍历预测结果列表（Ultralytics返回的对象）
        img = r.orig_img.copy()                              # 拿到原图（numpy数组）
        if hasattr(r, 'obb') and r.obb is not None:          # 确认存在OBB结果（旋转框）
            boxes = r.obb.xyxyxyxy.cpu().numpy()             # 取四点坐标（x1y1...x4y4），形状(N,8)
            confs = r.obb.conf.cpu().numpy()                 # 置信度 (N,)
            clss = r.obb.cls.cpu().numpy().astype(int)       # 类别索引 (N,)

            for pts, conf, cls_id in zip(boxes, confs, clss):# 逐个目标绘制
                pts = pts.reshape(4, 2).astype(int)          # 8值->(4,2)四个点，并转为整型像素坐标
                color = (0, 255, 0)                          # 绿色轮廓
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)   # 画四边形
                label = f"{class_names[cls_id]} {conf:.2f}"  # 文字标签：类别名+置信度
                cv2.putText(img, label, (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # 写字到图上

        save_path = os.path.join(save_dir, os.path.basename(r.path))  # 输出图像路径（沿用原文件名）
        cv2.imwrite(save_path, img)                         # 保存可视化图
        print(f"可视化结果保存到: {save_path}")             # 打印保存位置

# ========== 5. 绘制 Loss 和 mAP 曲线 ==========
def plot_training_curves(csv_path="runs/detect/train/weights/results.csv", save_path="runs/detect/train/weights/curves.png"):
    if not os.path.exists(csv_path):                         # 若找不到结果CSV，直接返回
        print(f"未找到 {csv_path}，无法绘制曲线")
        return
    df = pd.read_csv(csv_path)                               # 读入训练过程的metrics CSV

    plt.figure(figsize=(12, 5))                              # 新建画布

    # Loss 曲线
    plt.subplot(1, 2, 1)                                     # 左子图：Loss
    if "train/box_loss" in df.columns:
        plt.plot(df["epoch"], df["train/box_loss"], label="box_loss")
    if "train/cls_loss" in df.columns:
        plt.plot(df["epoch"], df["train/cls_loss"], label="cls_loss")
    if "train/dfl_loss" in df.columns:
        plt.plot(df["epoch"], df["train/dfl_loss"], label="dfl_loss")
    if "train/obb_loss" in df.columns:
        plt.plot(df["epoch"], df["train/obb_loss"], label="obb_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    # mAP 曲线
    plt.subplot(1, 2, 2)                                     # 右子图：mAP
    if "metrics/mAP50" in df.columns:
        plt.plot(df["epoch"], df["metrics/mAP50"], label="mAP@0.5")
    if "metrics/mAP50-95" in df.columns:
        plt.plot(df["epoch"], df["metrics/mAP50-95"], label="mAP@0.5:0.95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()                                       # 子图布局优化
    plt.savefig(save_path)                                   # 保存曲线图
    plt.show()                                               # 显示图像（交互环境中可见）
    print(f"训练曲线已保存到 {save_path}")                   # 打印保存路径

# ========== 6. 训练 & 评估 ==========
def train_and_eval():
    class2id = {"ship": 0, "aircraft": 1, "car": 2, "tank": 3, "bridge": 4, "harbor": 5}  # 类别到id映射
    id2class = {v: k for k, v in class2id.items()}           # 反向映射：id到类别名

    print("=" * 50)
    print("SAR图像目标检测训练开始")
    print("图像尺寸: 256x256像素")
    print("=" * 50)

    # 检查数据目录
    print("检查数据目录...")
    required_dirs = [                                        # 训练/验证的标注与图像目录
        "train/labelTxt",
        "val/labelTxt",
        "train/images",
        "val/images"
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):                     # 缺任何目录直接报错返回
            print(f"错误：目录不存在 - {dir_path}")
            return
        print(f"✓ {dir_path}")                               # 存在则打勾

    # 创建labels目录（YOLO需要 labels 与 images 同名文件对应）
    os.makedirs("train/labels", exist_ok=True)
    os.makedirs("val/labels", exist_ok=True)

    # 检查图像文件（这里只统计png）
    train_images = [f for f in os.listdir("train/images") if f.endswith('.png')]
    val_images = [f for f in os.listdir("val/images") if f.endswith('.png')]

    print(f"训练图像数量: {len(train_images)}")
    print(f"验证图像数量: {len(val_images)}")

    if len(train_images) == 0 or len(val_images) == 0:       # 若没有图片，则退出
        print("错误：没有找到PNG图像文件！")
        return

    # 转换数据格式 - 使用正确的YOLOv8-OBB格式（归一化坐标）
    print("开始转换数据格式...")
    convert_dota_to_yoloobb("train/labelTxt", "train/labels", class2id, img_width=256, img_height=256)
    convert_dota_to_yoloobb("val/labelTxt", "val/labels", class2id, img_width=256, img_height=256)

    # 检查转换后的标注文件格式（抽查）
    if not check_label_format("train/labels"):
        print("标注文件格式检查失败！")
        return

    if not check_label_format("val/labels"):
        print("验证集标注文件格式检查失败！")
        return

    # 生成YOLO数据配置文件yaml
    write_yaml("rsar.yaml")

    # 加载模型（预训练OBB权重）
    print("加载YOLOv8-OBB模型...")
    try:
        model = YOLO("yolov8m-obb.pt")                       # 指定旋转框版本权重
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")                          # 捕获异常（如权重不存在/路径错误）
        return

    # 训练配置
    print("开始训练...")
    model.train(
        data="rsar.yaml",                                    # 数据配置文件
        epochs=50,                                           # 训练轮数（这里写50做快速试验）
        imgsz=256,                                           # 输入尺寸
        batch=16,                                            # batch size
        device=0,                                            # 使用GPU 0（无GPU时需改为'cpu'或省略）
        optimizer="AdamW",                                   # 优化器
        lr0=0.001,                                           # 初始学习率
        workers=4,                                           # dataloader的worker数量
        verbose=True                                         # 训练日志更详细
    )

    # 验证（评估指标）
    print("开始验证...")
    try:
        metrics = model.val()                                # 运行验证
        print("评估结果：")
        print(f"mAP50: {metrics.box.map50:.4f}")             # 打印mAP@0.5（注意：API版本差异可能在metrics命名上有所不同）
        print(f"mAP50-95: {metrics.box.map:.4f}")            # 打印mAP@0.5:0.95
    except Exception as e:
        print(f"验证失败: {e}")                              # 捕获验证异常

    # 推理测试（在验证集上画出预测）
    print("进行推理测试...")
    try:
        results = model.predict(
            source="val/images",                             # 输入目录
            save=True,                                       # YOLO会将可视化结果保存到runs目录
            imgsz=256,
            conf=0.3,                                        # 置信度阈值
            device=0
        )
        print("推理完成")

        # 自定义再可视化（用我们定义的旋转框绘制函数）
        visualize_predictions(results, id2class)

    except Exception as e:
        print(f"推理失败: {e}")

    # 绘制训练曲线（Loss/mAP）
    plot_training_curves()

    print("=" * 50)
    print("训练完成！")
    print("=" * 50)

if __name__ == "__main__":            # 仅当该脚本作为主程序执行时才启动训练流程
    train_and_eval()                   # 调用主流程函数
