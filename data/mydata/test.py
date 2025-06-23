import os
import json
import shutil
from pathlib import Path

# 固定路径配置
base_dir = Path("/root/TensoRF/data/mydata")
images_dir = base_dir / "images"
train_dir = base_dir / "train"
test_dir = base_dir / "test"
json_path = base_dir / "transforms.json"
train_json_path = base_dir / "transforms_train.json"
test_json_path = base_dir / "transforms_test.json"
train_ratio = 0.8  # 训练集占比

# 创建输出目录
train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

# 加载 transforms.json
with open(json_path, "r") as f:
    transforms = json.load(f)

frames = transforms["frames"]
print(f"共读取 frames: {len(frames)} 张图像信息")

# 按图像编号排序（例如 ./images/0001）
frames_sorted = sorted(frames, key=lambda x: int(Path(x["file_path"]).stem.split('/')[-1]))
total = len(frames_sorted)
train_count = int(total * train_ratio)

# 均匀抽样：每隔 round(1/train_ratio) 留一个给 test，其余给 train
step = round(1 / (1 - train_ratio))  # 约为5
test_indices = list(range(0, total, step))
train_indices = [i for i in range(total) if i not in test_indices]

frames_train = [frames_sorted[i] for i in train_indices]
frames_test = [frames_sorted[i] for i in test_indices]

# 拷贝图片到 train/test，并更新 JSON 中的 file_path
def copy_and_update(frames_list, target_dir):
    updated = []
    for f in frames_list:
        src_path = base_dir / Path(f["file_path"][2:] + ".png")
        filename = src_path.name
        dst_path = target_dir / filename
        shutil.copy2(src_path, dst_path)
        f_new = f.copy()
        f_new["file_path"] = f"./{target_dir.name}/{filename[:-4]}"  # 去掉 .png 后缀
        updated.append(f_new)
    return updated

frames_train = copy_and_update(frames_train, train_dir)
frames_test = copy_and_update(frames_test, test_dir)

# 保存新 JSON 文件
def save_json(frames, path):
    new_meta = {k: v for k, v in transforms.items() if k != "frames"}
    new_meta["frames"] = frames
    with open(path, "w") as f:
        json.dump(new_meta, f, indent=2)

save_json(frames_train, train_json_path)
save_json(frames_test, test_json_path)

print(f"\n✅ 划分完成！")
print(f"训练集: {len(frames_train)} 张  → {train_dir}")
print(f"测试集: {len(frames_test)} 张  → {test_dir}")
print(f"JSON 保存: transforms_train.json / transforms_test.json")
