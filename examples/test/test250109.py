import numpy as np
import torch

# 假设 false_indices 和 batch_time 已经定义
false_indices = [5, 6, 10, 11, 12, 15]  # 示例数据
batch_time = 3  # 假设我们设置了 batch_time
args = {'data_size': 1, 'batch_size': 2}  # 示例参数

# 获取 intervals
intervals = []
if len(false_indices) > 0:
    # 初始化区间的开始位置
    start_idx = false_indices[0]

    for i in range(1, len(false_indices)):
        # 如果当前的索引不是前一个索引的连续位置，则说明区间结束
        if false_indices[i] != false_indices[i-1] + 1:
            # 将当前区间的开始和结束位置保存
            intervals.append((start_idx, false_indices[i-1]))
            # 更新开始位置为当前索引
            start_idx = false_indices[i]
    
    # 将最后一个区间添加到 intervals
    intervals.append((start_idx, false_indices[-1]))

# 打印 intervals
print("Intervals:", intervals)

# 生成新的随机起始值集合
start_values = []

for interval in intervals:
    # 计算起始值 - batch_time 并将其加入集合
    start_idx = interval[0] - batch_time
    if start_idx >= 0:  # 确保不会生成负数索引
        start_values.append(start_idx)

# 打印计算得到的新的起始值
print("Start values after subtraction of batch_time:", start_values)

# 从新的起始值集合中随机选择 batch_size 个起始值
s = torch.from_numpy(np.random.choice(start_values, args['batch_size'], replace=False))

# 打印最终的随机选择结果
print("Randomly chosen indices:", s)
