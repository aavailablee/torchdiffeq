import os
import argparse
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=240)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--test_freq', type=int, default=50)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

dir = f'png_{args.niters}_{args.test_freq}'

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

t = torch.linspace(0., 25., args.data_size).to(device)

# 1. 读取 CSV 文件，Known先不用管
'''
Time,Temperature,Known
2024-01-01 00:00:00,3.4858446,True
2024-01-01 01:00:00,6.13371,True
2024-01-01 02:00:00,4.770752,True
2024-01-01 03:00:00,9.528632,True
2024-01-01 04:00:00,10.105482,True
'''
df = pd.read_csv('./temperature_data_with_datetime.csv')

# 确保 'Time' 列是 datetime 类型
df['Time'] = pd.to_datetime(df['Time'])

# 2. 提取小时信息
df['Hour'] = df['Time'].dt.hour  # 获取小时部分

# 3. 使用 'Known' 列作为掩码
mask = df['Known'].values  # 已知数据的掩码（True为已知，False为缺失）

# 1. 获取所有 'False' 的索引
false_indices = np.where(mask == False)[0]

# 2. 找到连续的 False 区间
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

# 输出所有的 False 区间
print("False 区间：", intervals)
print(intervals[0][0])
# 已知的时间（小时）和温度数据
t_known = df['Hour'].values  # 获取小时数据
h_known = df['Temperature'].values

# 获取最大最小值
min_temp, max_temp = df['Temperature'].min(), df['Temperature'].max()

# 对温度数据进行 Min-Max 归一化
t_known = t_known / 23
h_known_normalized = (h_known - min_temp) / (max_temp - min_temp)  # 归一化到 [0, 1]


# 将时间和温度数据转换为 tensor
t_known = torch.tensor(t_known, dtype=torch.float32).unsqueeze(-1)  # shape: (N, 1)
h_known_normalized = torch.tensor(h_known_normalized, dtype=torch.float32).unsqueeze(-1)  # shape: (N, 1)

# 拼接时间和温度数据，得到一个形状为 (N, 1, 2) 的 Tensor
data = torch.cat([t_known, h_known_normalized], dim=-1)  # 在最后一维拼接
true_data = data.unsqueeze(-2).to(device) # shape: (data_size, 1, channels)
true_y0 = true_data[0] # shape: (1, channels)

def denormalize(pred_y_normalized, min_temp, max_temp):
    return pred_y_normalized * (max_temp - min_temp) + min_temp

def get_batch_tmp():
    # todo 将空值作为分割点，不要取这些去拟合模型，看行不行
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_data[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_data[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs(dir)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(111, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-1, 1)
        ax_traj.legend()

        fig.tight_layout()
        plt.savefig('{s}/{:03d}'.format(itr, s=dir))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.Tanh(), # tanh8, relu 12
            nn.Linear(256, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch_tmp()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        # 计算损失时使用反归一化
        # loss = torch.mean(torch.abs(denormalize(pred_y, min_temp, max_temp) - denormalize(batch_y, min_temp, max_temp)))
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                # 1. 填补缺失数据
                data_filled = true_data.clone().to(device)

                # 2. 遍历所有缺失的数据点
                for s, e in intervals:
                    pred_y = odeint(func, true_data[s-1], t[s:e+1]).to(device)
                    data_filled[s:e+1] = pred_y.squeeze(0)
                # 3. 拼接填补后的数据，作为新的数据集
                pred_y = data_filled.to(device)  # 将填补后的数据转换为适合模型的形式

                # 4. 计算损失
                loss = torch.mean(torch.abs(pred_y - true_data))  # 使用填补后的数据计算损失

                # pred_y = odeint(func, true_y0, t)
                # loss = torch.mean(torch.abs(pred_y - true_data))
                # 计算损失时使用反归一化
                # loss = torch.mean(torch.abs(denormalize(pred_y, min_temp, max_temp) - denormalize(batch_y, min_temp, max_temp)))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                print(true_data.shape)
                print(pred_y.shape)
                visualize(true_data, pred_y, func, ii)
                ii += 1

        end = time.time()

# nohup python ode_demo_copy.py --viz > output.log 2>&1 &