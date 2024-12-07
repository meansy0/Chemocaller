import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys

# 读取数据文件
iterations = []
losses = []
acc=[]


epoch_num=sys.argv[1]
in_folder=sys.argv[2]
loss_file=f"{in_folder}/batch.log"
png_file=f"{in_folder}/batch.png"
# 读取文件并从第num行开始处理数据
num=0

with open(loss_file, 'r') as file:
    next(file)  # 跳过标题行
    for i, line in enumerate(file, start=1):
        if i < num+1:
            continue  # 跳过前num个数据
        # if i > 2*(num+1):
        # if i > num+200:
        #     break
        parts = line.strip().split()
        iterations.append(int(parts[0]))  # 读取 Iteration 列
        losses.append(float(parts[1]))    # 读取 Loss 列
        acc.append(float(parts[3]))    # 读取 Loss 列

# 转换为NumPy数组
iterations = np.array(iterations) - iterations[0] + num  # 使横坐标从第num次迭代开始
losses = np.array(losses)
acc = np.array(acc)

# # 定义拟合函数-二次多项式
# def poly2(x, a, b, c):
#     return a * x**2 + b * x + c

# # 拟合曲线
# popt, _ = curve_fit(poly2, iterations, losses)

# # 生成拟合曲线的y值
# fitted_losses = poly2(iterations, *popt)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(iterations, losses, marker='o', linestyle='-', color='green', markersize=5, label='loss')
plt.plot(iterations, acc, marker='o', linestyle='-', color='pink', markersize=5, label='acc')
# plt.plot(iterations, fitted_losses, linestyle='--', color='pink', label='Fitted Curve')
plt.title(f'CU-Loss over Iterations(epoch={epoch_num})')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig(png_file)