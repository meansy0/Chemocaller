import torch

# 使用 torch.jit.load 直接加载 TorchScript 模型
checkpoint_path = '/public/home/xiayini/project/NewL/Decoder/results/T_5mC/epoch1_test/model_best.pt'  # 你的模型文件路径

# 加载 TorchScript 模型
model = torch.jit.load(checkpoint_path)

# 将模型移动到设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 将模型设置为评估模式
model.eval()

# 模型现在已经加载好，可以进行评估
print("Model loaded and ready for evaluation.")
