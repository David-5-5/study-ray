import torch
import numpy as np

# 张量初始化
# 直接从 数据创建
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# 从 NumPy 数组创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 从另一个张量创建
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor:\n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor:\n {x_rand} \n")

# 使用随机值或常数值
shape = (2, 3,)
rand_tensor = torch.rand(shape)
one_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor:\n {rand_tensor} \n")
print(f"Ones Tensor:\n {one_tensor} \n")
print(f"Zeros Tensor:\n {zeros_tensor} \n")

# 张量属性
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 张量运算包括包括转置、索引、切片、数学运算、线性代数、随机采样等 100 多种张量运算
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")

# 类似 NumPy 的标准索引和切片
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 张量乘法 This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax
print(f"tensor*tensor \n {tensor*tensor} \n")

# 张量矩阵乘法
print(f"tensor.matmul(tensor) \n {tensor.matmul(tensor)} \n")
# Alternative syntax
print(f"tensor @ tensor \n {tensor @ tensor} \n")

# 原地操作，后缀为 _
print(tensor, "\n")
tensor.add_(5) # t1 = tensor.add(5) # tensor 不变
print(tensor, "\n")



