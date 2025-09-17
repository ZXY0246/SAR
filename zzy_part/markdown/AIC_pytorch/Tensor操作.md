**形状变换**
```python
x = torch.randn(4,4) #  生成一个形状为4x4的随机矩阵。
x = x.reshape(2,8) # 通过reshape操作，可以将4x4的矩阵改变为2x8的矩阵。
```

**生成操作**
```python
rand_tensor = torch.rand(shape) # 生成一个从[0,1]均匀抽样的tensor。  
randn_tensor = torch.randn(shape) # 生成一个从标准正态分布抽样的tensor。  
ones_tensor = torch.ones(shape) #生成一个值全为1的tensor。  
zeros_tensor = torch.zeros(shape) # 生成一个值全为0的tensor。  
twos_tensor = torch.full(shape, 2) #  生成一个值全为2的tensor。
```

---

permute函数来交换tensor的维度（转置）
- reshape是按元素顺序重新组织维度，
- permute会改变元素的顺序
```python

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x_reshape = x.reshape(3,2)
x_transpose = x.permute(1,0) #交换第0个和第1个维度。对于二维矩阵就是行列互换，进行转置。
print("reshape:",x_reshape)
print("permute:",x_transpose)

```
输出如下：
```python
reshape: tensor([[1, 2],
        [3, 4],
        [5, 6]])
permute: tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

**统计函数**

一个tensor中包含多个元素，对这些元素可以进行统计操作。比如通过
- `tensor.sum()`求和，
- `tensor.mean()`求均值，
- `tensor.std()`求标准差，
- `tensor.min()`求最小值等。