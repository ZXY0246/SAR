### Tensor是多维数组

Tensor是PyTorch里对多维数组的表示。可以用它来表示：

**标量（0维）**：单个数，比如 `torch.tensor(3.14)`

**向量（1维）**：一列数，比如 `torch.tensor([1,2,3])`

**矩阵（2维）**：行列数据，比如 `torch.tensor([[1,2],[3,4]])`

**高维张量（3维及以上）**：高维数据，比如`torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])`

---

在创建tensor时，PyTorch会根据你传入的数据，自动推断tensor的类型，当然，你也可以自己指定类型。比如：

```
import torch
t1 = torch.tensor((2,2),dtype=torch.float32)
print(t1)
```

PyTorch里的数据类型，主要为：

**整数型** torch.uint8、torch.int32、torch.int64。其中**torch.int64为默认的整数类型**。

**浮点型** torch.float16、torch.bfloat16、 torch.float32、torch.float64，其中**torch.float32为默认的浮点数据类型**。

**布尔型** torch.bool


---

tensor有几个常用的关键属性，
- 第一个是tensor的**形状**
- 第二个是tensor内**元素的类型**
- 第三个是tensor的**设备**