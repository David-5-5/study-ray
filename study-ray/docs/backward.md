# 复合函数链式求导法则

先一句话结论：

> 
> `loss.backward()` 的唯一工作：**自动完成链式求导，算出损失函数对所有参数的偏导数**，计算结果存入各自的 `.grad`。
> 也就是算出 \(\dfrac{\partial L}{\partial w}\) 放到 `w.grad`；算出 \(\dfrac{\partial L}{\partial b}\) 放到 `b.grad`。

---

## 拿咱们这个模型拆开

线性模型 \(\hat y_i = w x_i + b\)

损失函数 \(L = \frac1n\sum_{i=1}^n (\hat y_i - y_i)^2\)。

我们要求两个偏导数：\(\dfrac{\partial L}{\partial w},\ \dfrac{\partial L}{\partial b}\)。

把整个式子看成**多层复合函数**。
设：

\(z_i = \hat y_i - y_i\)

\(L=\frac1n\sum z_i^2\)

复合关系：
L 依赖 \(z_i\)；\(z_i\) 依赖 \(\hat y_i\)；\(\hat y_i\) 依赖 \(w,b\)。

复合函数链式法则

\(\frac{\partial L}{\partial w}= \sum_i \frac{\partial L}{\partial z_i}\cdot \frac{\partial z_i}{\partial \hat y_i}\cdot \frac{\partial \hat y_i}{\partial w}\)逐项求导

1. \(L=\frac1n\sum z_i^2 \implies \dfrac{\partial L}{\partial z_i}=\dfrac1n\cdot 2 z_i\)
2. \(z_i=\hat y_i-y_i \implies \dfrac{\partial z_i}{\partial \hat y_i}=1\)
3. \(\hat y_i = w x_i + b \implies \dfrac{\partial \hat y_i}{\partial w}=x_i\)

合并：

\(\frac{\partial L}{\partial w}=\frac1n\sum_{i} 2(\hat y_i-y_i)\cdot x_i\)同理：

\(\frac{\partial L}{\partial b}=\frac1n\sum_{i} 2(\hat y_i-y_i)\cdot 1\)> 
> **`loss.backward()`，就是帮你把上面这整套链式求导全部算完。**
> 算出这两个偏导数，存入 `w.grad`、`b.grad`。

## 对应你代码的四步，重新标注每一步的任务

```
y_pred = w * x_data + b   # 【前向】代入当前w,b，算出预测值，构建计算关系（计算图）
loss = torch.mean((y_pred - y_target)**2)

loss.backward()           # 【反向传播】链式求导，算出 ∂L/∂w, ∂L/∂b，写入w.grad、b.grad

with torch.no_grad():
    w -= lr * w.grad      # 【参数更新，人为梯度下降策略】
    b -= lr * b.grad

w.grad.zero_()            # 【梯度清零】清空本次梯度，防止下一轮累加
b.grad.zero_()
```

## 两个关键点（非常容易混淆）

1. **前向传播：只算函数的值，不算导数；但是会记住变量之间的依赖关系（计算图）**
没有前向算出 \(y_{pred}\)，就不知道各个中间变量的值，链式求导无法代入数值。
2. **反向传播：从最后的 loss，反向沿着链式法则，一层一层算出导数**
不是什么魔法，本质就是高数复合函数链式求导的**自动化批量执行**。

## 区分

- 高数的链式法则：是数学公式。
- 反向传播：【工程实现策略】，把链式求导交给计算机自动执行。不是高数定理。

