"""前向的形状链：
  Z1 = X  @ W1 + b1      (50,784)@(784,128) → (50,128)
  A1 = relu(Z1)                              → (50,128)
  Z2 = A1 @ W2 + b2      (50,128)@(128,10)  → (50,10)
  p  = softmax(Z2)                           → (50,10)
  L  = 交叉熵(p, y)                           → 标量

反向必须凑出的 7 个形状：
  dlogits  (50,10)
  dW2      (128,10)     ← 必须和 W2 一样
  db2      (10,)        ← 必须和 b2 一样
  dA1      (50,128)
  dZ1      (50,128)
  dW1      (784,128)    ← 必须和 W1 一样
  db1      (128,)       ← 必须和 b1 一样

两个"记住的结论"（不用推）：
  softmax + 交叉熵：  dlogits = (p - onehot) / B
  ReLU：              dZ1 = dA1 * (Z1 > 0)      ← 逐元素乘，不是 @

手上有这些变量：
  X(50,784)  Z1(50,128)  A1(50,128)  Z2(50,10)  W2(128,10)  dlogits(50,10)"""

"""手写两层 MLP 反向传播 —— 阶段 2：小数据 + 梯度检验

结构：x(50,784) → [W1,b1] → z1(50,128) → ReLU → a1(50,128)
                → [W2,b2] → z2(50,10) → softmax → 交叉熵 → L

这一步不用真数据：随机 50 个样本就够。
目的不是训练效果，而是用「数值梯度 vs 解析梯度」验证手写反向传播是否正确。
"""
import numpy as np

# ==================== 1. 随机小数据 ====================
rng = np.random.default_rng(42)

X = rng.normal(0, 1, (50, 784))                      # (50, 784)
y = rng.integers(0, 10, 50)                          # (50,)  10 分类

params = {
    'W1': rng.normal(0, np.sqrt(2/784), (784, 128)),  # (784,128)  He 初始化
    'b1': np.zeros(128),                              # (128,)
    'W2': rng.normal(0, np.sqrt(2/128), (128, 10)),   # (128,10)
    'b2': np.zeros(10),                               # (10,)
}


# ==================== 2. 基础函数 ====================
def relu(x):
    return np.maximum(x, 0.0)


def relu_derivative(x):
    """ReLU 的导数：x > 0 处为 1，否则为 0"""
    return (x > 0).astype(x.dtype)


def softmax(z):
    """z:(B,10) → 每行归一化成概率"""
    z = z - z.max(axis=1, keepdims=True)      # 减最大值，防止 exp 溢出
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ==================== 3. 前向 ====================
def forward(params, X):
    """X:(B,784) → logits:(B,10), cache"""
    z1 = X @ params['W1'] + params['b1']      # (B,784)@(784,128) → (B,128)
    a1 = relu(z1)                             # (B,128)
    z2 = a1 @ params['W2'] + params['b2']     # (B,128)@(128,10)  → (B,10)
    return z2, (X, z1, a1, z2)


# ==================== 4. 反向 ====================
def backward(params, cache, y):
    """返回 grads，形状必须和 params 完全一致"""
    # x(50,784) → [W1,b1] → z1(50,128) → ReLU → a1(50,128)
    #             → [W2,b2] → z2(50,10) → softmax → 交叉熵 → L
    #   X(50,784)
    #   y(50,)

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    X,z1,a1,z2 = cache

    B = X.shape[0]
    p = softmax(z2)
    onehot = np.zeros_like(p)
    onehot[np.arange(B),y] = 1.0
    #   dz2(50,10) 
    dlogits = (p - onehot) / B

    #   w2(128,10)
    dW2 = a1.T @ dlogits 

    #   b2(128,)
    db2 = dlogits.sum(axis=0)

    #   a1(50,128)
    da1 = dlogits @ W2.T

    dz1 = da1 * relu_derivative(z1)

    dW1 = X.T @ dz1

    db1 = dz1.sum(axis=0)

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


# ==================== 5. 损失 ====================
def loss_fn(params, X, y):
    """交叉熵：取正确类别的概率，-log 之后按 batch 平均"""
    logits, _ = forward(params, X)
    p = softmax(logits)                       # (B,10)
    B = X.shape[0]
    return -np.log(p[np.arange(B), y] + 1e-12).mean()


# ==================== 6. 梯度检验 ====================
def numeric_grad(f, params, key, idx, eps=1e-5):
    """数值梯度：(f(w+eps) - f(w-eps)) / (2*eps)"""
    p = params[key]
    old = p[idx]
    p[idx] = old + eps; fp = f(params)
    p[idx] = old - eps; fm = f(params)
    p[idx] = old
    return (fp - fm) / (2 * eps)


# ==================== 7. 主程序 ====================
if __name__ == '__main__':
    logits, cache = forward(params, X)
    grads = backward(params, cache, y)

    # ---- 形状自检 ----
    for k in params:
        assert grads[k].shape == params[k].shape, \
            f'{k} 形状错: {grads[k].shape} != {params[k].shape}'
    print('形状自检通过 ✅\n')

    # ---- 梯度检验 ----
    f = lambda p: loss_fn(p, X, y)
    print(f'loss = {f(params):.6f}\n')

    max_rel = 0.0
    for key in ['W1', 'b1', 'W2', 'b2']:
        picks = rng.choice(params[key].size, 3, replace=False)
        for flat in picks:
            idx = np.unravel_index(flat, params[key].shape)
            ng = numeric_grad(f, params, key, idx)
            ag = grads[key][idx]
            rel = abs(ng - ag) / (abs(ng) + abs(ag) + 1e-12)
            max_rel = max(max_rel, rel)
            print(f'{key}{str(idx):12s} 数值={ng:+.8f} 解析={ag:+.8f} 相对误差={rel:.2e}')

    print(f'\n最大相对误差 = {max_rel:.2e}')
    print('✅ 通过' if max_rel < 1e-5 else '❌ 未通过，检查梯度')