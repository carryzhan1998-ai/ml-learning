"""手写两层 MLP 反向传播 —— MNIST 训练版

结构：x(784) → Linear(784,128) → ReLU → Linear(128,10) → softmax → 交叉熵
只依赖 numpy + torchvision（后者仅用于下载/读取 MNIST 原始数据）。
"""
import numpy as np
import torch
from torchvision import datasets

# ==================== 1. 数据 ====================
train_ds = datasets.MNIST(root='../data', train=True,  download=True)
test_ds  = datasets.MNIST(root='../data', train=False, download=True)   # ← train=False

X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_train = train_ds.targets.numpy()
X_test  = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_test  = test_ds.targets.numpy()


# ==================== 2. 基础函数 ====================
def relu(x):
    return np.maximum(x, 0.0)


def relu_derivative(x):
    return (x > 0).astype(np.float32)


def softmax(z):
    """沿类别维做 softmax；先减最大值防止 exp 溢出"""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, y):
    """logits:(B,10), y:(B,) → 标量。用 log-sum-exp 保证数值稳定"""
    z = logits - logits.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - log_z
    return -logp[np.arange(len(y)), y].mean()


# ==================== 3. 模型 ====================
def init_params(in_dim=784, hidden=128, out_dim=10, seed=42):
    """He 初始化：std = sqrt(2/fan_in)，适合 ReLU"""
    rng = np.random.default_rng(seed)
    return {
        'W1': rng.normal(0, np.sqrt(2.0/in_dim), (in_dim, hidden)),
        'b1': np.zeros(hidden, dtype=np.float32),
        'W2': rng.normal(0, np.sqrt(2.0/hidden), (hidden, out_dim)),
        'b2': np.zeros(out_dim, dtype=np.float32),
    }


def forward(params, X):
    """X:(B,784) → logits:(B,10), cache"""
    z1 = X @ params['W1'] + params['b1']        # (B,128)
    a1 = relu(z1)                                # (B,128)
    z2 = a1 @ params['W2'] + params['b2']        # (B,10)
    return z2, (X, z1, a1, z2)


def backward(params, cache, y):
    """返回的 grads 形状必须和 params 完全一致"""
    X, z1, a1, z2 = cache
    B = X.shape[0]

    # --- softmax + 交叉熵 的联合梯度 ---
    p = softmax(z2)                          # (B,10)
    onehot = np.zeros_like(p)
    onehot[np.arange(B), y] = 1.0
    dlogits = (p - onehot) / B               # (B,10)

    # --- 第 2 层：Z2 = A1 @ W2 + b2 ---
    dW2 = a1.T @ dlogits                     # (128,10)
    db2 = dlogits.sum(axis=0)                # (10,)
    dA1 = dlogits @ params['W2'].T           # (B,128)

    # --- ReLU：A1 = relu(Z1) ---
    dZ1 = dA1 * relu_derivative(z1)          # (B,128) 逐元素

    # --- 第 1 层：Z1 = X @ W1 + b1 ---
    dW1 = X.T @ dZ1                          # (784,128)
    db1 = dZ1.sum(axis=0)                    # (128,)

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


def accuracy(params, X, y, batch=1000):
    correct = 0
    for i in range(0, len(X), batch):
        logits, _ = forward(params, X[i:i+batch])
        correct += (logits.argmax(axis=1) == y[i:i+batch]).sum()
    return correct / len(X)


# ==================== 4. 训练一个 epoch ====================
def train_epoch(params, X, y, rng, lr, batch_size):
    perm = rng.permutation(len(X))
    total_loss = 0.0

    for i in range(0, len(X), batch_size):
        idx = perm[i:i+batch_size]
        Xb, yb = X[idx], y[idx]

        logits, cache = forward(params, Xb)
        total_loss += cross_entropy(logits, yb) * len(idx)   # 复用 logits，不重复前向

        grads = backward(params, cache, yb)
        for k in params:                                     # 手写 SGD
            params[k] -= lr * grads[k]

    return total_loss / len(X)


# ==================== 5. 主程序 ====================
if __name__ == '__main__':
    print(f'X_train {X_train.shape}  y_train {y_train.shape}')
    print(f'X_test  {X_test.shape}  y_test  {y_test.shape}\n')

    params = init_params()
    rng = np.random.default_rng(42)

    for epoch in range(10):
        loss = train_epoch(params, X_train, y_train, rng, lr=0.1, batch_size=64)
        print(f'epoch {epoch:2d}  loss={loss:.4f}  '
              f'train_acc={accuracy(params, X_train[:5000], y_train[:5000]):.4f}  '
              f'test_acc={accuracy(params, X_test, y_test):.4f}')