import numpy as np
# ---------------------- 1. 构造数据集 & 参数 ----------------------
rng = np.random.default_rng(42)
X = rng.normal(0, 1, (50, 10))    # 50样本，10维特征
y = rng.integers(0, 3, 50)        # 3分类标签 0,1,2
params = {
    'W1': rng.normal(0, 0.1, (10, 8)),
    'b1': np.zeros(8),
    'W2': rng.normal(0, 0.1, (8, 3)),
    'b2': np.zeros(3),
}
# ---------------------- 2. 基础激活函数 & ReLU导数 ----------------------
def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    # ReLU导数：大于0=1，小于等于0=0
    return (x > 0).astype(float)

def softmax(z):
    # 按样本做softmax，keepdims保留维度防止广播错误
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
# ---------------------- 3. 前向传播（写好，不用改） ----------------------
def forward(X, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    z1 = X @ W1 + b1      # (50,8)
    a1 = relu(z1)         # (50,8)
    z2 = a1 @ W2 + b2     # (50,3)
    a2 = softmax(z2)      # (50,3) 每个样本3个类别概率
    cache = {
        "X":X, "z1":z1, "a1":a1, "z2":z2, "a2":a2, "W2":W2
    }
    return a2, cache
# ---------------------- 4. 交叉熵损失（写好，不用改） ----------------------
def cross_entropy_loss(a2, y):
    n = X.shape[0]
    log_probs = -np.log(a2[np.arange(n), y])
    loss = np.sum(log_probs) / n
    return loss
# ---------------------- 5. 反向传播【补全】 ----------------------
def backward(cache, y):
    # 从cache拿中间变量
    X = cache["X"]
    z1 = cache["z1"]
    a1 = cache["a1"]
    a2 = cache["a2"]
    W2 = cache["W2"]
    B = X.shape[0]   # batch size=50
    
    # softmax+交叉熵合并求导：dz2 = (yhat - y_onehot)/B
    yhat = softmax(cache["z2"])
    y_onehot = np.zeros_like(yhat)
    y_onehot[np.arange(B), y] = 1
    dlogits = (yhat - y_onehot) / B   # dz2 (50,3)

    # =====第二层梯度 W2, b2=====
    dW2 = a1.T @ dlogits       # (8,50) @ (50,3) = (8,3)
    db2 = np.sum(dlogits, axis=0) # (3,) 对样本维度求和

    # =====第一层梯度 W1, b1=====
    da1 = dlogits @ W2.T               # (50,3) @ (3,8) = (50,8)
    dz1 = da1 * relu_derivative(z1)    # ReLU链式求导，(50,8)
    dW1 = X.T @ dz1                    # (10,50) @ (50,8) = (10,8)
    db1 = np.sum(dz1, axis=0)          # (8,)

    grads = {"dW2":dW2, "db2":db2, "dW1":dW1, "db1":db1}
    return grads
# ---------------------- 6. 梯度检验（写好！用来验证你的backward） ----------------------
def grad_check(X, y, params, epsilon=1e-5):
    a2, _ = forward(X, params)
    loss_old = cross_entropy_loss(a2, y)
    # 把所有参数摊平成列表
    param_list = [('W1',params['W1']),('b1',params['b1']),('W2',params['W2']),('b2',params['b2'])]
    _, cache = forward(X, params)
    grads_analytic = backward(cache, y)
    print("===== 梯度检验结果（数值梯度 vs 解析梯度）=====")
    for name, p in param_list:
        grad_ana = grads_analytic["d"+name]
        # 遍历每一个元素
        it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
        for _ in range(5): # 随机抽5个点测试，不用全部算，加快速度
            idx = it.multi_index
            old_val = p[idx]
            p[idx] = old_val + epsilon
            a2_plus, _ = forward(X, params)
            loss_plus = cross_entropy_loss(a2_plus, y)
            p[idx] = old_val - epsilon
            a2_minus, _ = forward(X, params)
            loss_minus = cross_entropy_loss(a2_minus, y)
            p[idx] = old_val #恢复原值
            grad_num = (loss_plus - loss_minus) / (2*epsilon)
            diff = abs(grad_num - grad_ana[idx]) / max(abs(grad_num),abs(grad_ana[idx]),1e-10)
            print(f"{name}{idx} | 数值梯度:{grad_num:.6f} 解析梯度:{grad_ana[idx]:.6f} 相对误差:{diff:.2e}")
            it.iternext()
# ---------------------- 运行入口 ----------------------
if __name__ == "__main__":
    a2, cache = forward(X, params)
    loss = cross_entropy_loss(a2, y)
    print("初始损失：", loss)
    grad_check(X, y, params)
