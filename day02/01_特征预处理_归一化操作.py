"""
特征预处理之 归一化的介绍：
    目的：
        防止因为量纲问题，导致特征列的方差值相差较大， 影响模型的最终结果
        所以通过公式把 各列的值 映射到[0，1] 区间

    公式：
        x' = (当前值 - 该列最小值) / (该列最大值 - 该列最小值)
        x'' = x' * (mx - mi) + mi

    弊端：
        容易受到最大值 和 最小值的影响， 所以它一般用于处理小数据集

"""

# 导包
from sklearn.preprocessing import MinMaxScaler #归一化对象

# 1. 准备数据集(归一化之前的原数据)
x_train = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 2. 创建归一化对象
# 参数feature_range 表示生成范围， 默认为：(0, 1)
transfer = MinMaxScaler(feature_range=(3, 5))

# 3. 对原数据集进行归一化操作
x_train_new = transfer.fit_transform(x_train)

# 4. 打印处理后的数据
print(f'归一化处理后的数据: \n{x_train_new}')