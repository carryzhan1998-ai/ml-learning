"""
特征预处理之 归一化的介绍：
    目的：
        防止因为量纲问题，导致特征列的方差值相差较大， 影响模型的最终结果
        所以通过公式把 各列的值 映射到 均值为0, 标准差为1的 正态分布序列

    公式：
        x' = (当前值 - 该列平均值) / 该列的方差



"""

# 导包
from sklearn.preprocessing import StandardScaler

# 1. 准备数据集(标准化之前的原数据)
x_train = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 2. 创建标准化对象
transfer = StandardScaler()

# 3. 对原数据集进行标准化操作
x_train_new = transfer.fit_transform(x_train)

# 4. 打印处理后的数据集
print(f'标准化后的数据集为: \n {x_train_new}')

# 5. 打印数据集的均值和方差
print(f'数据集的均值为:{transfer.mean_}')
print(f'数据集的方差为:{transfer.var_}')  # var()
print(f'数据集的标准差为:{transfer.scale_}') # std()

