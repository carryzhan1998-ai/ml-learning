"""
案例：通过KNN算法实现 鸢尾花 的分类操作

回顾： 机器学习项目的研发流程
    1. 加载数据
    2. 数据预处理
    3. 特征工程
    4. 模型训练
    5. 模型评估
    6. 模型预测
"""

# 导包
from sklearn.datasets import load_iris  # 加载鸢尾花测试集
import seaborn as sns  # 基于matplolib的高级绘图库，专门做数据可视化的
import pandas as pd  # 把鸢尾花数据集的numpy数组转成表格
import matplotlib.pyplot as plt  # python底层绘图库，所有可视化的基础
from sklearn.model_selection import train_test_split  # sklearn的数据集划分工具- 把全部数据拆成「训练集」和「测试集」
#   - 训练集：给 KNN 模型学习规律
#   - 测试集：评估模型好坏（模型从来没见过的数据）
# - 常用参数：`test_size=0.2`，代表 20% 数据做测试集，80% 训练。
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  # **模型评估指标**，metrics 是度量模块-
from sklearn.preprocessing import StandardScaler


# accuracy_score`：**准确率**

# 1. 定义函数， 加载鸢尾花数据集，并查看数据集
def dm01_load_iris():
    # 1. 加载鸢尾花数据集
    iris_data = load_iris()
    # 2. 查看数据集
    # print(f'数据集: {iris_data}')
    # print(f'数据集的类型: {type(iris_data)}')
    # 3. 查看数据集的所有键
    print(f'数据集的所有键: {iris_data.keys()}')
    # 4. 查看数据集的键对应的值
    print(f'具体的数据: \n {iris_data.data[:5]}')
    print(f'具体的标签: \n {iris_data.target[:5]}')
    print(f'标签对应的名称: \n {iris_data.target_names[:5]}')
    print(f'特征对应的名称: \n {iris_data.feature_names[:5]}')
    # print(f'数据集的描述: \n {iris_data.DESCR}')
    print(f'数据集的框架: \n {iris_data.frame}')
    print(f'数据集的文件名: \n {iris_data.filename}')  # iris.csv
    print(f'数据集的模型(在哪个包下): {iris_data.data_module}')


# 2. 定义函数, 绘制数据集的散点图
def dm02_show_iris():
    # 1. 加载数据集。
    iris_data = load_iris()
    # 2. 把鸢尾花数据集封装成 DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    # 3. 给df对象新增1列 -> 标签列
    iris_df['label'] = iris_data.target
    # print(iris_df)

    # 4. 通过Seaborn绘制散点图
    sns.lmplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='label', fit_reg=False)
    plt.title('iris data')
    plt.tight_layout()  # 自动调整子图参数， 以使整个图像的边界与子图匹配
    plt.show()


# 3. 定义函数, 切分训练集和测试集
def dm03_split_train_test():
    # 1. 加载数据集
    iris_data = load_iris()

    # 2. 数据的预处理: 从150个特征和标签中, 按照8:2的比例, 切分训练集和数据集
    # 参数1: 特征数据, 参数2: 标签数据. 参数3: 测试集的比例
    # 返回值: 训练集的特征数据, 测试集的特征数据, 训练集的标签数据, 测试集的标签数据
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=34)

    # 3. 打印切割后的结果
    print(f'训练集的特征: {x_train}, 个数: {len(x_train)}')
    print(f'训练集的标签: {y_train}, 个数: {len(y_train)}')
    print(f'测试集的特征: {x_test}, 个数: {len(x_test)}')
    print(f'测试集的标签: {y_test}, 个数: {len(y_test)}')


# 4. 定义函数, 实现鸢尾花完整案例 -> 加载数据, 数据预处理, 特征工程, 模型训练, 模型预测, 模型评估
def dm04_iris_evaluate_test():
    # 1.加载数据
    iris_data = load_iris()
    # 2.数据预处理
    x_train, x_test, y_train, y_test=train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=25)
    # 3.特征工程(提取, 预处理...)
    # 思考1: 特征提取: 因为元数据只有4个特征值, 且特征列均会被用到， 所以这里无需做特征提取
    # 思考2: 特征预处理: 因为源数据的4列特征值差值不大, 所以我们无需做特征预处理, 但是加入特征预处理会使我们的代码更加完善
    # 3.1 创建标准化对象.
    transfer = StandardScaler()
    # 3.2 对特征列进行标准化, 即: x_train: 训练集的特征数据, x_test: 测试集的特征数据.
    # fit_transform: 兼具fit和transform功能: 训练和转换, 该参数适用于第一次进行特征标准化处理时使用，一般用于处理： 训练集
    x_train = transfer.fit_transform(x_train)
    # transform：只有转换，该函数适用于: 重复进行标准化动作时使用， 一般用于对测试集进行标准化
    x_test = transfer.transform(x_test)

    # 4.模型训练
    # 4.1 创建模型对象。
    estimator = KNeighborsClassifier(n_neighbors=5)
    # 4.2 具体的训练模型的动作
    estimator.fit(x_train, y_train)
    # 5.模型评估
    # 场景1：对刚才切分的 测试集(30条) 进行测试
    # 5.1 直接预测即可，获取：预测结果
    y_pred = estimator.predict(x_test)  #x_test: 测试集的特征数据
    # 5.2 打印预测结果.
    print(f'预测值为: {y_pred}')

    # 场景2：对新的数据集 (原数据集以外的的数据) 进行测试
    # 5.1 自定义测试数据集。
    my_data = [[7.5, 1.8, 3.9, 1.7]]

    # 5.2 对数据集进行标准化处理
    my_data = transfer.transform(my_data)

    # 5.3 模型预测
    y_pred_new = estimator.predict(my_data)
    print(f'预测值为：{y_pred_new}')

    # 5.4 查看上述数据集, 每种分类的预测概率
    y_pre_proba = estimator.predict_proba(my_data)
    print(f'(各分类)预测概率为: {y_pre_proba}')

    # 6.模型预测
    # 方式1：直接评分，基于：训练集的特征 和 训练集的标签
    print(f'正确率(准确率)：{estimator.score(x_train, y_train)}')

    # 方式2：基于测试集的标签 和预测结果 进行评分
    print(f'正确率(准确率)：{accuracy_score(y_test, y_pred)}')

# 5. 测试
if __name__ == '__main__':
    # dm01_load_iris()
    # dm02_show_iris()
    # dm03_split_train_test()
    dm04_iris_evaluate_test()