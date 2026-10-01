"""
案例: 演示网格搜索 和 交叉验证

交叉验证解释：
    原理：
        将数据分成n份，例如分成：4份 -> 也叫4折交叉验证
        第1次： 把第1份数据作为 验证集(测试集)， 其它作为训练集，训练模型，模型预测，获取：准确率 -> 准确率1
        第2次： 把第2份数据作为 验证集(测试集)， 其它作为训练集，训练模型，模型预测，获取：准确率 -> 准确率1
        第3次： 把第3份数据作为 验证集(测试集)， 其它作为训练集，训练模型，模型预测，获取：准确率 -> 准确率1
        第4次： 把第4份数据作为 验证集(测试集)， 其它作为训练集，训练模型，模型预测，获取：准确率 -> 准确率1
        ......
        然后 计算上述4次准确率的平均值，作为：模型的最终准确率

        假如第四次准确率最高，则用全部数据(训练集 + 测试集)训练模型，再用第4次的数据作为测试集对模型进行测试

        目的：
            为了让模型的最终验证结果更加准确

网格搜索：
    目的：
        寻找最优超参数
    原理：
        接受超参可能出现的值， 然后对于超参的每个值进行交叉验证，寻找到最优超参组合

大白话解释：
    网格搜索 + 交叉验证， 本质上指的是 GridSearchCV这个API， 它会帮我们寻找最优超参
"""

# 导包
from sklearn.datasets import load_iris                              # 加载鸢尾花测试集
from sklearn.model_selection import train_test_split, GridSearchCV  # 把全部数据拆成「训练集」和「测试集」, 寻找最优超参(网格搜索 + 交叉验证)
from sklearn.neighbors import KNeighborsClassifier                  #KNN算法 分类对象
from sklearn.metrics import accuracy_score                          # **模型评估指标**，metrics 是度量模块-
from sklearn.preprocessing import StandardScaler                    # 数据标准化的

# 1. 加载鸢尾花数据集
iris_data = load_iris()

# 2. 数据预处理，这里是：切分训练集和测试集，比例：8:2
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state = 42)

# 3. 特征工程 -> 特征预处理 -> 特征标准化
# 3.1 创建标准化对象
transfer = StandardScaler()
# 3.2 对训练集和测试集的特征数据进行标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4 模型训练
# 4.1 创建 KNN分类对象
estimator = KNeighborsClassifier()
# 4.2 定义字典，记录 超参可能出现的情况值。
param_dict = {'n_neighbors': [i for i in range(1, 11)]}
# 4.3 创建 GridSearchCV对象 -> 寻找最优超参, 使用网格搜索和交叉验证
# 参数1：要计算最有超参的模型对象
# 参数2：给模型超参可能出现的值
# 参数3：交叉验证的折数，这里的4折表示：每个超参组合，都会进行4次交叉验证，这里共计4 * 10 = 40次
# 返回值 estimator -> 处理后的模型对象
estimator = GridSearchCV(estimator, param_dict, cv = 4)
# 4.4 具体的模型训练动作
estimator.fit(x_train, y_train)

# 4.5 打印最优超参组合
print(f'最优评分: {estimator.best_score_}')
print(f'最优超参组合: {estimator.best_params_}')
print(f'最优的评估器对象: {estimator.best_estimator_}')
print(f'具体的交叉验证结果: {estimator.cv_results_}')

# 5.模型评估
# 5.1 获取最优超参的 模型对象
# estimator = estimator.best_estimator
estimator = KNeighborsClassifier(n_neighbors=3)
# 5.2 模型训练。
estimator.fit(x_train, y_train)
# 5.3 模型预测
y_pred = estimator.predict(x_test)
# 5.4 模型评估
# 参数1：测试集，参数2：预测集
print(f'准确率: {accuracy_score(y_test, y_pred)}')
