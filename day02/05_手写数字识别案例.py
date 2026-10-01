"""
案例：演示KNN算法 识别图片, 即：手写数字识别案例

介绍：
    每张图片都是由28 * 28 像素组成的，即：我们的csv文件中每一行都有 784 个像素点， 表示图片(每个像素点)的 颜色
    最终构成图像

"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter

# 1. 定义函数，接收用户传入的索引，展示该索引对应的图片
def show_digit(idx):
    # 1. 读取数据集，获取到源数据
    df = pd.read_csv('./data/手写数字识别.csv')
    # print(df)       # 42000 rows x 785 columns

    # 2. 判断传入的索引是否越界
    if idx < 0 or idx >= len(df):
        return

    # 3. 如果没有越界，则正常获取数据
    x = df.iloc[:, 1:]  #结合写入文件来看：x获取的是测试集的特征数据，除去第1列外的所有数据
    y = df.iloc[:, 0]

    # 4. 查看用户传入索引所对应的图片
    print(f'用户所传入的数字为{y.iloc[idx]}')

    # 5. 查看下 用户传入索引对应图片的 形状
    print(x.iloc[idx].shape)    #(784, ) 我们要将(784, ) -> (28, 28)
    # print(x.iloc[idx].values)     # 具体的784个像素点数据

    # 6. 将(784, ) -> (28, 28)
    x = x.iloc[idx].values.reshape(28, 28)
    # print(x)

    # 7. 具体的绘制灰度图的动作
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.show()

# 2. 定义函数，训练模型，保存已经训练好的模型
def train_model():
    df = pd.read_csv('./data/手写数字识别.csv')
    # 2.1 拆分出特征列
    x = df.iloc[:, 1:].to_numpy()
    # 2.2 拆分出标签列
    y = df.iloc[:, 0].to_numpy()

    # 2.3 打印特征和标签的形状
    print(f'x的形状{x.shape}')
    print(f'y的形状{y.shape}')
    print(f'查看所有标签的分布情况{Counter(y)}')

    # 2.4 对特征列进行归一化
    x = x / 255.0

    # 2.5 拆分数据集和训练集                                                                            # 参数stratify: 参考y值进行，按照标签比例进行抽取(使数据类型保持均衡)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y)

    # 3 训练模型
    # 3.1 创建模型对象
    estimator = KNeighborsClassifier()

    # 3.2 模型训练
    estimator.fit(x_train, y_train)

    # 4 模型评估
    print(f'模型的得分为: {estimator.score(x_test, y_test)}')
    print(f'准确率: {accuracy_score(y_test, estimator.predict(x_test))}')


    # 5. 保存模型
    # 参数1：模型对象      参数2：模型保存路径
    joblib.dump(estimator, './model/手写数字识别.pkl')
    print('模型保存成功！ ')

# def use_model():
#     # 1. 加载图片
#     x = plt.imread('./data/demo.png')
#
#     # 2绘制图片
#     # plt.imshow(x, cmap='gray')
#     # plt.axis('off')
#     # plt.show()
#
#
#     # 3. 加载模型
#     estimator = joblib.load('./model/手写数字识别.pkl')
#
#     # 4. 模型预测
#     # 4.1 查看数据集转换
#     # x = x.reshape(1, -1) / 255      # 可能会预测失败，因为读取图片时，图片的像素值可能不会特别精准
#     x = x.reshape(1, -1)
#     print(x.shape)
#
#     # 4.2 模型预测
#     y_pre = estimator.predict(x)
#     print(f'预测值为: {y_pre}')
def use_model():
    # 1. 加载图片
    x = plt.imread('./data/demo.png')
    print("图片原始shape：", x.shape)
    print("像素范围 min={}, max={}".format(x.min(), x.max()))

    # 处理RGB彩色图片，转灰度单通道
    # if len(x.shape) == 3:
    #     x = np.mean(x, axis=-1)

    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.show()

    # 现在x必须是 (28,28)，否则图片尺寸不对
    print("灰度后shape：", x.shape)

    # 展平为模型需要的 (1,784)
    x = x.reshape(1, -1)
    print("输入模型shape", x.shape)

    estimator = joblib.load('./model/手写数字识别.pkl')
    y_pre = estimator.predict(x)
    print(f"预测数字：{y_pre[0]}")



if __name__ == '__main__':
    # show_digit(9)
    train_model()
    use_model()