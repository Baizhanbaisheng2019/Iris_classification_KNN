# -*- coding: utf-8 -*-

"""
    任务：鸢尾花识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_FILE = './data/Iris.csv'

# 相比于自定义分类器, 标签多加了编号信息
SPECIES_LABEL_DICT = {
    'Iris-setosa':      0,  # 山鸢尾
    'Iris-versicolor':  1,  # 变色鸢尾
    'Iris-virginica':   2   # 维吉尼亚鸢尾
}

# 使用的特征列
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def main():
    """
        主函数
    """
    # 读取数据集
    iris_data = pd.read_csv(DATA_FILE, index_col='Id')
    iris_data['Label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)

    # SPECIES_LABEL_DICT 这种写法可以直接将对应的Species映射成对应的编号

    # 获取数据集特征
    X = iris_data[FEAT_COLS].values

    # 获取数据标签
    y = iris_data['Label'].values

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

    # 声明模型
    knn_model = KNeighborsClassifier()
    # 训练模型
    knn_model.fit(X_train, y_train)
    # 评价模型
    accuracy = knn_model.score(X_test, y_test)
    print('预测准确率:{:.2f}%'.format(accuracy * 100))

    # 输出每个测试样本预测情况
    for idx in range(50):
        test_sample_feat = [X_test[idx, :]]
        y_true = y_test[idx]
        y_pred = knn_model.predict(test_sample_feat)
        print('真实标签{}，预测标签{}'.format(y_true, y_pred))


if __name__ == '__main__':
    main()
