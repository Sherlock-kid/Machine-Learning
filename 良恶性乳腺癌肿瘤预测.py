import pandas as pd
import numpy as np

#使用 sklearn.model_selection里的 train_test_split 模块用于分割数据
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier

from sklearn.metrics import classification_report

#良/恶性乳腺癌肿瘤数据预处理
#创建特征列表
column_names=['示例代码编号', '丛块厚度', '细胞大小的一致性', '细胞形状的一致性', '边缘附着力', '单个上皮细胞大小', '裸核', '平淡的染色质', '正常的核仁', '有丝分裂', '类']
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names = column_names)
#将？替换为标准缺失值表示
data = data.replace(to_replace='?', value=np.nan)
#丢弃带有缺失值的数据（只要有一个维度有缺失）
data = data.dropna(how='any')
print(data.shape)


#准备良/恶性乳腺癌肿瘤训练、测试数据
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
#查验训练样本的数量和类别分布
print(y_train.value_counts())
#查验测试样本的数量和类别分布
print(y_test.value_counts())


#使用线性分类模型从事良/恶性乳腺癌肿瘤预测任务
#标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
#初始化 LogisticRegression 和 SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()
#调用 LogisticRegression 中的 fit 函数/模块用来训练模型参数
lr.fit(X_train, y_train)
#使用训练好的模型 lr 对 X_test进行预测，结果储存在变量 lr_y_predict 中
lr_y_predict = lr.predict(X_test)
#调用 SGDClassifier 中的 fit 函数/模块用来训练模型参数
sgdc.fit(X_train, y_train)
#使用训练好的模型 sgdc 对 X_test进行预测，结果储存在变量 sgdc_y_predict 中
sgdc_y_predict = sgdc.predict(X_test)



#使用线性分类模型从事良/恶性肿瘤预测任务的性能分析
#使用逻辑斯蒂回归模型自带的评分函数 score 获得模型在测试集上的准确性结果
print('LR分类器的精度：', lr.score(X_test, y_test))
#利用 classification_report 获得 LogisticRegression 其他三个指标的结果
print(classification_report(y_test, lr_y_predict, target_names=['良性', '恶性']))
