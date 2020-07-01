# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串

# ----------------------------------------------------------------------------------
# 第一步 切分训练集和测试集
# ----------------------------------------------------------------------------------

X = []  # 定义图像名称
Y = []  # 定义图像分类类标
Z = []  # 定义图像像素
# 记得更改此处4或者10
for i in range(0, 10):
    # 遍历文件夹，读取图片
    for f in os.listdir("photo2/%s" % i):
        # 获取图像名称
        X.append("photo2//" + str(i) + "//" + str(f))
        # 获取图像类标即为文件夹名称
        Y.append(i)
# print(X)
# print(Y)
# os.walk 可以遍历多层路径，使用root, dirs, files
#
# for root, dirs, files in os.walk("photo"):
#         # Y.append(type_name)
#     for file in files:
#         X.append(os.path.join(root, file))
# print(X)
# print(Y)

X = np.array(X)
Y = np.array(Y)

# 随机率为100% 选取其中的20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=1)

print(len(X_train), len(X_test), len(y_train), len(y_test))

# ----------------------------------------------------------------------------------
# 第二步 图像读取及转换为像素直方图
# ----------------------------------------------------------------------------------

# 训练集
XX_train = []
for i in X_train:
    # 读取图像
    # print i
    image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 图像像素大小一致
    img = cv2.resize(image, (256, 256),
                     interpolation=cv2.INTER_CUBIC)

    # 计算图像直方图并存储至X数组
    hist = cv2.calcHist([img], [0, 1], None,
                        [256, 256], [0.0, 255.0, 0.0, 255.0])

    XX_train.append(((hist / 255).flatten()))

# 测试集
XX_test = []
for i in X_test:
    # 读取图像
    # print i
    # 不使用imread，而是用imdecode以识别中文路径
    image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 图像像素大小一致
    img = cv2.resize(image, (256, 256),
                     interpolation=cv2.INTER_CUBIC)

    # 计算图像直方图并存储至X数组
    hist = cv2.calcHist([img], [0, 1], None,
                        [256, 256], [0.0, 255.0, 0.0, 255.0])

    XX_test.append(((hist / 255).flatten()))

# ----------------------------------------------------------------------------------
# 第三步 基于支持向量机的图像分类处理
# ----------------------------------------------------------------------------------
# 0.5
# 常见核函数‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
clf = SVC().fit(XX_train, y_train)
clf = SVC(kernel="linear").fit(XX_train, y_train)
predictions_labels = clf.predict(XX_test)

# ----------------------------------------------------------------------------------
# 第三步 基于决策树的图像分类处理
# ----------------------------------------------------------------------------------
# 0.36
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier().fit(XX_train, y_train)
# predictions_labels = clf.predict(XX_test)

# ----------------------------------------------------------------------------------
# 第三步 基于KNN的图像分类处理
# ----------------------------------------------------------------------------------
# 0.11
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=11).fit(XX_train, y_train)
# predictions_labels = clf.predict(XX_test)

# ----------------------------------------------------------------------------------
# 第三步 基于朴素贝叶斯的图像分类处理
# ----------------------------------------------------------------------------------
# 0.01
# from sklearn.naive_bayes import BernoulliNB
# clf = BernoulliNB().fit(XX_train, y_train)
# predictions_labels = clf.predict(XX_test)

print(u'预测结果:')
print(predictions_labels)
print(u'算法评价:')
print(classification_report(y_test, predictions_labels))

# 输出前10张图片及预测结果
# k = 0
# while k < 10:
#     # 读取图像
#     print(X_test[k])
#     image = cv2.imread(X_test[k])
#     print(predictions_labels[k])
#     # 显示图像
#     # cv2.imshow("img", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     k = k + 1

# -*-coding:utf-8-*-

# labels表示你不同类别的代号，比如这里的demo中有10个类别
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# labels = ['条形缺陷', '未焊透', '未熔合', '圆形缺陷']

'''
具体解释一下re_label.txt和pr_label.txt这两个文件，比如你有100个样本
去做预测，这100个样本中一共有10类，那么首先这100个样本的真实label你一定
是知道的，一共有10个类别，用[0,9]表示，则re_label.txt文件中应该有100
个数字，第n个数字代表的是第n个样本的真实label（100个样本自然就有100个
数字）。
同理，pr_label.txt里面也应该有1--个数字，第n个数字代表的是第n个样本经过
你训练好的网络预测出来的预测label。
这样，re_label.txt和pr_label.txt这两个文件分别代表了你样本的真实label和预测label，然后读到y_true和y_pred这两个变量中计算后面的混淆矩阵。当然，不一定非要使用这种txt格式的文件读入的方式，只要你最后将你的真实
label和预测label分别保存到y_true和y_pred这两个变量中即可。
'''
y_true = y_test  # 正确标签
y_pred = predictions_labels  # 预测标签
# 如果用10类照片就把下面注释掉
# y_true = []  # 正确标签
# y_pred = []  # 预测标签
# for i in range(len(y_test)):
#     if y_test[i] == 0:
#         y_true.append('条形缺陷')
#     elif y_test[i] == 1:
#         y_true.append('未焊透')
#     elif y_test[i] == 2:
#         y_true.append('未熔合')
#     elif y_test[i] == 3:
#         y_true.append('圆形缺陷')
#
# for i in range(len(predictions_labels)):
#     if predictions_labels[i] == 0:
#         y_pred.append('条形缺陷')
#     elif predictions_labels[i] == 1:
#         y_pred.append('未焊透')
#     elif predictions_labels[i] == 2:
#         y_pred.append('未熔合')
#     elif predictions_labels[i] == 3:
#         y_pred.append('圆形缺陷')

tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig('matrix.png', format='png')
plt.show()
