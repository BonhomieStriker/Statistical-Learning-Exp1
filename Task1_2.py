from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
dataset = load_boston()
x_data = dataset.data # 特征变量
name_data = dataset.feature_names
print(name_data)
print(x_data.shape)
y_data = dataset.target # 预测目标房价
#去除异常数据
abnlist = np.array([])
for i in range(len(y_data)):
    if y_data[i] == 50.0:
        abnlist = np.append(abnlist, i) #存储异常值下标

x_data = np.delete(x_data, abnlist, axis=0)
y_data = np.delete(y_data, abnlist)

#Task - a 做出数据集中所有变量的散点图
for i in range(13):
    plt.subplot(4,4,i+1)
    plt.scatter(x_data[:,i],y_data)
    plt.xlabel(name_data[i])
    plt.ylabel('House Price')
plt.show()

#Task - b 计算变量之间的相关系数矩阵
#数据标准化处理
xc_data = np.transpose(x_data)
from sklearn.preprocessing import StandardScaler
#初始化特征标准化器
ss_X=StandardScaler()
#对特征进行标准化处理
x_corr=ss_X.fit_transform(xc_data)

#做出相关性矩阵图
x_corr = pd.DataFrame(x_corr)
corr = np.corrcoef(x_corr)  #计算变量之间的相关系数矩阵
print(corr)
print(corr.shape)
# plot correlation matrix
fig = plt.figure() #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)  #绘制热力图，从-1到1
fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
ticks = np.arange(0,13,1)
ax.set_xticks(ticks)  #生成刻度
ax.set_yticks(ticks)
ax.set_xticklabels(name_data,fontdict={'fontsize': 8}) #生成x轴标签
ax.set_yticklabels(name_data,fontdict={'fontsize': 8})
plt.show()

#Task - c 多元线性回归并给出性能评估
from sklearn.preprocessing import PolynomialFeatures
print("x_data:",x_data.shape)
print("______")
x_data_poly = PolynomialFeatures(degree = 2).fit_transform(x_data)
print("x_data_poly:",x_data_poly.shape)
print("______")

#数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,random_state = 33,test_size = 0.2)

#数据标准化处理
from sklearn.preprocessing import StandardScaler
#分别初始化对特征和目标值的标准化器
ss_X=StandardScaler()
ss_y=StandardScaler()

#分别对训练和测试数据的特征以及目标值进行标准化处理
x_train=ss_X.fit_transform(x_train)
x_test=ss_X.transform(x_test)
y_train=ss_y.fit_transform(y_train.reshape(-1,1))
y_test=ss_y.transform(y_test.reshape(-1,1))

#Linear Regression训练
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_y_predict = lr.predict(x_test)

#Linear Regression模型评估
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print("Linear Regression")
#r2_score
print("r2_score:",r2_score(y_test,lr_y_predict))
#MSE
print("MSE:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
#MAE
print("MAE:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

#Task - d 进行交叉验证分析
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(lr,x_data,y_data,cv = 10)

plt.subplot()
plt.scatter(y_data,predicted,edgecolors = (0,0,0))
plt.plot([y_data.min(),y_data.max()],[y_data.min(),y_data.max()],'k--',lw = 4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.show()
print(y_data.shape)
print(predicted.shape)
print('''------------------------------''')


#Task - e 改变划分比例观察指标变化
r2 = np.zeros(20)
mae = np.zeros(20)
mse = np.zeros(20)
for i in range(20):
    k = i+2
    predicted = cross_val_predict(lr, x_data, y_data, cv=k)
    # print(predicted)

    # print(r2_score(y_data, predicted))
    r2[i] = r2_score(y_data,predicted)

    # print(mean_absolute_error(y_data, predicted))
    mae[i] = mean_absolute_error(y_data,predicted)

    # print(mean_squared_error(y_data, predicted))
    mse[i] = mean_squared_error(y_data,predicted)

k = np.linspace(2, 21, 20, dtype=np.int)
plt.subplot(311)
# print(k)
# print(r2)
plt.plot(k,r2)
plt.xticks(np.arange(2, 21, 1))
plt.xlim([2, 21])
plt.xlabel("Number of K")
plt.ylabel("R2")

plt.subplot(312)
plt.plot(k, mae)
plt.xlim([2, 21])
plt.xticks(np.arange(2, 21, 1))
plt.xlabel("Number of K")
plt.ylabel("MAE")

plt.subplot(313)
plt.plot(k,mse)
plt.xlim([2, 21])
plt.xticks(np.arange(2, 21, 1))
plt.xlabel("Number of K")
plt.ylabel("MSE")

plt.show()