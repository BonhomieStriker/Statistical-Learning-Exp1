from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
dataset = load_boston()
x_data = dataset.data # 特征变量
name_data = dataset.feature_names
y_data = dataset.target # 预测目标房价
'''
#先观察数据分布情况
for i in range(13):
    plt.subplot(4,4,i+1)
    plt.scatter(x_data[:,i],y_data)
    plt.xlabel(name_data[i])
    plt.ylabel('House Price')
plt.show()
'''
#去除异常数据
abnlist = np.array([])
for i in range(len(y_data)):
    if y_data[i] == 50.0:
        abnlist = np.append(abnlist, i) #存储异常值下标

x_data = np.delete(x_data, abnlist, axis=0)
y_data = np.delete(y_data, abnlist)
print("x_data.shape = {}".format(x_data.shape))
print("y_data.shape = {}".format(y_data.shape))
print("__________")

#数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,random_state = 33,test_size = 0.2)
print("x_train.shape = {}".format(x_train.shape))
print("y_train.shape = {}".format(y_train.shape))
print("__________")

#Task - a 简单线性回归
from sklearn.linear_model import LinearRegression
lr = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0]
for i in range(13):
    # # 检查x和y的形状
    print(x_train[:, i].reshape(-1, 1).shape)
    print(y_train.reshape(-1, 1).shape)

    # 拟合
    lr[i] = LinearRegression(fit_intercept=True).fit(x_train[:, i].reshape(-1, 1), y_train.reshape(-1, 1))  #参数估计
    print(lr[i].predict( x_test[:, i].reshape(-1, 1) ).shape)

    # 把每个特征对应的预测结果存起来
    if i == 0:
        lr_y_predict = np.expand_dims(lr[i].predict( x_test[:, i].reshape(-1, 1) ), axis=1)
    else:
        pred_temp = np.expand_dims(lr[i].predict( x_test[:, i].reshape(-1, 1) ), axis=1)
        lr_y_predict = np.concatenate((lr_y_predict, pred_temp), axis=1)

    print(lr_y_predict.shape)
    print("+++++++")

lr_y_predict = np.squeeze(lr_y_predict, axis=-1)
print(lr_y_predict.shape)



#Task - b 绘制变量关系图和最小二乘回归线
# x_test = np.sort(x_test, axis=0)
# y_test = np.sort(y_test, axis=0)
for i in range(13):
    plt.subplot(4, 4, i+1)
    plt.scatter(x_train[:, i], y_train)
    plt.scatter(x_test[:, i], y_test, color='red')
    plt.plot(x_test[:, i], lr[i].predict( x_test[:, i].reshape(-1, 1) ), color='black')
    plt.xlabel(name_data[i])
    plt.ylabel('House Price')
plt.show()

# #Task - c 评估回归结果
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
for i in range(13):
    print("Feature_name",name_data[i])
    print("r2_score:",r2_score(y_test,lr[i].predict( x_test[:, i].reshape(-1, 1))))
    print("MSE:",mean_squared_error(y_test,lr[i].predict( x_test[:, i].reshape(-1, 1))))
    print("MAE:",mean_absolute_error(y_test,lr[i].predict( x_test[:, i].reshape(-1, 1))))
    print("__________")

#Task - d 分别使用LinearRegression和SGDRegressor训练和预测，评估结果
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

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
sgdr = SGDRegressor()
sgdr.fit(x_train,y_train)
sgdr_y_predict = sgdr.predict(x_test)
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_y_predict = lr.predict(x_test)
plt.plot(y_test,color = 'blue')
plt.plot(lr_y_predict,color = 'red')
plt.plot(sgdr_y_predict,color = 'black')
plt.legend(['House Price','Linear Regression','SGD Regressor'],loc='upper right')
plt.show()

#Linear Regression模型评估
#r2_score
print("Linear Regression")
print("r2_score:",r2_score(y_test,lr_y_predict))
#MSE
print("MSE:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
#MAE
print("MAE:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

#SGDRegressor模型评估
print("SGDRegressor")
#r2_score
print("r2_score:",r2_score(y_test,sgdr_y_predict))
#MSE
print("MSE:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
#MAE
print("MAE:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))