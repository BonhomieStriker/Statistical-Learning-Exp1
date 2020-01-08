from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
dataset = load_boston()
x_data = dataset.data # 特征变量
name_data = dataset.feature_names
y_data = dataset.target # 预测目标房价
#去除异常数据
abnlist = np.array([])
for i in range(len(y_data)):
    if y_data[i] == 50.0:
        abnlist = np.append(abnlist, i) #存储异常值下标
x_data = np.delete(x_data, abnlist, axis=0)
y_data = np.delete(y_data, abnlist)
#数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,random_state = 33,test_size = 0.2)

#Task - a 分别实现岭回归和Lasso回归，分析不同输入特征和输出变量相关性
from sklearn.linear_model import Ridge,Lasso
from sklearn.linear_model import LinearRegression
rig = Ridge(alpha= 0.8)
rig.fit(x_train, y_train)
las = Lasso(alpha= 0.8)
las.fit(x_train,y_train)
lr = LinearRegression()
lr.fit(x_train,y_train)

#Task - b 在测试集上完成预测，并输出评估结果，与一般多元线性回归对比
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
y_lr_pred = lr.predict(x_test)
y_rig_pred = rig.predict(x_test)
y_las_pred = las.predict(x_test)
plt.plot(y_test)
plt.plot(y_lr_pred)
plt.plot(y_rig_pred)
plt.plot(y_las_pred)
plt.legend(['House Price','Linear Regression','Ridge Regression','Lasso Regression'])
plt.show()
print("Linear Regression")
#r2_score
print("r2_score:",r2_score(y_test,y_lr_pred))
#MSE
print("MSE:",mean_squared_error(y_test,y_lr_pred))
#MAE
print("MAE:",mean_absolute_error(y_test,y_lr_pred))
print("Ridge Regression")
#r2_score
print("r2_score:",r2_score(y_test,y_rig_pred))
#MSE
print("MSE:",mean_squared_error(y_test,y_rig_pred))
#MAE
print("MAE:",mean_absolute_error(y_test,y_rig_pred))
print("Lasso Regression")
#r2_score
print("r2_score:",r2_score(y_test,y_las_pred))
#MSE
print("MSE:",mean_squared_error(y_test,y_las_pred))
#MAE
print("MAE:",mean_absolute_error(y_test,y_las_pred))

#Task - c 改变岭回归和Lasso回归的α值，绘制回归系数变化图
alphas = np.logspace(0, 1, 9, base=0.1)
print(alphas)
rig_coef = []
rig_intercept = []
las_coef = []
las_intercept = []

print('--------------')
rig = Ridge(fit_intercept=True)
las = Lasso(fit_intercept=False)
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
ridge_alpha = Ridge(fit_intercept=False)
lasso_alpha = Lasso(fit_intercept=False)
coef_ridge = []
coef_lasso = []
for a in alphas:
    ridge_alpha.set_params(alpha=a,max_iter = 5000)
    lasso_alpha.set_params(alpha=a,max_iter = 5000)
    ridge_alpha.fit(x_train, y_train)
    lasso_alpha.fit(x_train, y_train)
    coef_ridge.append(ridge_alpha.coef_)
    coef_lasso.append(lasso_alpha.coef_)

plt.figure()
ax = plt.gca()
ax.plot(alphas, coef_ridge)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.axis('tight')
plt.title('Ridge coefficients as a function of the regularization')
plt.legend(name_data)
plt.show()

plt.figure()
ax = plt.gca()
ax.plot(alphas, coef_lasso)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.axis('tight')
plt.title('Lasso coefficients as a function of the regularization')
plt.legend(name_data)
plt.show()

