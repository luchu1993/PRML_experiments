import matplotlib.pyplot as plt 
import numpy as np 
from numpy.linalg import inv
from functools import partial

def plot_predict(func, range, label='', resolution=0.02, color='red'):
    _x = np.arange(range[0], range[1], 0.02)
    _y = [ func(x) for x in _x ]
    plt.plot(_x, _y, color=color, label=label)


def phi(x, m):
    _phi = [ 1 ]
    for i in range(m):
        _phi.append(x**(i+1))
    return _phi

def Phi(X, m):
    _Phi = [ ]
    for x in X:
        _Phi.append(phi(x, m))
    return np.array(_Phi)

def predict(x, omega):
    return  omega.dot(phi(x, len(omega)-1))

RANGE = [0, 1]
X = np.linspace(*RANGE, 10)
target = np.sin(2*np.pi*X) + np.random.normal(0, 0.2, len(X))

for idx, dim in enumerate([2, 3, 6, 9]):
    # 计算模型参数
    _Phi = Phi(X, dim)
    omega = inv(_Phi.T.dot(_Phi)).dot(_Phi.T).dot(target)
    ## 正则化
    lambda_ = 1e-3
    omega_r = inv(lambda_*np.eye(dim+1)+ _Phi.T.dot(_Phi)).dot(_Phi.T).dot(target)

    # 预测函数
    predict = partial(predict, omega=omega)
    predict_r = partial(predict, omega=omega_r)

    # 绘制图像
    ax = plt.subplot(2, 2, idx+1)
    plot_predict(predict, RANGE, label='no regularization', color='red')
    plot_predict(predict_r, RANGE, label='regularization', color='green')
    plt.scatter(X, target)
    plt.title('Dim = $%d$' % dim)
    plt.legend()
    
plt.show()