import matplotlib.pyplot as plt
import numpy as np 
from numpy.linalg import inv, det

def sampling(Pi, mean, cov, N):
    samples = np.array([])
    for idx in range(len(Pi)):
        _sample = np.random.multivariate_normal(mean[idx], cov[idx], int(N*Pi[idx]))
        samples = np.append(samples, _sample)
    return samples.reshape((-1, mean[0].shape[0]))


# 多维高斯分布
def gaussion(x, mu, Sigma):
    dim = len(x)
    constant = (2*np.pi)**(-dim/2) * det(Sigma)**(-0.5)
    return constant * np.exp(-0.5*(x-mu).dot(inv(Sigma)).dot(x-mu))


# 高斯混合分布
def gaussion_mixture(x, Pi, mu, Sigma):
    z = 0
    for idx in range(len(Pi)):
        z += Pi[idx]* gaussion(x, mu[idx], Sigma[idx])
    return z


def plot_gaussion(Pi, mu, Sigma):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    x, y = np.meshgrid(x, y)
    X = np.array([x.ravel(), y.ravel()]).T
    z = [ gaussion_mixture(x, Pi, mu, Sigma) for x in X ]
    z = np.array(z).reshape(x.shape)
    return plt.contour(x, y, z)


def EM_step(X, Pi, mu, Sigma):
    N = len(X); K = len(Pi)
    gamma = np.zeros((N, K))

    # E-step
    for n in range(N):
        p_xn = 0
        for k in range(K):
            t = Pi[k]*gaussion(X[n], mu[k], Sigma[k]) 
            p_xn += t
            gamma[n, k] = t
        gamma[n] /= p_xn

    # M-step
    for k in range(K):
        _mu = np.zeros(mu[k].shape)
        _Sigma = np.zeros(Sigma[k].shape)
        N_k = np.sum(gamma[:,k])

        # 更新均值
        for n in range(N):
            _mu += gamma[n,k]*X[n]
        mu[k] = _mu / N_k

        # 更新方差
        for n in range(N):
            delta = np.matrix(X[n]- mu[k]).T
            _Sigma += gamma[n, k]*np.array( delta.dot(delta.T) )
        Sigma[k] = _Sigma / N_k

        # 更新权重
        Pi[k] = N_k / N

    return Pi, mu, Sigma


if __name__ == '__main__':
    Pi = np.array([ 0.3, 0.3, 0.4 ])
    mu = np.array([
        [-6, 3],
        [3, 6],
        [0, -6]
    ])
    Sigma = np.array([
        [[4,0], [0,4]],
        [[4,1], [1,4]],
        [[6,2], [2,6]]
    ])

    # EM 算法
    _Pi = np.array([
        0.33, 
        0.33, 
        0.34
    ])
    _mu = np.array([
        [0, -1],
        [1, 0],
        [-1, 0]
    ])
    _Sigma = np.array([
        [[1,0], [0,1]],
        [[1,0], [0,1]],
        [[1,0], [0,1]]
    ])

    n_iter = 3
    samples = sampling(Pi, mu, Sigma, 200)
    
    # 初始状态
    plt.subplot(2, 2, 1)
    plt.title('Initialization')
    plt.scatter(*samples.T)
    plot_gaussion(_Pi, _mu, _Sigma)

    for i in range(n_iter):
        # EM算法迭代
        _Pi, _mu, _Sigma = EM_step(samples, _Pi, _mu, _Sigma)
        # 绘制迭代结果
        plt.subplot(2, 2, i+2)
        plt.title('Iteration = $%d$' % (i+1))
        plt.scatter(*samples.T)
        plot_gaussion(_Pi, _mu, _Sigma)
    plt.show()