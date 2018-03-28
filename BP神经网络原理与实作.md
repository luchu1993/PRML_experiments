# 深度前馈网络
本文参照**Deep Learning**第六章，实作了用于回归的多层神经网络 

## 引言
深度前馈网络是典型的，最基本的深度学习模型。前馈网络的目的是近似某个函数 $f^*$。前馈网络定义了一个模型 $\mathbf y = f(\mathbf x;\mathbf\theta)$，其中需要学习参数$\mathbf \theta$，使得函数能够得到最佳近似。

前馈网络通常由许多函数复合而成，该模型可以使用一个有向无环图描述，如下图所示

![深度前馈神经网络](DNN.png)

例如，从输入层$(input\ layer)$到隐藏层$(hidden\ layer)$的函数映射可以记作
$$
\begin{align}
	& \mathbf a^{(1)} = f_1(\mathbf x) = \mathbf W^{(1)}\mathbf x +\mathbf b^{(1)}\\
	& \mathbf h^{(1)} = \sigma(\mathbf a^{(1)})
\end{align}
$$
其中$\sigma(\cdot)$为*激活函数*, $h^{(1)}$为第一个隐藏层输出, $\mathbf W^{(1)}$和$\mathbf b^{(1)}$是第一层网络中需要学习的参数。前馈网络的最后一层为输出层$(output\ layer)$,输出层的函数映射也可以类似记作
$$
	\begin{align}
	& \mathbf a^{(L)} = f_L\left(\mathbf h^{(L-1)}\right) = \mathbf W^{(L)}\mathbf x +\mathbf b^{(L)}\\
	& \mathbf y =\mathbf h^{(L)} = \sigma(\mathbf a^{(L)})
\end{align}
$$
其中$L$为前馈网络的层数。深层前馈网络可以包含多个隐藏层，用以近似更为复杂的映射函数。$\mathbf y$为前馈网络的输出。

现假定有训练数据集$\{\mathbf x_n, \mathbf t_n\}_{n=1}^N$，其中$N$为训练集大小。在深度前馈网络的训练中，需要让其所代表的函数 $f(\mathbf x)$去匹配训练集中包含的真实映射$f^*(\mathbf x)$。

可以从线性模型开始理解深度前馈神经网络。对于线性模型，无论是通过解析解还是凸优化，都能够高效且可靠的拟合数据；但也有个明显的缺陷，线性模型的能力被局限在线性函数内，无法理解任意两个变量之间的相互作用。为了扩展线性模型能够表示非线性函数，可以将输入$\mathbf x$进行一次非线性变换，将线性模型作用在变换后的$\phi(\mathbf x)$上，这里$\phi(\mathbf x)$是一个非线性变换(例如，x的多项式变换)。

那么，如何选择映射$\phi(\mathbf x)$呢？

* 选择一个通用的$\phi(\mathbf x)$。例如，选择一个无限维的$\phi$，它隐含的用在基于RBF核的核机器是上。若$\phi(\mathbf x)$不是无限维的，只要有足够高的维度，总是有能力来你和训练集的，但是泛化性能往往不是太好。
* 手动设计$\phi(\mathbf x)$。在深度学习之前，这是主流的方法。但是需要丰富的特定领域的知识，并且不同领域之间很难迁移。
* 自动学习$\phi(\mathbf x)$。深度学习的策略是自动去学习特征映射$\phi$。在这种方法中，模型可以写作
$$
	\mathbf y = f(\mathbf x; \mathbf\theta) = \phi(\mathbf x;\mathbf\theta)^\top\mathbf\omega
$$
现在，模型有两种参数：
	1. 用于学习$\phi$的参数$\mathbf\theta$ 
	2. 将$\phi(\mathbf x)$映射到所需输出的参数$\mathbf\omega$
其中$\phi$定义了隐藏层。这种方法中，我们将特征映射表示成$\phi(\mathbf x;\mathbf\theta)$，并且使用优化算法来寻找$\mathbf\theta$，使其能够得到一个好的表示。另外，可以将先验知识编码进网络来帮助提升泛化性，只需要设计那些期望能够表现优异的函数族$\phi(\mathbf x;\mathbf \theta)$即可。