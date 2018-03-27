import numpy as np 
import matplotlib.pyplot as plt 
# Sigmoid 函数实现
def sigmoid(x):
    return 1/(1+np.exp(-x))

def _sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

# ReLU 函数实现
def relu(x):
    return x * (x > 0)

def _relu(x):
    return x > 0

class Net():
    
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)

        # 初始化参数
        self.W = [ np.random.randn(y, x) for (x, y) in zip(layers[:-1], layers[1:]) ]
        self.b = [ np.random.randn(x, 1) for x in layers[1:] ]

    def forward(self, x):
        for w, b in zip(self.W[:-1], self.b[:-1]):
            x = relu( np.dot(w, x) + b ) 
        return np.dot(self.W[-1], x) + self.b[-1]
 
    def backword(self, x, y):
        activations = []
        hiddens = [x]

        h = x
        for w, b in zip(self.W[:-1], self.b[:-1]):
            a = np.dot(w, h) + b
            h = relu(a)
            activations.append(a)
            hiddens.append(h)
        out = np.dot(self.W[-1], hiddens[-1]) + self.b[-1]
        activations.append(out)
        hiddens.append(out)
        
        nabla_W = [ np.zeros(w.shape) for w in self.W ]
        nabla_b = [ np.zeros(b.shape) for b in self.b ]

        delta = self.cost_deriative(hiddens[-1], y) # * _sigmoid(activations[-1])
        nabla_W[-1] = np.dot( delta, hiddens[-2].T)
        nabla_b[-1] = delta

        for l in range(2, self.num_layers):
            delta = np.dot(self.W[-l+1].T, delta) * _relu(activations[-l])
            nabla_W[-l] = np.dot( delta, hiddens[-l-1].T)
            nabla_b[-l] = delta

        return nabla_W, nabla_b

    def update_gradient(self, batch, eta):
        batch_nabla_w = [ np.zeros(w.shape) for w in self.W ]
        batch_nabla_b = [ np.zeros(b.shape) for b in self.b ]
        N = len(batch)
        
        for x, y in batch:
            nabla_w, nabla_b = self.backword(x, y)
            batch_nabla_w = [ b_nw + nw for b_nw, nw in zip(batch_nabla_w, nabla_w) ]
            batch_nabla_b = [ b_nb + nb for b_nb, nb in zip(batch_nabla_b, nabla_b) ]

        self.W = [ w - (eta/N)*nabla_w for w, nabla_w in zip(self.W, batch_nabla_w) ]
        self.b = [ b - (eta/N)*nabla_b for b, nabla_b in zip(self.b, batch_nabla_b) ]

    def SGD(self, training_data, epochs, bath_size, eta, testing_data=None):
        N = len(training_data)
        loss = []
        for epoch in range(epochs):
            batches =[ training_data[k : k+bath_size] for k in range(0, N, bath_size) ]
            for batch in batches:
                self.update_gradient(batch, eta)
            train_loss = np.sum( [ (self.forward(x)-y)**2/N for (x, y) in training_data ] ) 
            loss.append(train_loss)
            if testing_data:
                test_loss = np.sum( [ (self.forward(x)-y)**2/N for (x, y) in training_data ] ) 
                print("epoch %d completed, tain_loss = %f, test_loss = %f." % (epoch+1, train_loss, test_loss))
            else:
                print("epoch %d completed, tain_loss = %f." % (epoch+1, train_loss) )
        
        return loss

    def cost_deriative(self, predict, y):
        return predict - y


if __name__ == '__main__':
    X = np.linspace(0, 1, 100)
    Y = np.sin(2*np.pi*X) + np.random.normal(0, 0.3, len(X))
    training_data = [ (x , y ) for (x, y) in zip(X, Y) ]

    net = Net([1, 16, 16, 1])
    train_loss = net.SGD(training_data, epochs=500, bath_size=5, eta=0.02)

    plt.subplot(1, 2, 1)
    plt.scatter(X, Y)
    _x = np.linspace(0, 1, 100)
    predict = [ np.squeeze(net.forward(x)) for x in _x ]
    plt.plot(_x, predict, color='red')

    plt.subplot(1, 2, 2)
    plt.plot(train_loss)

    plt.show()