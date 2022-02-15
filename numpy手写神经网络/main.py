import numpy as np
import random
import struct
import pickle
from batchnorm import BatchNorm
import math


def load_mnist_data(kind):
    '''
    加载数据集
    :param kind: 加载训练数据还是测试数据
    :return: 打平之后的数据和one hot编码的标签
    '''
    labels_path = '../data/%s-labels-idx1-ubyte' % kind
    images_path = '../data/%s-images-idx3-ubyte' % kind
    with open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images / 255., np.eye(10)[labels]


def sigmoid(z):
    '''
    sigmoid激活函数
    :param z: 神经网络的输出
    :return: z激活之后的值
    '''
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    '''
    sigmoid激活函数的导数
    :param z: 神经网络的输出
    :return: 关于z的导数
    '''
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    '''
    relu激活函数
    :param z: 神经网络的输出
    :return: z激活之后的值
    '''
    return np.maximum(0, z)


def relu_prime(z):
    '''
    relu激活函数的导数
    :param z: 神经网络的输出
    :return: 关于z的导数
    '''
    z_ = np.copy(z)
    z_[z > 0] = 1
    z_[z < 0] = 0
    z_[z == 0] = 0.5
    return z_


def leaky_relu(z):
    '''
    leaky relu激活函数
    :param z: 神经网络的输出
    :return: z激活之后的值
    '''
    return np.where(z > 0, z, z * 0.01)


def leaky_relu_prime(z):
    '''
    leaky relu激活函数的导数
    :param z: 神经网络的输出
    :return: 关于z的导数
    '''
    z_ = np.copy(z)
    z_[z > 0] = 1
    z_[z < 0] = 0.01
    z_[z == 0] = 0.5
    return z_


def mean_squared_loss(z, y_true):
    """
    均方误差损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值
    :return:
    """
    # y_predict = sigmoid(z)
    # y_predict = relu(z)
    y_predict = leaky_relu(z)
    loss = np.mean(np.mean(np.square(y_predict - y_true), axis=-1))  # 损失函数值
    # dy = 2 * (y_predict - y_true) * sigmoid_prime(z) / y_true.shape[1]  # 损失函数关于网络输出的梯度
    # dy = 2 * (y_predict - y_true) * relu_prime(z) / y_true.shape[1]
    dy = 2 * (y_predict - y_true) * leaky_relu_prime(z) / y_true.shape[1]
    return loss, dy


def cross_entropy_loss(y_predict, y_true):
    """
    交叉熵损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值,shape(N,d)
    :return:
    """
    y_exp = np.exp(y_predict)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
    dy = y_probability - y_true
    return loss, dy


class MLP_Net:
    def __init__(self, sizes, loss_type='mse'):
        self.sizes = sizes
        self.num_layers = len(sizes)
        weights_scale = 0.01
        self.weights = [np.random.randn(ch1, ch2) * weights_scale for ch1, ch2 in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(1, ch) * weights_scale for ch in sizes[1:]]
        # self.weights = [np.zeros((ch1, ch2)) * weights_scale for ch1, ch2 in zip(sizes[:-1], sizes[1:])]
        # self.biases = [np.zeros((1, ch)) * weights_scale for ch in sizes[1:]]
        # with open('../weights.pkl', 'rb') as f:
        #     weights = pickle.load(f)
        # with open('../biases.pkl', 'rb') as f:
        #     biases = pickle.load(f)
        # self.weights = [w.T for w in weights]
        # self.biases = [b.T for b in biases]
        self.X = None
        self.Z = None

        self.loss_type = loss_type
        self.drop_ratio = 1
        self.normalise = False
        self.dropout_X = None
        self.training = True

        self.norm_layers = [BatchNorm(shape=784, requires_grad=False, affine=False)]
        for size in self.sizes[1: -1]:
            self.norm_layers.append(BatchNorm(size))

    def forward(self, x):
        if self.normalise is True:
            x = self.norm_layers[0].forward(x)
        self.X = [x]
        self.dropout_X = []
        self.Z = []
        for layer_idx, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(x, w) + b
            if self.normalise is True and layer_idx < self.num_layers - 2 and self.training is True:
                # 前向过程的Batch Normalization
                self.norm_layers[layer_idx + 1].is_test = not self.training
                z = self.norm_layers[layer_idx + 1].forward(z)

            if self.drop_ratio != 1 and self.training is True:
                # 前向过程的dropout
                self.dropout_X.append(np.random.rand(z.shape[0], z.shape[1]) <= self.drop_ratio)
                z *= self.dropout_X[-1]
                z /= self.drop_ratio
            # x = sigmoid(z)
            # x = relu(z)
            x = leaky_relu(z)
            self.X.append(x)
            self.Z.append(z)
        return self.X[-1]

    def backward(self, y):
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        if self.loss_type == 'mse':
            loss, delta = mean_squared_loss(self.Z[-1], y)
        else:
            loss, delta = cross_entropy_loss(self.Z[-1], y)
        batch_size = len(y)
        for l in range(self.num_layers - 2, -1, -1):
            x = self.X[l]

            if self.drop_ratio != 1 and self.training is True:
                # 反向过程的dropout
                delta *= self.dropout_X[l]
                delta /= self.drop_ratio
            db[l] = np.sum(delta, axis=0) / (batch_size)
            dw[l] = np.dot(x.T, delta) / batch_size

            if l > 0:
                # delta = np.dot(delta, self.weights[l].T) * sigmoid_prime(self.Z[l - 1])
                # delta = np.dot(delta, self.weights[l].T) * relu_prime(self.Z[l - 1])
                delta = np.dot(delta, self.weights[l].T) * leaky_relu_prime(self.Z[l - 1])
                if self.normalise is True and self.training is True:
                    # 后向过程的Batch Normalization
                    self.norm_layers[l].backward(delta)
        return dw, db

    def update_para(self, dw, db, lr, l1=0, l2=0):
        if l1 != 0:
            # L1范数正则化
            self.weights = [w - lr * (nabla + l1 * np.sign(w)) for w, nabla in zip(self.weights, dw)]
            self.biases = [b - lr * nabla for b, nabla in zip(self.biases, db)]
        elif l2 != 0:
            # L2范数正则化
            self.weights = [w - lr * (nabla + l2 * w) for w, nabla in zip(self.weights, dw)]
            self.biases = [b - lr * nabla for b, nabla in zip(self.biases, db)]
        else:
            # 不进行正则化
            self.weights = [w - lr * nabla for w, nabla in zip(self.weights, dw)]
            self.biases = [b - lr * nabla for b, nabla in zip(self.biases, db)]


def plot_trainning(order1, order2, img_name):
    '''
    画出训练过程的对比图
    :param order1: 第一种网络结构
    :param order2: 第二种网络结构
    :param img_name: 图片名称
    :return:
    '''
    with open(order1, 'rb') as f1, open(order2, 'rb') as f2:
        accs1 = pickle.load(f1)
        accs2 = pickle.load(f2)

    import matplotlib.pyplot as plt
    plt.figure()
    # x = [str(i) for i in range(1, len(accs1) + 1)]
    x = [i for i in range(1, len(accs1) + 1)]
    plt.plot(x, accs1, label=order1)
    plt.plot(x, accs2, label=order2)
    plt.legend()
    # plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(img_name)


def plot_single_training(order, img_name='best_acc.png'):
    '''
    画出最优参数下的训练过程
    :param order:
    :param img_name:
    :return:
    '''
    with open(order, 'rb') as f1:
        accs = pickle.load(f1)
    import matplotlib.pyplot as plt
    plt.figure()
    x = [i for i in range(1, len(accs) + 1)]
    plt.plot(x, accs)
    # plt.legend()
    # plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(img_name)


def train(net, train_images, train_labels, test_images, test_labels, epochs=1000, lr=0.1, l2=0, batch_size=128, l1=0, orders='first', gamma=1, step_size=0):
    lr0 = lr
    n_test = len(test_labels)
    n = len(train_images)
    accs = []
    for epoch in range(epochs):
        net.training = True
        for batch_index in range(0, n, batch_size):
            lower_range = batch_index
            upper_range = batch_index + batch_size
            if upper_range > n:
                upper_range = n
            train_x = train_images[lower_range: upper_range, :]
            train_y = train_labels[lower_range: upper_range]
            net.forward(train_x)
            dw, db = net.backward(train_y)
            net.update_para(dw, db, lr, l1=l1, l2=l2)
        print(lr, end='\t')
        if step_size != 0:
            # 阶梯式衰减
            if (epoch + 1) % step_size == 0:
                lr *= gamma
        elif gamma != 1:
            # 指数衰减
            lr = math.pow(gamma, epoch) * lr0
        acc = evaluate(net, test_images, test_labels)
        accs.append(acc / 10000.0)
        print('Epoch {0}: {1} / {2}'.format(epoch, acc / 10000.0, n_test))
        with open(orders, 'wb') as f:
            pickle.dump(accs, f)
    plot_single_training(orders)
    # plot_trainning(accs)


def evaluate(net, test_images, test_labels):
    net.training = False
    result = []
    n = len(test_images)
    for batch_indx in range(0, n, 128):
        lower_range = batch_indx
        upper_range = batch_indx + 128
        if upper_range > n:
            upper_range = n
        test_x = test_images[lower_range: upper_range, :]
        result.extend(np.argmax(net.forward(test_x), axis=1))
    correct = sum(int(pred == y) for pred, y in zip(result, test_labels))
    return correct


def main():
    train_images, train_labels = load_mnist_data(kind='train')
    test_images, test_labels = load_mnist_data('t10k')
    test_labels = np.argmax(test_labels, axis=1)
    net = MLP_Net([784, 1024, 64, 10], 'ce')
    orders1 = 'no_regular'
    train(net, train_images, train_labels, test_images, test_labels, epochs=100, orders=orders1, batch_size=64, lr=0.3, gamma=0.5, step_size=30)


if __name__ == '__main__':
    np.random.seed(1)
    main()

