from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST 
import torch

def load_data_fashion_mnist(batch_size, resize=False):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    
    trans = transforms.Compose(trans)
    mnist_train = FashionMNIST('../data', train=True, transform=trans, download=True)
    mnist_test = FashionMNIST('../data', train=False, transform=trans, download=True)
    return (
        data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
        data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True))

# 训练
def train_epoch(train_iter, net, loss_func, updater):
    for X, y in train_iter:
        y_hat = net(X)
        loss = loss_func(y_hat, y)
        updater.zero_grad()
        loss.mean().backward()
        updater.step()
    
# 准确率
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 评估
def evaluate_accuracy(test_iter, net):
    metrics = Accumulator(2)
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X)
            acc = accuracy(y_hat, y)
            metrics.add(acc, y.numel())
    return metrics[0]/metrics[1]

# 迭代训练
def train(net, train_iter, test_iter, updater, loss_func, num_epochs):
    for epoch in range(num_epochs):
        train_epoch(train_iter, net, loss_func, updater)
        acc = evaluate_accuracy(test_iter, net)
        print(f'epoch: {epoch}, acc: {acc}')


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]