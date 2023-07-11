from torch.utils import data
from torchvision import transforms
from torchvision.datasets import FashionMNIST 

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


train_iter, _ = load_data_fashion_mnist(5)
for X, y in train_iter:
    print(X.shape, y.shape)
    break