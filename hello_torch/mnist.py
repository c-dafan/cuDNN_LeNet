import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

batch_size = 256
epochs = 15


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, (5, 5), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, (5, 5), bias=False),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(120, 84, bias=False)
        self.classifier2 = nn.Linear(84, 10, bias=False)

    def forward(self, inputs):
        out_ = self.model(inputs)
        out_ = out_.view(out_.shape[0], out_.shape[1])
        out_ = self.classifier(out_)
        out_ = self.classifier2(out_)
        return out_


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

le_net = LeNet().to(device)
train = datasets.MNIST("D:/", download=True,
                       transform=transforms.Compose(
                           [transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x * 255)
                            ]))
dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(le_net.parameters())

for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        le_net.zero_grad()
        data_, target = data
        data_, target = Variable(data_).to(device), Variable(target).to(device)

        output = le_net(data_)
        err = criterion(output, target)
        err.backward()
        optimizer.step()
        pre = torch.argmax(F.softmax(output), 1)
        acc = (pre == target).sum().cpu().numpy() / batch_size

        print('[%d/%d][%d/%d] Loss:%.4f, acc:%.4f'
              % (epoch, epochs, i, len(dataloader), err.data, acc))

