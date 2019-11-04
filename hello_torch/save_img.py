import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 1
epochs = 15

train = datasets.MNIST("D:/", download=True,
                       transform=transforms.Compose(
                           [transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x * 255)
                            ]))
dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        data_, target = data
        cv2.imshow("aaa", data_[0][0].numpy().astype('uint8'))
        cv2.waitKey(100)
        cv2.imwrite("./minst/{}.png".format(target.numpy()[0]), data_[0][0].numpy().astype('uint8')[:, :, np.newaxis])
    break
