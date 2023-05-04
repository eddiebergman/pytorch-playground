from __future__ import annotations

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from playground.operator.shapes import shape

torch.manual_seed(0)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
matplotlib.use("QT5Cairo")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ],
)

batch_size = 4

trainset, testset = (
    MNIST(root="./data", train=train, download=True, transform=transform)
    for train in (True, False)
)
trainloader, testloader = (
    DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
    )
    for dataset, shuffle in ((trainset, True), (testset, False))
)

# Get image dimensions
h, w = next(iter(trainloader))[0].shape[-2:]


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# show images
# imshow(make_grid(images))
# print(" ".join(f"{trainset.classes[labels[j]]:5s}" for j in range(batch_size)))


class Net(nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3))
        h, w = shape(self.conv1, (h, w)).out

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        h, w = shape(self.pool, (h, w)).out

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=(5, 5))
        fc1_in = shape(self.conv2, (h, w)).flattened

        self.fc1 = nn.Linear(in_features=fc1_in, out_features=120)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=84)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        logger.debug(f"input shape: {x.shape} {[x.names]}")

        x = F.relu(self.conv1(x))
        logger.debug(f"convolved 1 shape: {x.shape}")

        x = self.pool(x)
        logger.debug(f"pooled shape: {x.shape}")

        x = F.relu(self.conv2(x))
        logger.debug(f"convolved 2 shape: {x.shape}")

        x = torch.flatten(x, start_dim=1)  # flatten all dimensions except batch
        logger.debug(f"flattened shape: {x.shape}")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net(h, w)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1)
scheduler = optim.lr_scheduler.ConstantLR(optimizer)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")
