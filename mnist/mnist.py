import torch
import torchvision
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from time import time
# from MyNet import Net
from ImplicitNet import ImplicitNet


# data_path = '/home/qg/data/datasets/MNIST/'
data_path = 'D:/datasets/MNIST/'

n_epocks = 4
batch_size_train = 32
batch_size_test = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1

torch.manual_seed(random_seed)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,),(0.3081,))])
train_set = torchvision.datasets.MNIST(data_path + 'train/', train=True, download=True, transform=transform)
val_set = torchvision.datasets.MNIST(data_path + 'validation/', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_test, shuffle=True)

data_iter = iter(train_loader)
images, labels = data_iter.next()
print(images.shape)
print(labels.shape)

criterion = nn.NLLLoss()



# from enum import Enum
# class Layers(Enum):
#     OptNet = 5
#     OptNet_imp_fun = 7

model = ImplicitNet(whichlayer=7)
print(model)


# -------------------------------   train   -------------------------------


optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
model.train()
for e in range(n_epocks):
    running_loss = 0
    nProcessed = 0
    nTrain = len(train_loader.dataset)
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # images = images.view(images.shape[0], -1)
        # images, labels = images.cuda(), labels.cuda()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        nProcessed += len(images)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(labels).sum()
        err = 100. * incorrect / len(images)
        partialEpoch = e + batch_idx / len(train_loader) - 1
        average_loss = running_loss/nProcessed
        if batch_idx % 10 == 1:
            print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tAverage_Loss: {:.6f}\tError: {:.6f}'.format(
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
                average_loss, err))
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
torch.save(model, './my_mnist_model.pt')

# test
# model = torch.load('./my_mnist_model.pt')

# images, labels = next(iter(val_loader))
# img = images[0].view(1, 784)
# with torch.no_grad():
#     logps = model(img)
# ps = torch.exp(logps)
# probab = list(ps.numpy()[0])
# print('predicted digit = ', probab.index(max(probab)))
# plt.imshow(img.view(28,-1))
# plt.show()

correct_count, all_count = 0,0
model.eval()
for images, labels in val_loader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1
print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))