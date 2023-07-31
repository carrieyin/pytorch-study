import torch
import random
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class MyData(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyData, self).__init__()
        file_txt = open(datatxt,'r')
        imgs = []
        for line in file_txt:
            line = line.rstrip()
            words = line.split('|')
            imgs.append((words[0], words[1]))

        self.imgs = imgs
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        random.shuffle(self.imgs)
        name, label = self.imgs[index]
        img = Image.open(self.root + name).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = int(label)
        #label_tensor = torch.Tensor([0,0])
        #label_tensor[label]=1
        return img, label

    def __len__(self):
        return len(self.imgs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d((2, 2))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(36*36*32, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = x.view(-1, 36*36*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

classes = ['猫','狗']
transform = transforms.Compose(
    [transforms.Resize((152, 152)),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data=MyData(root ='C:/Users/Elegantmadman/Desktop/data/cats_and_dogs_filtered/train/',
                  datatxt='C:/Users/Elegantmadman/Desktop/data/cats_and_dogs_filtered/'+'6.txt',
                  transform=transform)
test_data=MyData(root ='C:/Users/Elegantmadman/Desktop/data/cats_and_dogs_filtered/validation/',
                 datatxt='C:/Users/Elegantmadman/Desktop/data/cats_and_dogs_filtered/'+'7.txt',
                 transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(labels)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

PATH = 'C:/Users/Elegantmadman/Desktop//cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


