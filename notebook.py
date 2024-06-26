#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


get_ipython().system('pip install seaborn')


# In[2]:


import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


def fileCounter(directory: str):
    lst = os.listdir(directory)
    fileCount = len(lst)
    
    return fileCount


# In[4]:


trainingPizzaPath = 'pizza-notpizza/train/pizza'
trainingNotPizzaPath = 'pizza-notpizza/train/not_pizza'

validationPizzaPath = 'pizza-notpizza/val/pizza'
validationNotPizzaPath = 'pizza-notpizza/val/not_pizza'

testingPizzaPath = 'pizza-notpizza/test/pizza'
testingNotPizzaPath = 'pizza-notpizza/test/not_pizza'


# In[5]:


trainPizzaCount = fileCounter(trainingPizzaPath)
trainNotPizzaCount = fileCounter(trainingNotPizzaPath)

training = [trainPizzaCount, trainNotPizzaCount]
training


# In[6]:


valPizzaCount = fileCounter(validationPizzaPath)
valNotPizzaCount = fileCounter(validationNotPizzaPath)

validation = [valPizzaCount, valNotPizzaCount]
validation


# In[7]:


testPizzaCount = fileCounter(testingPizzaPath)
testNotPizzaCount = fileCounter(testingNotPizzaPath)

testing = [testPizzaCount, testNotPizzaCount]
testing


# In[8]:


classes = ['Pizza', 'Not Pizza']
classesCount = {
    'Training Data': training,
    'Validation Data': validation,
    'Testing Data': testing
}

x = np.arange(len(classes))
width = 0.33
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for dataset, eachClassCount in classesCount.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, eachClassCount, width, label = dataset)
    ax.bar_label(rects, padding=3)
    multiplier += 1
    
ax.set_ylabel('Quantity')
ax.set_ylabel('Class')
ax.set_title('Pizza or Not Pizza by Dataset')
ax.set_xticks(x + width, classes)
ax.legend(loc = 'center', ncols = 3)
ax.set_ylim(0, 900)

plt.show()


# # First attepmt to train
# - Lasso Regularization
# - CNN Network
# - 3 x 256 x 256 input size

# In[9]:


import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as F
from torchvision.transforms import v2

import torchmetrics as metrics

from sklearn.metrics import confusion_matrix, classification_report


# In[10]:


trainData = 'pizza-notpizza/train'
valData = 'pizza-notpizza/val'
testData = 'pizza-notpizza/test'


# In[11]:


trainTransforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

valTestTransforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

trainDataset = datasets.ImageFolder(trainData, transform=trainTransforms)
valDataset = datasets.ImageFolder(valData, transform=valTestTransforms)
testDataset = datasets.ImageFolder(testData, transform=valTestTransforms)


# In[12]:


class_to_idx = trainDataset.class_to_idx

for label, value in class_to_idx.items():
    print(f'Class Name: {label}, Numeric Value: {value}')


# In[13]:


trainDataset[3]


# In[14]:


BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 1e-5


# In[15]:


trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valLoader = torch.utils.data.DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


# In[15]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.dropout = nn.Dropout(0.6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.pool(nn.functional.leaky_relu(self.conv1(x)))
        x = self.pool(nn.functional.leaky_relu(self.conv2(x)))
        x = self.pool(nn.functional.leaky_relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        x = nn.functional.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# In[16]:


model = CNN()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

trainLossHistory = []
trainAccHistory = []

for epoch in range(50):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 1000 == 999:
            print(f'[{epoch + 1}, {i + 1} Loss: {running_loss / 1000}]')
            running_loss = 0.0
            
    epoch_loss = running_loss / len(trainLoader)
    epoch_acc = 100 * correct / total
    trainLossHistory.append(epoch_loss)
    trainAccHistory.append(epoch_acc)
    
    print(f'=== EPOCH [{epoch + 1}]===\n    Loss: {epoch_loss} Accuracy: {epoch_acc}')
    
print('FINISHED TRAINING!')


# In[17]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1, len(trainLossHistory) + 1), trainLossHistory, color='green', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.suptitle('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(trainAccHistory) + 1), trainAccHistory, color='pink', label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.suptitle('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[19]:


model.eval()
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for data in valLoader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
acc = 100 * correct / total

print('Accuracy of the network on the test images: %d %%' % acc)


# In[22]:


conf_matrix = confusion_matrix(true_labels, predicted_labels)

class_names = [str(i) for i in range(2)]

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('CNN with Unbalanced Data vol. 1')
plt.savefig('conf_matrix_cnn1.png')

plt.show()


# In[23]:


report = classification_report(true_labels, predicted_labels)
print(report)


# In[25]:


f1 = metrics.classification.BinaryF1Score()
f1_score = f1(torch.tensor(predicted_labels), torch.tensor(true_labels))
prec = metrics.classification.BinaryPrecision()
precision = prec(torch.tensor(predicted_labels), torch.tensor(true_labels))
rec = metrics.classification.BinaryRecall()
recall = rec(torch.tensor(predicted_labels), torch.tensor(true_labels))

print(f'F1-Score: {f1_score * 100}, Precision: {precision * 100}, Recall: {recall * 100}')


# ## EfficientNet + Data Augmentation
# Data Augmentations:
# - Flips (Horizontal and Vertical)
# - Rotation
# - Color Gitter
# 
# Additionaly:
# - Normalization

# In[16]:


from torchvision.models import efficientnet_b0


# In[ ]:


trainTransforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

trainDataset = datasets.ImageFolder(trainData, transform=trainTransforms)


# In[ ]:


trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


# In[19]:


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model = efficientnet_b0(pretrained = True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.model.classifier[1].in_features, 1)  # Adjusted for binary classification
        )
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.model(x)
        x = self.leaky_relu(x)
        
        return x


# In[20]:


model = EfficientNet()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

trainLossHistory = []
trainAccHistory = []

for epoch in range(50):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        labels = torch.tensor(labels, dtype=torch.float32)
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 1000 == 999:
            print(f'[{epoch + 1}, {i + 1} Loss: {running_loss / 1000}]')
            running_loss = 0.0
            
    epoch_loss = running_loss / len(trainLoader)
    epoch_acc = 100 * correct / total
    trainLossHistory.append(epoch_loss)
    trainAccHistory.append(epoch_acc)
    
    print(f'=== EPOCH [{epoch + 1}]===\n    Loss: {epoch_loss} Accuracy: {epoch_acc}')
    
print('FINISHED TRAINING!')


# In[21]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1, len(trainLossHistory) + 1), trainLossHistory, color='green', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.suptitle('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(trainAccHistory) + 1), trainAccHistory, color='pink', label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.suptitle('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[22]:


model.eval()
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for data in valLoader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
acc = 100 * correct / total

print('Accuracy of the network on the test images: %d %%' % acc)


# In[24]:


conf_matrix = confusion_matrix(true_labels, predicted_labels)

class_names = [str(i) for i in range(2)]

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('EfficientNet')

plt.show()


# In[25]:


report = classification_report(true_labels, predicted_labels)
print(report)


# In[26]:


f1 = metrics.classification.BinaryF1Score()
f1_score = f1(torch.tensor(predicted_labels), torch.tensor(true_labels))
prec = metrics.classification.BinaryPrecision()
precision = prec(torch.tensor(predicted_labels), torch.tensor(true_labels))
rec = metrics.classification.BinaryRecall()
recall = rec(torch.tensor(predicted_labels), torch.tensor(true_labels))

print(f'F1-Score: {f1_score * 100}, Precision: {precision * 100}, Recall: {recall * 100}')


# # ResNet18 + Data Augmentation

# In[16]:


from torchvision.models import resnet18


# In[17]:


class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        self.resnet18= resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet18(x)


# In[18]:


net = ResNet18()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_loss_history = []
train_acc_history = []

for epoch in range(50):
    net.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 1000 == 999:
            print('[%d, %d] loss: %.3f' %
                 (epoch +1, i+1, running_loss / 1000))
            running_loss = 0.0
            
    epoch_loss = running_loss / len(trainLoader)
    epoch_acc = 100 * correct / total
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)
    
    print('=== EPOCH [%d] ===\n    LOSS: %.3f, ACCURACY: %.3f %%' % (epoch+1, epoch_loss, epoch_acc))

print('======\nFINISHED TRAINING\n======')


# In[19]:


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, color='purple', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.suptitle('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, color='blue', label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.suptitle('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('ResNet50_lp.png')
plt.show()


# In[20]:


net.eval()

correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for data in valLoader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
acc = 100 * correct / total

print('Accuracy of the network on the test images: %d %%' % acc)


# In[23]:


conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_names = [str(i) for i in range(2)]

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('ResNet18 Confusion Matrix')
plt.savefig('resnet18confmatrix.png')

plt.show()


# In[25]:


report = classification_report(true_labels, predicted_labels)

print(report)


# In[27]:


f1 = metrics.F1Score(task='multiclass', num_classes=4)
f1 = f1(torch.tensor(predicted_labels), torch.tensor(true_labels)) * 100

prec = metrics.Precision(task='multiclass', average='macro', num_classes=4)
prec = prec(torch.tensor(predicted_labels), torch.tensor(true_labels)) * 100

recall = metrics.Recall(task='multiclass', average='macro', num_classes=4)
recall = recall(torch.tensor(predicted_labels), torch.tensor(true_labels)) * 100

print(f'F1 Score: {f1}, Precision: {prec}, Recall: {recall}')


# # ResNet50 + ImageNet Weights

# In[32]:


from torchvision import models
from torchvision.models import resnet50


# In[33]:


model = resnet50(weights = models.ResNet50_Weights.DEFAULT)
model.conv1 = nn.Conv2d(3, 64, 2, 3, bias=False)
model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# In[37]:


epochLossList = []
epochAccList = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(20):
    model.train()
    
    epochLoss = 0.0
    epochACC = 0.0
    step = 0
    
    for i, data in enumerate(trainLoader):
        img, label = data
        img, label = img.to(device), label.to(device)
        
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            model_out = model(img)
            loss = loss_fn(model_out, label)
            _, pred = torch.max(model_out, 1)
            
            loss.backward()
            optimizer.step()
            
            epochACC += torch.sum(pred == label.data)
            epochLoss += loss.item() * len(model_out)
            
        step += 1
        
    data_size = len(trainLoader.dataset)
    epochLoss = epochLoss / data_size
    epochACC = epochACC.double() / data_size * 100
    epochLossList.append(epochLoss)
    epochAccList.append(epochACC)
    print('=== EPOCH [%d] ===\n    LOSS: %.3f, ACCURACY: %.3f %%' % (epoch+1, epoch_loss, epoch_acc))


# In[43]:


def to_numpy(element):
    if torch.is_tensor(element):
        return element.cpu().numpy()
    else:
        return np.array(element)


# In[44]:


epochLossList = np.array([to_numpy(element) for element in epochLossList])
epochAccList = np.array([to_numpy(element) for element in epochAccList])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(epochLossList) + 1), epochLossList, color='purple', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.suptitle('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(epochAccList) + 1), epochAccList, color='blue', label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.suptitle('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('ResNet50_lp.png')
plt.show()


# In[45]:


model.eval()

correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for data in valLoader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
acc = 100 * correct / total

print('Accuracy of the network on the test images: %d %%' % acc)


# In[46]:


conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_names = [str(i) for i in range(2)]

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('ResNet50 Confusion Matrix')
plt.savefig('resnet50confmatrix.png')

plt.show()


# In[47]:


report = classification_report(true_labels, predicted_labels)

print(report)


# In[48]:


f1 = metrics.F1Score(task='multiclass', num_classes=4)
f1 = f1(torch.tensor(predicted_labels), torch.tensor(true_labels)) * 100

prec = metrics.Precision(task='multiclass', average='macro', num_classes=4)
prec = prec(torch.tensor(predicted_labels), torch.tensor(true_labels)) * 100

recall = metrics.Recall(task='multiclass', average='macro', num_classes=4)
recall = recall(torch.tensor(predicted_labels), torch.tensor(true_labels)) * 100

print(f'F1 Score: {f1}, Precision: {prec}, Recall: {recall}')


# In[ ]:




